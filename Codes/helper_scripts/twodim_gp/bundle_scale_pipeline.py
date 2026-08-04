#!/usr/bin/env python3
"""Spectrogram bundle correction: time bundles, intra-bundle χ² scaling, optional GP+synth calibration.

Stages
--------
1. Cluster **spectroscopic epochs** (unique normalized phases) whose physical times lie
   within ``--max-bundle-minutes`` (default 5); photometry is never clustered.
2. **Within each spectroscopic epoch**, split disjoint λ arms (gap rule: ``--arm-gap-factor``,
   ``--arm-min-gap-norm``) and align them with the same overlap χ² + seam band logic as between
   epochs (requires a wavelength gap in ``x₁``; uniform per-order sampling has no such gap).
3. Within each time bundle, align **epochs** with a **forward-only Kruskal MST** in median-λ
   order (edges bluer→redder): each tree link uses overlap χ² + seam, or a **gap-midpoint**
   seam when λ ranges do not overlap (same ±``--seam-fit-half-width-aa`` Å affine bands).
   Gap scales use a line-based seam fit; optional ``--gap-veto-min-rel-gain`` /
   ``--gap-veto-edge-err-ratio`` re-enable conservative rejection (default: apply the fit scale).
4. **Photometric anchoring (default on):** with **``--global-scale-iters``** default **1**,
   global scales bring spectra toward photometry while preserving intra-bundle relative scaling.
   If **both** an **enrich** npz (``--enrich`` or auto-discovered beside ``-i``) **and** a **filter YAML**
   are available, the pipeline runs **``run_gp``** (unless you already passed ``--run-gp``) and uses
   synthetic broadband flux vs a **per-band smoother** of ``µ_train``. **Otherwise** (no enrich, no
   filter, or either missing) it uses **rough / pooled χ²** photometry anchoring from training phot
   rows alone — **no enrich is required**. Use **``--skip-global-phot-anchor``** or
   **``--global-scale-iters 0``** for relative-only output.

Output ``*.npz`` matches the collaborator layout with added ``spec_bundle_id`` (‑1 = phot)
and preserves all other arrays. A JSON sidecar **``_<stem>_scale_report.json``** records
applied scales.

Examples::

    # Relative spectroscopy scaling only (no photometry anchor):
    python bundle_scale_pipeline.py -i gp_minimal_bundle.npz -o gp_scaled.npz \\
        --skip-global-phot-anchor

    # Default: phot anchor without enrich (rough / pooled χ²); filter YAML may still auto-resolve:
    python bundle_scale_pipeline.py -i gp_work.npz -o gp_work_scaled.npz \\
        --runs-dir runs --gp-tag-prefix bscale

    python bundle_scale_pipeline.py -i gp_minimal_bundle.npz -o gp_full.npz \\
        --run-gp --runs-dir runs --gp-tag-prefix bscale \\
        --filter-config configs/filter_pipeline.example.yaml --enrich enrich.npz \\
        --global-scale-iters 2 -- \\
        --max-iter 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar

import bundle_meta as bmeta
import bundle_preprocess as bpre
import gp_utils as gu
import spectrum_bundles as sb
from filter_synthesis import load_filter_config, synthesize_for_bands
from plot_results import linear_flux_yerr, phase_days_from_norm_x2, scaled_ln_to_linear, scatter_train_vector_to_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND_PHASE = 9


def _abspath_file(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p).strip()))


def discover_enrich_npz(input_npz: str, explicit: Optional[str]) -> Optional[str]:
    """Return path to an existing enrich npz, or None.

    If ``explicit`` is set, only that path is tried (caller must verify it exists).
    Otherwise tries ``<input_dir>/<input_stem>_enrich.npz`` then ``<input_dir>/enrich.npz``.
    """
    if explicit is not None and str(explicit).strip():
        return _abspath_file(str(explicit))
    inp = os.path.abspath(os.path.expanduser(str(input_npz).strip()))
    d = os.path.dirname(inp)
    stem = os.path.splitext(os.path.basename(inp))[0]
    for cand in (
        os.path.join(d, f"{stem}_enrich.npz"),
        os.path.join(d, "enrich.npz"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def discover_filter_config_yaml(explicit: Optional[str], repo_root: str = HERE) -> Optional[str]:
    """Return path to an existing filter YAML, or None.

    If ``explicit`` is set, only that path is tried. Otherwise tries
    ``configs/filter_pipeline.yaml`` then ``configs/filter_pipeline.example.yaml`` under
    ``repo_root``.
    """
    if explicit is not None and str(explicit).strip():
        return _abspath_file(str(explicit))
    for cand in (
        os.path.join(repo_root, "configs", "filter_pipeline.yaml"),
        os.path.join(repo_root, "configs", "filter_pipeline.example.yaml"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def wavelength_aa_from_x1_norm(x1_norm: np.ndarray, gn: dict) -> np.ndarray:
    """Convert normalized log10-wavelength (X[:,0]) to linear wavelength in Å."""
    u = np.asarray(x1_norm, dtype=float).ravel()
    if gn.get("_normalized_only"):
        return u
    wl_log_phys = float(gn["x1_mean"]) + float(gn["x1_std"]) * u
    return np.power(10.0, wl_log_phys).astype(float)


def _robust_weighted_log_scale(log_r: np.ndarray, w: np.ndarray, clip: float) -> float:
    """Robust weighted mean of log ratios with symmetric clipping."""
    lr = np.asarray(log_r, dtype=float).ravel()
    ww = np.asarray(w, dtype=float).ravel()
    ok = np.isfinite(lr) & np.isfinite(ww) & (ww > 0)
    lr = lr[ok]
    ww = ww[ok]
    if lr.size == 0:
        return 0.0
    mu = float(np.average(lr, weights=ww))
    if not np.isfinite(mu):
        return 0.0
    if clip is not None and float(clip) > 0:
        c = float(clip)
        lr = np.clip(lr, mu - c, mu + c)
    return float(np.average(lr, weights=ww))


def estimate_epoch_log_scale_from_phot_wavelength_points(
    *,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    phot_mask: np.ndarray,
    ph_epoch: float,
    wl_spec: np.ndarray,
    fl_spec: np.ndarray,
    phot_phase_window_norm: float,
    delta_log_clip: float = 1.25,
    min_points: int = 1,
) -> tuple[float, dict[str, Any]]:
    """Estimate one log scale using nearby-phase phot points at their effective wavelengths.

    This is a rough fallback when band labels / filter synthesis are unavailable:
    choose photometry within ±window in normalized phase, interpolate the epoch spectrum
    to each phot point's wavelength, and fit a single multiplicative scale in log space.
    """
    meta: dict[str, Any] = {"applied": False, "n_used": 0}
    if wl_spec.size < 1:
        meta["reason"] = "empty_spectrum"
        return 0.0, meta
    wm = np.asarray(wl_spec, dtype=float).ravel()
    fm = np.asarray(fl_spec, dtype=float).ravel()
    ok_spec = np.isfinite(wm) & np.isfinite(fm) & (wm > 0) & (fm > 0)
    wm = wm[ok_spec]
    fm = fm[ok_spec]
    if wm.size < 1:
        meta["reason"] = "nonfinite_spectrum"
        return 0.0, meta
    if wm.size >= 2:
        o = np.argsort(wm)
        wm = wm[o]
        fm = fm[o]

    x2 = np.asarray(X[:, 1], dtype=float).ravel()
    near = phot_mask & np.isfinite(x2) & (np.abs(x2 - float(ph_epoch)) <= float(phot_phase_window_norm))
    idx = np.flatnonzero(near)
    meta["n_phot_near"] = int(idx.size)
    if idx.size < int(min_points):
        meta["reason"] = "too_few_phot_near"
        return 0.0, meta

    wl_ph = wavelength_aa_from_x1_norm(X[idx, 0], gn)
    f_ph = scaled_ln_to_linear(y[idx], gn)
    e_ph = linear_flux_yerr(y[idx], yerr[idx], gn)
    ok_ph = np.isfinite(wl_ph) & np.isfinite(f_ph) & np.isfinite(e_ph) & (wl_ph > 0) & (f_ph > 0) & (e_ph > 0)
    wl_ph = np.asarray(wl_ph, dtype=float)[ok_ph]
    f_ph = np.asarray(f_ph, dtype=float)[ok_ph]
    e_ph = np.asarray(e_ph, dtype=float)[ok_ph]
    if wl_ph.size < int(min_points):
        meta["reason"] = "too_few_phot_valid"
        return 0.0, meta

    if wm.size == 1:
        f_spec_i = np.full(wl_ph.shape[0], float(fm[0]), dtype=float)
        meta["n_phot_in_range"] = int(wl_ph.size)
    else:
        in_range = (wl_ph >= wm[0]) & (wl_ph <= wm[-1])
        wl_ph = wl_ph[in_range]
        f_ph = f_ph[in_range]
        e_ph = e_ph[in_range]
        meta["n_phot_in_range"] = int(wl_ph.size)
        if wl_ph.size < int(min_points):
            meta["reason"] = "too_few_phot_in_wl_range"
            return 0.0, meta
        f_spec_i = np.interp(wl_ph, wm, fm)
    ok_i = np.isfinite(f_spec_i) & (f_spec_i > 0)
    wl_ph = wl_ph[ok_i]
    f_ph = f_ph[ok_i]
    e_ph = e_ph[ok_i]
    f_spec_i = f_spec_i[ok_i]
    if wl_ph.size < int(min_points):
        meta["reason"] = "too_few_interp"
        return 0.0, meta

    log_r = np.log(f_ph) - np.log(f_spec_i)
    w = (e_ph ** -2)
    dlog = _robust_weighted_log_scale(log_r, w, clip=float(delta_log_clip))
    dlog = float(np.clip(dlog, -float(delta_log_clip), float(delta_log_clip)))
    meta.update({"applied": True, "n_used": int(wl_ph.size), "delta_log": float(dlog)})
    return dlog, meta


def mirror_bundle_meta_from_input(src_bundle_npz: str, dst_bundle_npz: str) -> Optional[str]:
    """Copy collaborator ``*_meta.json`` beside ``dst_bundle_npz`` from the input bundle stem.

    Updates ``files.npz`` / ``files.meta`` in the JSON to match the destination basename so
    ``run_gp`` / plotters resolve ``grid_norm_info`` next to scaled bundles.
    """
    src_npz = os.path.abspath(os.path.expanduser(src_bundle_npz))
    dst_npz = os.path.abspath(os.path.expanduser(dst_bundle_npz))
    src_meta = bmeta.bundle_meta_json_path(src_npz)
    if not os.path.isfile(src_meta):
        return None
    dst_meta = bmeta.bundle_meta_json_path(dst_npz)
    with open(src_meta, encoding="utf-8") as f:
        meta = json.load(f)
    base_npz = os.path.basename(dst_npz)
    base_meta = os.path.basename(dst_meta)
    if isinstance(meta.get("files"), dict):
        meta["files"] = dict(meta["files"])
        meta["files"]["npz"] = base_npz
        meta["files"]["meta"] = base_meta

    dm = os.path.dirname(dst_meta)
    if dm:
        os.makedirs(dm, exist_ok=True)
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.flush()
    return dst_meta


def load_grid_norm(bundle_path: str, meta_override: Optional[str]) -> dict:
    meta_path = meta_override
    if not meta_path:
        babs = os.path.abspath(os.path.expanduser(str(bundle_path)))
        bdir = os.path.dirname(babs)
        bname = os.path.splitext(os.path.basename(babs))[0]
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(bdir, f"{bname}_meta.json"),
            os.path.join(here, f"{bname}_meta.json"),
            os.path.join(here, "gp_scaled_bundle_meta.json"),
            os.path.join(here, "gp_minimal_bundle_meta.json"),
            os.path.join(here, "gp_bundle_corrected_meta.json"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                meta_path = cand
                break
    return bmeta.grid_norm_from_bundle_or_meta(bundle_path, meta_path=meta_path)


def unique_spec_epochs(
    X: np.ndarray,
    spec_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = X.shape[0]
    epoch_of_row = np.full(n_train, -1, dtype=np.int32)
    rows = np.where(spec_mask)[0]
    if rows.size == 0:
        return np.zeros(0, dtype=float), epoch_of_row

    rnd_keys, inv_rows = np.unique(np.round(X[rows, 1], ROUND_PHASE), return_inverse=True)
    n_eps = rnd_keys.size
    canonical_phases = np.zeros(n_eps, dtype=float)
    epoch_row_lists: list[list[int]] = []
    for e in range(n_eps):
        sel = rows[inv_rows == e]
        canonical_phases[e] = float(np.median(X[sel, 1]))
        epoch_row_lists.append(list(sel.astype(int).tolist()))

    for e, rl in enumerate(epoch_row_lists):
        for r in rl:
            epoch_of_row[int(r)] = int(e)

    return canonical_phases, epoch_of_row


def canon_phase(canonical_phases: np.ndarray, epoch_id: int) -> float:
    return float(canonical_phases[int(epoch_id)])


def time_per_epoch(
    X: np.ndarray,
    canonical_phases: np.ndarray,
    gn: dict,
    enrich: Optional[dict[str, np.ndarray]],
) -> np.ndarray:
    mj_raw = enrich.get("mjd") if enrich else None
    mj_arr = np.asarray(mj_raw, dtype=float).ravel() if mj_raw is not None else None
    times = np.zeros(canonical_phases.size, dtype=float)
    for e in range(canonical_phases.size):
        ph = canonical_phases[e]
        mrows = (
            np.isfinite(X[:, 0])
            & np.isfinite(X[:, 1])
            & np.isclose(X[:, 1], ph, rtol=0.0, atol=10 ** (-(ROUND_PHASE - 1)))
        )
        idx = np.flatnonzero(mrows)
        if mj_arr is not None and mj_arr.size >= max(1, X.shape[0] - 1) and idx.size:
            mj_sub = mj_arr[np.minimum(idx, mj_arr.size - 1)]
            mj_sub = mj_sub[np.isfinite(mj_sub)]
            if mj_sub.size:
                times[e] = float(np.median(mj_sub))
                continue
        times[e] = float(phase_days_from_norm_x2(np.array([ph], dtype=float), gn)[0])
    return times


def composite_epoch_linear(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    phase_canon: float,
    *,
    phase_atol: float,
    epoch_of_row: Optional[np.ndarray] = None,
    epoch_id: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wavelength-coordinate, linear flux, linear σ.

    Rows are selected by ``round(x₂, ROUND_PHASE)`` so this matches ``unique_spec_epochs``.
    A loose ``phase_atol`` would merge nearly-coincident spectroscopic exposures (Δx₂ ~1e-6)
    and break intra-bundle χ² scaling.

    When ``epoch_of_row`` and ``epoch_id`` are both set, only rows with
    ``epoch_of_row[row] == epoch_id`` are used (spectroscopic epoch index from
    ``unique_spec_epochs``).  This **excludes photometry** and any other rows that
    might share the same rounded phase key but belong to a different epoch or class.
    In standard mode (with grid normalization metadata), wavelength output is linear λ
    from ``10 ** log10(λ_phys)`` so seam widths are in physical units.

    In ``_normalized_only`` mode, keep the original spectral coordinate ``X[:,0]`` for
    scaling/diagnostics rather than remapping through ``10 **``.
    """
    _ = float(phase_atol)  # kept for API compatibility with callers
    if (epoch_of_row is None) ^ (epoch_id is None):
        raise ValueError("composite_epoch_linear: pass both epoch_of_row and epoch_id, or neither")
    key = float(np.round(float(phase_canon), int(ROUND_PHASE)))
    mask = (
        np.isfinite(X[:, 0])
        & np.isfinite(y)
        & np.isfinite(X[:, 1])
        & (np.round(X[:, 1], int(ROUND_PHASE)) == key)
    )
    if epoch_of_row is not None:
        eor = np.asarray(epoch_of_row, dtype=np.int32).ravel()
        if eor.shape[0] != X.shape[0]:
            raise ValueError("epoch_of_row must match X row count")
        mask &= eor == int(epoch_id)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    u = np.asarray(X[idx, 0], dtype=float)
    if gn.get("_normalized_only"):
        wav = np.asarray(u, dtype=float).ravel()
    else:
        wl_log_phys = float(gn["x1_mean"]) + float(gn["x1_std"]) * u
        wav = np.power(10.0, wl_log_phys).ravel().astype(float)
    f_lin = scaled_ln_to_linear(y[idx], gn)
    e_lin = linear_flux_yerr(y[idx], yerr[idx], gn)
    o = np.argsort(wav)
    return wav[o], np.asarray(f_lin, dtype=float).ravel()[o], np.asarray(e_lin, dtype=float).ravel()[o]


def ab_mag_to_flam(mag_ab: float, wav_angstrom: float) -> float:
    wav_cm = float(max(wav_angstrom, 1.0)) * 1e-8
    c_hz_cm = 2.99792458e10
    fnu = np.power(10.0, -0.4 * (mag_ab + 48.60))
    return float(fnu * c_hz_cm / (wav_cm**2))


def affine_flux_seam_band_aa(
    w_aa: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
    wl_seam: float,
    *,
    half_width_aa: float,
    take_blueward: bool,
    min_points: int = 3,
) -> tuple[float, float]:
    """Weighted linear flux vs λ in a physical band touching ``wl_seam`` (Å).

    ``take_blueward``: use ``[wl_seam - half_width, wl_seam]`` (bluer arm toward the join);
    else ``[wl_seam, wl_seam + half_width]`` (redder arm).  Stabilizes noisy edge pixels vs a
    handful of pixels at the array end.
    """
    ok = np.isfinite(w_aa) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    wl, ff, ee = w_aa[ok], flux[ok], ferr[ok]
    if wl.size == 0:
        return np.nan, np.nan
    hw = max(float(half_width_aa), 1.0)
    if take_blueward:
        m = (wl <= wl_seam) & (wl >= wl_seam - hw)
    else:
        m = (wl >= wl_seam) & (wl <= wl_seam + hw)
    if int(np.sum(m)) < min_points:
        return affine_flux_prediction_at_lambda(
            w_aa,
            flux,
            ferr,
            wl_seam,
            n_pix_edge=min(120, max(3, w_aa.size // 2)),
            side="left" if take_blueward else "right",
        )
    wx = wl[m]
    fy = ff[m]
    ferr_l = ee[m]
    wmean = float(np.mean(wx))
    mat = np.column_stack([np.ones_like(wx), wx - wmean])
    wgt = ferr_l**-2
    mtw = mat * wgt[:, np.newaxis]
    cov = np.linalg.inv(mat.T @ mtw + 1e-15 * np.eye(2))
    coef = cov @ (mat.T @ (wgt * fy))
    pv = np.array([1.0, float(wl_seam - wmean)], dtype=float)
    mu_hat = float(pv @ coef)
    sig_hat = float(np.sqrt(max(float(pv @ cov @ pv), 1e-30)))
    return mu_hat, sig_hat


def affine_flux_prediction_at_lambda(
    w_aa: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
    lambda_seam: float,
    *,
    n_pix_edge: int,
    side: str,
) -> tuple[float, float]:
    ok = np.isfinite(w_aa) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    wl, ff, ee = w_aa[ok], flux[ok], ferr[ok]
    if wl.size == 0:
        return np.nan, np.nan
    ordv = np.argsort(wl)
    wl, ff, ee = wl[ordv], ff[ordv], ee[ordv]
    k = max(3, min(int(n_pix_edge), wl.size))
    sl = slice(0, k) if side == "left" else slice(-k, None)
    wx, fy, ferr_l = wl[sl], ff[sl], ee[sl]
    wmean = float(np.mean(wx))
    mat = np.column_stack([np.ones_like(wx), wx - wmean])
    wgt = ferr_l**-2
    mtw = mat * wgt[:, np.newaxis]
    cov = np.linalg.inv(mat.T @ mtw + 1e-15 * np.eye(2))
    coef = cov @ (mat.T @ (wgt * fy))
    pv = np.array([1.0, float(lambda_seam - wmean)], dtype=float)
    mu_hat = float(pv @ coef)
    sig_hat = float(np.sqrt(max(float(pv @ cov @ pv), 1e-30)))
    return mu_hat, sig_hat


def chi2_overlap_seam(
    scale: float,
    wl_ref: np.ndarray,
    f_ref: np.ndarray,
    e_ref: np.ndarray,
    wl_mov: np.ndarray,
    f_mov: np.ndarray,
    e_mov: np.ndarray,
    *,
    wl_seam: float,
    mu_ref_hat: float,
    sig_ref_hat: float,
    seam_weight: float,
    n_grid: int,
    overlap_half_width_aa: float,
    mu_mov_hat: float,
    sig_mov_hat: float,
) -> float:
    lo = max(float(np.min(wl_ref)), float(np.min(wl_mov)))
    hi = min(float(np.max(wl_ref)), float(np.max(wl_mov)))
    chi = 0.0
    if hi > lo and n_grid >= 8:
        hw = max(float(overlap_half_width_aa), 1.0)
        lo2 = max(lo, float(wl_seam) - hw)
        hi2 = min(hi, float(wl_seam) + hw)
        if hi2 <= lo2 + 1e-12:
            lo2, hi2 = lo, hi
        grid = np.linspace(lo2, hi2, n_grid)
        r = np.interp(grid, wl_ref, f_ref, left=np.nan, right=np.nan)
        rr = np.interp(grid, wl_ref, e_ref, left=np.nan, right=np.nan)
        v = np.interp(grid, wl_mov, f_mov, left=np.nan, right=np.nan)
        vv = np.interp(grid, wl_mov, e_mov, left=np.nan, right=np.nan)
        ok = np.isfinite(r) & np.isfinite(v) & np.isfinite(rr) & np.isfinite(vv)
        ok &= vv > 0
        ok &= rr > 0
        if ok.sum() >= max(8, n_grid // 5):
            d = r[ok] - scale * v[ok]
            vr = rr[ok] ** 2 + (scale * vv[ok]) ** 2
            chi += float(np.sum(d**2 / np.maximum(vr, 1e-30)))

    # Prevent near-singular affine covariance from making the seam constraint ineffective.
    sig_ref_eff = max(float(sig_ref_hat), 0.05 * abs(float(mu_ref_hat)), 1e-30)
    sig_mov_eff = max(float(sig_mov_hat), 0.05 * abs(float(mu_mov_hat)), 1e-30)
    den = sig_ref_eff**2 + (scale * sig_mov_eff) ** 2 + 1e-30
    chi += seam_weight * (mu_ref_hat - scale * mu_mov_hat) ** 2 / den
    return chi


def _scale_log_prior(scale: float, *, weight: float) -> float:
    """Weak log-scale prior that discourages extreme multipliers.

    This is essential for gap/seam-only solves where the data term can be ill-conditioned
    (e.g. mu_hat≈0), otherwise the optimizer can run to the bounds and create catastrophic
    intra-bundle scales (seen as s=1e4 / m≈1e8 in diagnostics).
    """
    if weight <= 0:
        return 0.0
    s = float(scale)
    if not np.isfinite(s) or s <= 0:
        return 1e30
    return float(weight * (np.log(s) ** 2))


def chi2_seam_line_only(
    scale: float,
    *,
    mu_ref_hat: float,
    sig_ref_hat: float,
    mu_mov_hat: float,
    sig_mov_hat: float,
    seam_weight: float,
) -> float:
    """Seam-only χ² for affine flux predictions at a gap or overlap midpoint (no λ grid)."""
    if not (np.isfinite(mu_ref_hat) and np.isfinite(mu_mov_hat) and np.isfinite(sig_ref_hat) and np.isfinite(sig_mov_hat)):
        return 1e30
    den = sig_ref_hat**2 + (scale * sig_mov_hat) ** 2 + 1e-30
    return float(seam_weight * (mu_ref_hat - scale * mu_mov_hat) ** 2 / den)


def solve_gap_seam_scale(
    wl_ref: np.ndarray,
    f_ref: np.ndarray,
    e_ref: np.ndarray,
    wl_mov: np.ndarray,
    f_mov: np.ndarray,
    e_mov: np.ndarray,
    *,
    seam_weight: float,
    seam_band_half_width_aa: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> float:
    """Linear scale from **gap midpoint** seam when bluer ref and redder mov do not overlap in λ.

    ``wl_seam = 0.5*(max(wl_ref)+min(wl_mov))``; affine flux vs λ in ±``seam_band_half_width_aa`` Å
    on each arm (same geometry as ``solve_pair_scale`` seam bands).
    """
    if wl_ref.size < 2 or wl_mov.size < 2:
        return 1.0
    # Note: ``solve_pair_scale`` routes here when ``hi <= lo`` (no overlap or touching at a point).
    # Do not short-circuit to 1.0 for near-touching arms; those can still require a scale.
    wl_seam = 0.5 * (float(np.max(wl_ref)) + float(np.min(wl_mov)))
    mr = float(np.nanmedian(wl_ref))
    mm = float(np.nanmedian(wl_mov))
    hw = max(float(seam_band_half_width_aa), 1.0)
    prior_w = 0.0

    # --- Gap scaling rule (as requested) ---
    # Fit a weighted line to the *end* of each spectrum (≈50 Å each side),
    # then minimize χ² between the two lines over ~100 Å centered at wl_seam.
    n_grid = 21
    lam_grid = np.linspace(wl_seam - hw, wl_seam + hw, n_grid)

    def _fit_end_line(
        w: np.ndarray,
        f: np.ndarray,
        e: np.ndarray,
        *,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        ww, ff, ee = np.asarray(w[ok], dtype=float), np.asarray(f[ok], dtype=float), np.asarray(e[ok], dtype=float)
        if ww.size < 3:
            raise ValueError("too few points")
        if side == "right":
            edge = float(np.max(ww))
            m = (ww >= edge - hw) & (ww <= edge)
        else:
            edge = float(np.min(ww))
            m = (ww >= edge) & (ww <= edge + hw)
        if int(np.sum(m)) < 3:
            # fallback: use many edge pixels if the strict 50 Å window is sparse
            order = np.argsort(ww)
            ww, ff, ee = ww[order], ff[order], ee[order]
            k = max(3, min(120, ww.size))
            sl = slice(-k, None) if side == "right" else slice(0, k)
            ww, ff, ee = ww[sl], ff[sl], ee[sl]
        else:
            ww, ff, ee = ww[m], ff[m], ee[m]
        wmean = float(np.mean(ww))
        mat = np.column_stack([np.ones_like(ww), ww - wmean])
        wgt = ee**-2
        mtw = mat * wgt[:, np.newaxis]
        cov = np.linalg.inv(mat.T @ mtw + 1e-15 * np.eye(2))
        coef = cov @ (mat.T @ (wgt * ff))
        return coef, cov, wmean

    try:
        coef_ref, cov_ref, wmean_ref = _fit_end_line(wl_ref, f_ref, e_ref, side="right")
        coef_mov, cov_mov, wmean_mov = _fit_end_line(wl_mov, f_mov, e_mov, side="left")
    except Exception:
        # Worst-case fallback: seam-at-midpoint scalar prediction (legacy behavior)
        mh, sh = affine_flux_prediction_at_lambda(
            wl_ref, f_ref, e_ref, wl_seam, n_pix_edge=min(120, max(3, wl_ref.size // 2)), side="right"
        )
        mm_hat, sm_hat = affine_flux_prediction_at_lambda(
            wl_mov, f_mov, e_mov, wl_seam, n_pix_edge=min(120, max(3, wl_mov.size // 2)), side="left"
        )

        def obj(s: float) -> float:
            return chi2_seam_line_only(
                s,
                mu_ref_hat=mh,
                sig_ref_hat=sh,
                mu_mov_hat=mm_hat,
                sig_mov_hat=sm_hat,
                seam_weight=seam_weight,
            ) + _scale_log_prior(s, weight=prior_w)

        ans = minimize_scalar(obj, bounds=(1e-2, 1e2), method="bounded", options={"xatol": 1e-8})
        return float(np.clip(ans.x, 1e-2, 1e2))

    def _line_mu_sig(coef: np.ndarray, cov: np.ndarray, wmean: float, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lam = np.asarray(lam, dtype=float).ravel()
        pv = np.column_stack([np.ones_like(lam), lam - float(wmean)])
        mu = pv @ coef
        # var_i = pv_i^T cov pv_i
        var = np.einsum("...i,ij,...j->...", pv, cov, pv)
        sig = np.sqrt(np.maximum(var, 1e-30))
        # floor so the constraint has bite even when covariance is huge
        sig = np.maximum(sig, 0.05 * np.abs(mu))
        return np.asarray(mu, dtype=float), np.asarray(sig, dtype=float)

    mu_r, sig_r = _line_mu_sig(coef_ref, cov_ref, wmean_ref, lam_grid)
    mu_m, sig_m = _line_mu_sig(coef_mov, cov_mov, wmean_mov, lam_grid)

    # Closed-form weighted least-squares scale over the requested ~100 Å seam window.
    # This directly enforces the line-χ² matching the user asked for, without pulling s toward 1.
    wgt = 1.0 / np.maximum(sig_r**2 + sig_m**2, 1e-30)
    num = float(np.sum(wgt * mu_r * mu_m))
    den = float(np.sum(wgt * mu_m * mu_m))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return 1.0
    s_hat = num / den
    if not np.isfinite(s_hat) or s_hat <= 0:
        return 1.0
    s_hat = float(np.clip(s_hat, 1e-2, 1e2))

    # Edge-pixel seam χ²: line extrapolation can sit near s≈1 while ref/mov levels at the physical
    # gap disagree (visible steps in MST ``mode=gap`` panels). Use **tight** end caps (few pixels at
    # the abutment) so medians track the join, then pick the best of line s_hat, scalar-optimized edge
    # χ², and the direct abutment flux ratio when the line solution is near unity.
    k_ref = max(1, min(5, int(wl_ref.size)))
    k_mov = max(1, min(5, int(wl_mov.size)))
    frk = np.asarray(f_ref[-k_ref:], dtype=float)
    fmk = np.asarray(f_mov[:k_mov], dtype=float)
    erk = np.asarray(e_ref[-k_ref:], dtype=float)
    emk = np.asarray(e_mov[:k_mov], dtype=float)
    ref_edge = float(np.nanmedian(frk))
    mov_edge = float(np.nanmedian(fmk))
    if np.isfinite(ref_edge) and np.isfinite(mov_edge) and ref_edge > 0 and mov_edge > 0:
        sig_re = float(np.sqrt(np.nanmean(erk**2)))
        sig_me = float(np.sqrt(np.nanmean(emk**2)))
        sig_re = max(float(sig_re), 0.05 * abs(ref_edge))
        sig_me = max(float(sig_me), 0.05 * abs(mov_edge))

        def chi_edge(s: float) -> float:
            s = float(s)
            if not np.isfinite(s) or s <= 0:
                return 1e30
            d = ref_edge - s * mov_edge
            v = sig_re**2 + (s * sig_me) ** 2 + 1e-30
            return float((d * d) / v)

        ans_e = minimize_scalar(chi_edge, bounds=(1e-2, 1e2), method="bounded", options={"xatol": 1e-9})
        s_edge = float(np.clip(ans_e.x, 1e-2, 1e2))
        s_geo = float(np.clip(ref_edge / mov_edge, 1e-2, 1e2))

        best_s = float(s_hat)
        best_c = float(chi_edge(best_s))
        for cand in (float(s_edge), float(s_geo)):
            cc = float(chi_edge(cand))
            if cc < best_c * 0.999:
                best_c, best_s = cc, cand

        # If line WLS hugged unity but abutment ratio is not, prefer a scale that fixes the step
        # even when χ² differences are tiny (robust to plateau in chi_edge).
        if abs(float(np.log(max(s_hat, 1e-12)))) < 0.03 and abs(float(np.log(max(s_geo, 1e-12)))) > 0.012:
            if float(chi_edge(s_geo)) <= float(chi_edge(best_s)) * 1.02:
                best_s = float(s_geo)

        s_hat = float(np.clip(best_s, 1e-2, 1e2))

    if prior_w > 0:
        # Optional tiny regularization hook (currently disabled).
        def obj(s: float) -> float:
            s = float(s)
            if not np.isfinite(s) or s <= 0:
                return 1e30
            den2 = sig_r**2 + (s * sig_m) ** 2 + 1e-30
            chi = float(seam_weight) * float(np.sum((mu_r - s * mu_m) ** 2 / den2))
            return chi + _scale_log_prior(s, weight=prior_w)
        ans = minimize_scalar(obj, bounds=(1e-2, 1e2), method="bounded", options={"xatol": 1e-8})
        s_hat = float(np.clip(ans.x, 1e-2, 1e2))

    # Optional legacy vetoes (disabled by default): previously rejected many valid gap scales.
    if gap_veto_min_rel_gain is not None:
        w_wls = 1.0 / np.maximum(sig_r**2 + sig_m**2, 1e-30)
        chi_raw_w = float(np.sum(w_wls * (mu_r - mu_m) ** 2))
        chi_hat_w = float(np.sum(w_wls * (mu_r - s_hat * mu_m) ** 2))
        if not np.isfinite(chi_raw_w) or not np.isfinite(chi_hat_w) or chi_raw_w <= 0:
            return 1.0
        rel_gain = (chi_raw_w - chi_hat_w) / max(chi_raw_w, 1e-30)
        if rel_gain < float(gap_veto_min_rel_gain):
            return 1.0
    if gap_veto_edge_err_ratio is not None:
        k_ref = max(1, min(8, int(wl_ref.size)))
        k_mov = max(1, min(8, int(wl_mov.size)))
        ref_edge = float(np.nanmedian(np.asarray(f_ref, dtype=float)[-k_ref:]))
        mov_edge = float(np.nanmedian(np.asarray(f_mov, dtype=float)[:k_mov]))
        if not (np.isfinite(ref_edge) and np.isfinite(mov_edge) and ref_edge > 0 and mov_edge > 0):
            return 1.0
        raw_err = abs(float(np.log(ref_edge / mov_edge)))
        scaled_err = abs(float(np.log(ref_edge / (s_hat * mov_edge))))
        if scaled_err >= float(gap_veto_edge_err_ratio) * raw_err:
            return 1.0
    return s_hat


def solve_pair_scale(
    wl_ref: np.ndarray,
    f_ref: np.ndarray,
    e_ref: np.ndarray,
    wl_mov: np.ndarray,
    f_mov: np.ndarray,
    e_mov: np.ndarray,
    *,
    seam_weight: float,
    overlap_grid: int,
    seam_band_half_width_aa: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> float:
    if wl_ref.size < 2 or wl_mov.size < 2:
        return 1.0
    lo = max(float(np.min(wl_ref)), float(np.min(wl_mov)))
    hi = min(float(np.max(wl_ref)), float(np.max(wl_mov)))
    if hi <= lo + 1e-9:
        return solve_gap_seam_scale(
            wl_ref,
            f_ref,
            e_ref,
            wl_mov,
            f_mov,
            e_mov,
            seam_weight=seam_weight,
            seam_band_half_width_aa=seam_band_half_width_aa,
            gap_veto_min_rel_gain=gap_veto_min_rel_gain,
            gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
        )
    if wl_ref.size < 3 or wl_mov.size < 3:
        return 1.0
    wl_seam = 0.5 * (lo + hi)
    mr = float(np.nanmedian(wl_ref))
    mm = float(np.nanmedian(wl_mov))
    hw = max(float(seam_band_half_width_aa), 1.0)
    # Bluer spectrum (lower median λ): seam neighborhood is [wl_seam - hw, wl_seam].
    # Redder spectrum: [wl_seam, wl_seam + hw].  (Previously reversed, which biased scales badly.)
    if mr > mm:
        mh, sh = affine_flux_seam_band_aa(
            wl_ref, f_ref, e_ref, wl_seam, half_width_aa=hw, take_blueward=False
        )
        mm_hat, sm_hat = affine_flux_seam_band_aa(
            wl_mov, f_mov, e_mov, wl_seam, half_width_aa=hw, take_blueward=True
        )
    else:
        mh, sh = affine_flux_seam_band_aa(
            wl_ref, f_ref, e_ref, wl_seam, half_width_aa=hw, take_blueward=True
        )
        mm_hat, sm_hat = affine_flux_seam_band_aa(
            wl_mov, f_mov, e_mov, wl_seam, half_width_aa=hw, take_blueward=False
        )

    def obj(s: float) -> float:
        return chi2_overlap_seam(
            s,
            wl_ref,
            f_ref,
            e_ref,
            wl_mov,
            f_mov,
            e_mov,
            wl_seam=wl_seam,
            mu_ref_hat=mh,
            sig_ref_hat=sh,
            seam_weight=seam_weight,
            n_grid=overlap_grid,
            overlap_half_width_aa=hw,
            mu_mov_hat=mm_hat,
            sig_mov_hat=sm_hat,
        ) + _scale_log_prior(s, weight=0.01)

    ans = minimize_scalar(obj, bounds=(1e-2, 1e2), method="bounded", options={"xatol": 1e-8})
    return float(np.clip(ans.x, 1e-2, 1e2))


def _seam_band_interval_aa(wl_seam: float, half_width_aa: float, *, take_blueward: bool) -> tuple[float, float]:
    """Physical Å interval used by ``affine_flux_seam_band_aa`` (inclusive bounds)."""
    hw = max(float(half_width_aa), 1.0)
    if take_blueward:
        return (float(wl_seam - hw), float(wl_seam))
    return (float(wl_seam), float(wl_seam + hw))


def pair_scale_report(
    wl_ref: np.ndarray,
    f_ref: np.ndarray,
    e_ref: np.ndarray,
    wl_mov: np.ndarray,
    f_mov: np.ndarray,
    e_mov: np.ndarray,
    *,
    seam_weight: float,
    overlap_grid: int,
    seam_band_half_width_aa: float,
    scale: Optional[float] = None,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> tuple[float, dict[str, Any]]:
    """Same optimum as ``solve_pair_scale`` plus wavelength regions used (overlap + seam bands).

    Pass ``scale`` when the optimum is already known (avoids a duplicate ``solve_pair_scale``).

    Geometry logic is kept in lockstep with ``solve_pair_scale`` / ``solve_gap_seam_scale``.
    """
    if scale is None:
        s = solve_pair_scale(
            wl_ref,
            f_ref,
            e_ref,
            wl_mov,
            f_mov,
            e_mov,
            seam_weight=seam_weight,
            overlap_grid=overlap_grid,
            seam_band_half_width_aa=seam_band_half_width_aa,
            gap_veto_min_rel_gain=gap_veto_min_rel_gain,
            gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
        )
    else:
        s = float(scale)
    lo = max(float(np.min(wl_ref)), float(np.min(wl_mov)))
    hi = min(float(np.max(wl_ref)), float(np.max(wl_mov)))
    hw = max(float(seam_band_half_width_aa), 1.0)
    gap = hi <= lo + 1e-9
    if gap:
        wl_seam = 0.5 * (float(np.max(wl_ref)) + float(np.min(wl_mov)))
        mr = float(np.nanmedian(wl_ref))
        mm = float(np.nanmedian(wl_mov))
        if mr > mm:
            ref_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=False)
            mov_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=True)
        else:
            ref_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=True)
            mov_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=False)
        return float(s), {
            "mode": "gap",
            "scale": float(s),
            "overlap_aa": None,
            "overlap_shade_aa": None,
            "wl_seam_aa": float(wl_seam),
            "ref_seam_band_aa": ref_band,
            "mov_seam_band_aa": mov_band,
            "overlap_grid": int(overlap_grid),
        }

    wl_seam = 0.5 * (lo + hi)
    mr = float(np.nanmedian(wl_ref))
    mm = float(np.nanmedian(wl_mov))
    if mr > mm:
        ref_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=False)
        mov_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=True)
    else:
        ref_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=True)
        mov_band = _seam_band_interval_aa(wl_seam, hw, take_blueward=False)
    # Full λ intersection is where χ² resamples the overlap; for plots, shade a narrow
    # band (~2×seam half-width) around wl_seam so overlays match the seam scale (~100 Å).
    shade_lo = max(lo, wl_seam - hw)
    shade_hi = min(hi, wl_seam + hw)
    if shade_hi <= shade_lo + 1e-9:
        shade_lo, shade_hi = float(lo), float(min(hi, lo + 2.0 * hw))
    return float(s), {
        "mode": "overlap",
        "scale": float(s),
        "overlap_aa": (float(lo), float(hi)),
        "overlap_shade_aa": (float(shade_lo), float(shade_hi)),
        "wl_seam_aa": float(wl_seam),
        "ref_seam_band_aa": ref_band,
        "mov_seam_band_aa": mov_band,
        "overlap_grid": int(overlap_grid),
    }


def apply_intra_epoch_arm_scaling(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    epoch_of_row: np.ndarray,
    canonical_phases: np.ndarray,
    *,
    phase_atol: float,
    seam_weight: float,
    overlap_grid: int,
    seam_band_half_width_aa: float,
    arm_gap_factor: float,
    arm_min_gap_norm: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same overlap+seam χ² scaler **between λ arms** of one spectroscopic epoch.

    Time-bundle scaling only links **different** epochs; a single exposure can still be
    several disjoint wavelength arms (orders / chips). Without this step, bundle panels
    with ``1 exposure(s)`` keep visible jumps at arm boundaries.
    """
    n_eps = int(canonical_phases.size)
    yo = np.asarray(y, dtype=float).copy()
    ye = np.asarray(yerr, dtype=float).copy()
    for epi in range(n_eps):
        ph = float(canonical_phases[int(epi)])
        ph_key = float(np.round(ph, int(ROUND_PHASE)))
        ridx = np.flatnonzero(
            (epoch_of_row == int(epi))
            & np.isfinite(X[:, 0])
            & np.isfinite(yo)
            & (np.round(X[:, 1], int(ROUND_PHASE)) == ph_key)
        )
        if ridx.size < 6:
            continue
        order = np.argsort(X[ridx, 0])
        ms = ridx[order].astype(int)
        u = X[ms, 0].astype(float)
        dx = np.diff(u)
        med = float(np.median(dx[dx > 1e-12])) if np.any(dx > 1e-12) else 0.002
        thr = max(float(arm_gap_factor) * med, float(arm_min_gap_norm))
        cuts = [0]
        for ii in range(dx.size):
            if dx[ii] > thr:
                cuts.append(ii + 1)
        cuts.append(ms.size)
        segs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for a in range(len(cuts) - 1):
            i0, i1 = cuts[a], cuts[a + 1]
            if i1 - i0 < 2:
                continue
            sub = ms[i0:i1]
            wl_log = (
                u[i0:i1]
                if gn.get("_normalized_only")
                else float(gn["x1_mean"]) + float(gn["x1_std"]) * u[i0:i1]
            )
            wav = np.power(10.0, wl_log).astype(float)
            fl = scaled_ln_to_linear(yo[sub], gn)
            el = linear_flux_yerr(yo[sub], ye[sub], gn)
            segs.append((sub, wav, fl, el))
        if len(segs) < 2:
            continue
        cumulative = 1.0
        wl_p, fp, ep = segs[0][1].copy(), segs[0][2].copy(), segs[0][3].copy()
        for si in range(1, len(segs)):
            ms_k, wl_k, fk, ek = segs[si]
            if wl_p.size >= 2 and wl_k.size >= 2:
                s_ij = solve_pair_scale(
                    wl_p,
                    fp * cumulative,
                    ep * cumulative,
                    wl_k,
                    fk,
                    ek,
                    seam_weight=seam_weight,
                    overlap_grid=overlap_grid,
                    seam_band_half_width_aa=seam_band_half_width_aa,
                    gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                    gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
                )
            else:
                s_ij = 1.0
            cumulative *= float(s_ij)
            if abs(cumulative - 1.0) > 1e-14:
                yo, ye = apply_epoch_linear_multiplier(yo, ye, gn, ms_k, mult=float(cumulative))
            wl_p, fp, ep = wl_k.copy(), fk.copy(), ek.copy()
    return yo, ye


def _overlap_width_aa(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Physical Å overlap width between two wavelength arrays (for tests / diagnostics)."""
    if w_a.size == 0 or w_b.size == 0:
        return 0.0
    lo = max(float(np.min(w_a)), float(np.min(w_b)))
    hi = min(float(np.max(w_a)), float(np.max(w_b)))
    return float(hi - lo) if hi > lo else 0.0


def _forward_mst_edge_indices(
    elist: list[int], data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[int, int]]:
    """MST on bundle epochs in **center-λ order** using only edges ``i<j`` (bluer→redder).

    Kruskal with weight ``-overlap_Å`` plus consecutive-index backbone so propagation can
    always treat the lower index as the bluer endpoint.
    """
    n = len(elist)
    if n <= 1:
        return []
    backbone_w = 1e3
    edges: list[tuple[float, int, int]] = []
    for ii in range(n - 1):
        edges.append((backbone_w, ii, ii + 1))
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = elist[i], elist[j]
            ov = _overlap_width_aa(data[ai][0], data[aj][0])
            if ov > 1e-6:
                edges.append((-ov, i, j))
    edges.sort(key=lambda t: t[0])
    parent_u = list(range(n))

    def find(u: int) -> int:
        while parent_u[u] != u:
            parent_u[u] = parent_u[parent_u[u]]
            u = parent_u[u]
        return u

    def union(u: int, v: int) -> bool:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent_u[rv] = ru
        return True

    mst_edges: list[tuple[int, int]] = []
    for _w, i, j in edges:
        if union(i, j):
            mst_edges.append((i, j))
            if len(mst_edges) >= n - 1:
                break
    if len(mst_edges) < n - 1:
        mst_edges = [(ii, ii + 1) for ii in range(n - 1)]
    return mst_edges


def _intra_bundle_mults_from_mst(
    elist: list[int],
    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    mst_edges: list[tuple[int, int]],
    *,
    seam_weight: float,
    overlap_grid_points: int,
    seam_band_half_width_aa: float,
    trace: Optional[list[dict[str, Any]]] = None,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> dict[int, float]:
    """DFS from bluest (index 0); each tree step solves bluer→redder or inverse via division."""
    n = len(elist)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in mst_edges:
        adj[i].append(j)
        adj[j].append(i)
    mult_epoch: dict[int, float] = {int(elist[0]): 1.0}
    stack = [0]
    seen = {0}
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            stack.append(j)
            lo_idx, hi_idx = (i, j) if i < j else (j, i)
            e_lo, e_hi = int(elist[lo_idx]), int(elist[hi_idx])
            wl_b, fb, eb = data[e_lo]
            wl_r, fr, er = data[e_hi]
            m_lo = mult_epoch.get(e_lo)
            m_hi = mult_epoch.get(e_hi)
            if m_lo is not None and m_hi is None:
                f_ref = fb * m_lo
                e_ref = eb * m_lo
                s_ij = solve_pair_scale(
                    wl_b,
                    f_ref,
                    e_ref,
                    wl_r,
                    fr,
                    er,
                    seam_weight=seam_weight,
                    overlap_grid=overlap_grid_points,
                    seam_band_half_width_aa=seam_band_half_width_aa,
                    gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                    gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
                )
                # `solve_pair_scale(ref, mov)` returns the absolute multiplier for `mov`
                # against the already-scaled reference flux supplied in this call.
                mult_epoch[e_hi] = float(s_ij)
                if trace is not None:
                    _, geom = pair_scale_report(
                        wl_b,
                        f_ref,
                        e_ref,
                        wl_r,
                        fr,
                        er,
                        seam_weight=seam_weight,
                        overlap_grid=overlap_grid_points,
                        seam_band_half_width_aa=seam_band_half_width_aa,
                        scale=float(s_ij),
                        gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                        gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
                    )
                    trace.append(
                        {
                            "case": "bluer_known_scale_redder",
                            "graph_parent_idx": int(i),
                            "graph_child_idx": int(j),
                            "epoch_bluer": e_lo,
                            "epoch_redder": e_hi,
                            "mult_bluer_before": float(m_lo),
                            "s_ij": float(s_ij),
                            "mult_redder_after": float(mult_epoch[e_hi]),
                            "wl_bluer_aa": wl_b,
                            "f_bluer_in": np.asarray(f_ref, dtype=float).copy(),
                            "wl_redder_aa": wl_r,
                            "f_redder_in": np.asarray(fr, dtype=float).copy(),
                            "geometry": geom,
                        }
                    )
            elif m_hi is not None and m_lo is None:
                f_mov = fr * m_hi
                e_mov = er * m_hi
                s_ij = solve_pair_scale(
                    wl_b,
                    fb,
                    eb,
                    wl_r,
                    f_mov,
                    e_mov,
                    seam_weight=seam_weight,
                    overlap_grid=overlap_grid_points,
                    seam_band_half_width_aa=seam_band_half_width_aa,
                    gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                    gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
                )
                # Here `s_ij` scales the (already-scaled) redder spectrum to the bluer one,
                # so the absolute multiplier for the bluer epoch is 1/s_ij.
                mult_epoch[e_lo] = 1.0 / float(max(float(s_ij), 1e-12))
                if trace is not None:
                    _, geom = pair_scale_report(
                        wl_b,
                        fb,
                        eb,
                        wl_r,
                        f_mov,
                        e_mov,
                        seam_weight=seam_weight,
                        overlap_grid=overlap_grid_points,
                        seam_band_half_width_aa=seam_band_half_width_aa,
                        scale=float(s_ij),
                        gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                        gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
                    )
                    trace.append(
                        {
                            "case": "redder_known_scale_bluer",
                            "graph_parent_idx": int(i),
                            "graph_child_idx": int(j),
                            "epoch_bluer": e_lo,
                            "epoch_redder": e_hi,
                            "mult_redder_before": float(m_hi),
                            "s_ij": float(s_ij),
                            "mult_bluer_after": float(mult_epoch[e_lo]),
                            "wl_bluer_aa": wl_b,
                            "f_bluer_in": np.asarray(fb, dtype=float).copy(),
                            "wl_redder_aa": wl_r,
                            "f_redder_in": np.asarray(f_mov, dtype=float).copy(),
                            "geometry": geom,
                        }
                    )
    for e in elist:
        mult_epoch.setdefault(int(e), 1.0)
    return mult_epoch


def intra_bundle_epoch_scale_trace(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    canonical_phases: np.ndarray,
    epoch_members: np.ndarray,
    epoch_of_row: np.ndarray,
    *,
    phase_atol: float,
    seam_weight: float,
    overlap_grid_points: int,
    seam_band_half_width_aa: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> tuple[dict[int, float], list[dict[str, Any]], list[int], list[tuple[int, int]]]:
    """Like ``intra_bundle_epoch_scales`` but also return MST edge list and per-edge solver traces.

    Each trace entry includes ``wl_*_aa``, flux arrays **as passed to** ``solve_pair_scale``,
    ``s_ij``, and ``geometry`` from ``pair_scale_report`` (overlap interval, seam bands).
    """
    epochs = np.sort(np.asarray(epoch_members, dtype=int))
    mult: dict[int, float] = {}
    if epochs.size <= 1:
        if epochs.size == 1:
            mult[int(epochs[0])] = 1.0
        return mult, [], [int(epochs[0])] if epochs.size else [], []

    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    centers: dict[int, float] = {}
    for ee in epochs:
        w, f, e = composite_epoch_linear(
            X,
            y,
            yerr,
            gn,
            canon_phase(canonical_phases, int(ee)),
            phase_atol=phase_atol,
            epoch_of_row=epoch_of_row,
            epoch_id=int(ee),
        )
        data[int(ee)] = (w, f, e)
        centers[int(ee)] = float(np.nanmedian(w)) if w.size else float("inf")

    order_epochs = epochs[np.argsort(np.asarray([centers[int(e)] for e in epochs]))]
    elist = [int(e) for e in order_epochs]
    mst_ix = _forward_mst_edge_indices(elist, data)
    trace: list[dict[str, Any]] = []
    mult_out = _intra_bundle_mults_from_mst(
        elist,
        data,
        mst_ix,
        seam_weight=seam_weight,
        overlap_grid_points=overlap_grid_points,
        seam_band_half_width_aa=seam_band_half_width_aa,
        trace=trace,
        gap_veto_min_rel_gain=gap_veto_min_rel_gain,
        gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
    )
    return mult_out, trace, elist, mst_ix


def intra_bundle_epoch_scales(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    canonical_phases: np.ndarray,
    epoch_members: np.ndarray,
    epoch_of_row: np.ndarray,
    *,
    phase_atol: float,
    seam_weight: float,
    overlap_grid_points: int,
    seam_band_half_width_aa: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> dict[int, float]:
    """Cumulative linear multipliers per spectroscopic epoch within one time bundle.

    Epochs are **never merged** (discrete rounded phase per ``composite_epoch_linear``).
    Build a **forward-only Kruskal MST** on epochs sorted by median λ (edges ``i<j`` only),
    then propagate multipliers by DFS from the bluest epoch using ``solve_pair_scale``
    (overlap χ² or gap seam).  See ``_forward_mst_edge_indices`` / ``_intra_bundle_mults_from_mst``.
    """
    mult, _, _, _ = intra_bundle_epoch_scale_trace(
        X,
        y,
        yerr,
        gn,
        canonical_phases,
        epoch_members,
        epoch_of_row,
        phase_atol=phase_atol,
        seam_weight=seam_weight,
        overlap_grid_points=overlap_grid_points,
        seam_band_half_width_aa=seam_band_half_width_aa,
        gap_veto_min_rel_gain=gap_veto_min_rel_gain,
        gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
    )
    return mult


def apply_epoch_linear_multiplier(
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    row_idxs: np.ndarray,
    *,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mult == 1.0 or row_idxs.size == 0:
        return y.copy(), yerr.copy()
    yo = np.asarray(y, dtype=float).copy()
    eo = np.asarray(yerr, dtype=float).copy()
    yi = scaled_ln_to_linear(yo[row_idxs], gn)
    ferri = linear_flux_yerr(yo[row_idxs], eo[row_idxs], gn)
    yi2 = yi * mult
    ferri2 = ferri * abs(mult)
    yo[row_idxs] = bpre.latent_from_linear(yi2, gn)

    sf = float(gn.get("scale_factor", 1.0))
    denom = np.maximum(np.abs(yi2) * sf if not gn.get("_normalized_only") else np.ones_like(yi2), 1e-30)
    eo[row_idxs] = np.maximum(ferri2 / denom, 1e-12)

    return yo, eo


def phot_band_training_groups(
    X: np.ndarray,
    phot_mask: np.ndarray,
    enrich: Optional[dict[str, np.ndarray]],
    *,
    pseudo_digits: int,
) -> dict[str, np.ndarray]:
    idx = np.flatnonzero(phot_mask)
    if idx.size == 0:
        return {}

    labels_non_empty = enrich is not None and (
        "band_name" in enrich or "band_id" in enrich
    )
    groups: dict[str, list[int]] = {}

    if labels_non_empty and enrich is not None and "band_name" in enrich:
        bn = np.asarray(enrich["band_name"], dtype=str).ravel()
        for i_t in idx:
            key = str(bn[i_t]) if i_t < bn.size else ""
            if key:
                groups.setdefault(key, []).append(int(i_t))
        if groups:
            return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    if labels_non_empty and enrich is not None and "band_id" in enrich:
        bids = np.asarray(enrich["band_id"]).ravel()
        for i_t in idx:
            key = f"id_{int(bids[i_t])}" if i_t < bids.size else ""
            groups.setdefault(str(key), []).append(int(i_t))
        return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    pk = np.round(X[idx, 0], pseudo_digits)
    for j, tt in enumerate(idx):
        groups.setdefault(str(float(pk[j])), []).append(int(tt))
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _phot_gp_lengthscale(x: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float)
    xv = xv[np.isfinite(xv)]
    span = float(np.ptp(xv)) if xv.size else 0.0
    return float(max(0.03, 0.15 * (span + 1e-9)))


def _fit_phot_band_gp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    yerr: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Fixed-lengthscale RBF + nugget; returns ``(x, alpha, ls)`` with ``mu(xq)=k(xq) @ alpha``."""
    ok = np.isfinite(x_train) & np.isfinite(y_train) & np.isfinite(yerr) & (yerr > 0)
    x = np.asarray(x_train, dtype=float)[ok]
    y = np.asarray(y_train, dtype=float)[ok]
    e = np.asarray(yerr, dtype=float)[ok]
    n = x.size
    if n < 2:
        return None
    ls = _phot_gp_lengthscale(x)
    ysd = float(np.nanstd(y))
    sn2 = float(max(np.mean(e**2), (0.05 * ysd if ysd > 0 else 1e-12) ** 2, 1e-16))
    d = (x[:, None] - x[None, :]) / ls
    kmat = np.exp(-0.5 * d * d)
    kmat = kmat + sn2 * np.eye(n)
    try:
        alpha = np.linalg.solve(kmat, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(kmat, y, rcond=1e-9)[0]
    return (x, alpha, ls)


def _mu_phot_gp(xq: float, pack: tuple[np.ndarray, np.ndarray, float]) -> float:
    x, alpha, ls = pack
    k = np.exp(-0.5 * ((x - float(xq)) / ls) ** 2)
    return float(np.dot(k, alpha))


def estimate_bundle_per_epoch_log_scales(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    canonical_phases: np.ndarray,
    epoch_of_row: np.ndarray,
    epoch_ids_bundle: np.ndarray,
    band_groups: dict[str, np.ndarray],
    mu_train: np.ndarray,
    gn: dict,
    *,
    bands_try: list[str],
    aliases: dict[str, str],
    trds_roots: list[str],
    synth_system: str,
    phase_atol: float,
    delta_log_clip: float = 1.25,
) -> tuple[dict[int, float], dict[str, Any]]:
    """One log-scale per spec epoch: synth from **that epoch's** spectrum vs per-band GP µ(phase).

    Photometric training rows in each band are regressed in normalized phase with a simple
    RBF+ridge smoother; ``µ_phot`` is queried at the epoch's canonical phase (no pooling of
    spectroscopic epochs).
    """
    eps = np.asarray(epoch_ids_bundle, dtype=int)
    deltas: dict[int, float] = {}
    per_ep: dict[str, Any] = {}
    skipped_all: list[dict[str, Any]] = []

    for epi in eps.tolist():
        ph_ep = float(canon_phase(canonical_phases, int(epi)))
        wl_ep, fl_ep, _fe_ep = composite_epoch_linear(
            X,
            y,
            yerr,
            gn,
            ph_ep,
            phase_atol=float(phase_atol),
            epoch_of_row=epoch_of_row,
            epoch_id=int(epi),
        )
        synth_mag, rep = synthesize_for_bands(
            wl_ep,
            fl_ep,
            bands_try,
            band_aliases=aliases,
            trds_roots=trds_roots,
            system=synth_system,
        )

        dlogs: list[float] = []
        wts: list[float] = []
        skipped: list[dict[str, Any]] = []
        for bk in bands_try:
            if bk not in synth_mag:
                skipped.append({"band": bk, "reason": "no_synth"})
                continue
            gidx = band_groups.get(bk)
            if gidx is None or gidx.size == 0:
                skipped.append({"band": bk, "reason": "no_phot_anchor"})
                continue
            xtr = X[gidx, 1]
            mu_lin = scaled_ln_to_linear(mu_train[gidx], gn)
            yt_err = linear_flux_yerr(mu_train[gidx], yerr[gidx], gn)
            ok = np.isfinite(xtr) & np.isfinite(mu_lin) & np.isfinite(yt_err) & (yt_err > 0)
            if not np.any(ok):
                skipped.append({"band": bk, "reason": "no_mu"})
                continue
            gp_pack = _fit_phot_band_gp(xtr[ok], mu_lin[ok], yt_err[ok])
            if gp_pack is not None:
                mu_phot = _mu_phot_gp(ph_ep, gp_pack)
            else:
                j0 = int(np.argmin(np.abs(xtr[ok] - ph_ep)))
                mu_phot = float(mu_lin[ok][j0])

            ij = np.flatnonzero(ok)
            ij = ij[: min(512, ij.size)]
            x1_med = float(np.median(X[gidx[ij], 0]))
            if gn.get("_normalized_only"):
                log10_aa = float(x1_med)
            else:
                log10_aa = float(gn["x1_mean"]) + float(gn["x1_std"]) * float(x1_med)
            wav_eff_aa = np.power(10.0, log10_aa)
            f_syn_b = ab_mag_to_flam(float(synth_mag[bk]), wav_eff_aa)

            if f_syn_b <= 0 or mu_phot <= 0 or not np.isfinite(f_syn_b) or not np.isfinite(mu_phot):
                skipped.append({"band": bk, "reason": "bad_flux_conversion"})
                continue
            dlogs.append(float(np.log(mu_phot / f_syn_b)))
            wts.append(float(np.sum(ok)))

        ep_key = str(int(epi))
        if not wts:
            deltas[int(epi)] = 0.0
            per_ep[ep_key] = {
                "applied": False,
                "delta_log_scale": 0.0,
                "synth_mag": synth_mag,
                "skipped_band_details": skipped,
            }
            skipped_all.extend(skipped)
        else:
            ww = np.asarray(wts, dtype=float)
            ww /= float(np.sum(ww))
            dlog = float(np.sum(np.asarray(dlogs, dtype=float) * ww))
            dlog = float(np.clip(dlog, -float(delta_log_clip), float(delta_log_clip)))
            deltas[int(epi)] = dlog
            per_ep[ep_key] = {
                "applied": True,
                "delta_log_scale": dlog,
                "synth_mag": synth_mag,
                "skipped_band_details": skipped,
                "filters_report_jsonable": rep.to_jsonable(),
            }
        del rep

    any_applied = any(bool(v.get("applied")) for v in per_ep.values())
    meta: dict[str, Any] = {
        "per_epoch": per_ep,
        "skipped_band_details_flat": skipped_all,
        "any_applied": any_applied,
    }
    return deltas, meta


def _interp_spec_linear_flux_positive(
    wm: np.ndarray,
    fm: np.ndarray,
    wl: float,
    *,
    endpoint_clamp: bool,
) -> tuple[float, bool]:
    """Interpolate sorted linear flux ``fm(wm)`` at wavelength ``wl`` (Å).

    Returns ``(M, used_endpoint_clamp)``. ``M`` is NaN if no positive finite value.
    """
    wm = np.asarray(wm, dtype=float).ravel()
    fm = np.asarray(fm, dtype=float).ravel()
    if wm.size == 0 or fm.size == 0 or not np.isfinite(wl) or wl <= 0:
        return float("nan"), False
    if wm.size == 1:
        m = float(fm[0])
        return (m, False) if np.isfinite(m) and m > 0 else (float("nan"), False)
    if endpoint_clamp:
        wq = float(np.clip(wl, wm[0], wm[-1]))
        m = float(np.interp(wq, wm, fm))
        return (m, bool(wq != wl)) if np.isfinite(m) and m > 0 else (float("nan"), False)
    if wl < wm[0] or wl > wm[-1]:
        return float("nan"), False
    m = float(np.interp(wl, wm, fm))
    return (m, False) if np.isfinite(m) and m > 0 else (float("nan"), False)


def _linear_scale_chi2_ls(
    d_lin: np.ndarray,
    sig: np.ndarray,
    m_lin: np.ndarray,
    *,
    s_lo: float,
    s_hi: float,
) -> tuple[float, float, int]:
    """Return ``(s_star, chi2_at_s_star, n_points)`` for χ² = Σ ((D - s M)/σ)² with phot-only weights."""
    d_lin = np.asarray(d_lin, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float).ravel()
    m_lin = np.asarray(m_lin, dtype=float).ravel()
    ok = np.isfinite(d_lin) & np.isfinite(sig) & np.isfinite(m_lin) & (sig > 0) & (d_lin > 0) & (m_lin > 0)
    d_lin = d_lin[ok]
    sig = sig[ok]
    m_lin = m_lin[ok]
    n = int(d_lin.size)
    if n == 0:
        return 1.0, float("inf"), 0
    invv = 1.0 / (sig * sig)
    num = float(np.sum(d_lin * m_lin * invv))
    den = float(np.sum(m_lin * m_lin * invv))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return 1.0, float("inf"), n
    s_raw = num / den
    s_star = float(np.clip(s_raw, float(s_lo), float(s_hi)))
    chi2 = float(np.sum(((d_lin - s_star * m_lin) / sig) ** 2))
    return s_star, chi2, n


def estimate_bundle_pooled_phot_chi2_linear_scale(
    X_full: np.ndarray,
    y_full: np.ndarray,
    yerr_full: np.ndarray,
    gn: dict,
    phot_mask_full: np.ndarray,
    canonical_phases: np.ndarray,
    epoch_of_row: np.ndarray,
    epochs_b: np.ndarray,
    *,
    phase_epoch_atol: float,
    rough_phot_phase_window_norm: float,
    bundle_phot_pool_max_phase_window_norm: float,
    phot_anchor_min_points: int,
    linear_s_clip_lo: float = 1e-12,
    linear_s_clip_hi: float = 1e12,
) -> tuple[float, dict[str, Any]]:
    """One linear flux scale ``s*`` per bundle minimizing χ² vs pooled phot (phot errors only).

    Retry ladder: widen phase band around bundle spec phases, then all phot with strict
    in-range interpolation, then all phot with endpoint clamping on λ.
    """
    meta: dict[str, Any] = {
        "anchor_mode": "bundle_pooled_phot_chi2",
        "any_applied": False,
        "phot_anchor_ladder_stage": None,
        "phot_anchor_endpoint_clamp_used": False,
        "phot_anchor_n_points": 0,
        "chi2_at_s_opt": float("inf"),
        "pooled_linear_scale_s_opt": 1.0,
        "reason": None,
    }
    if not bool(np.any(phot_mask_full)):
        meta["reason"] = "no_photometry_training_rows"
        return 1.0, meta

    ep_list = [int(e) for e in np.asarray(epochs_b, dtype=int).ravel().tolist()]
    if not ep_list:
        meta["reason"] = "empty_bundle_epochs"
        return 1.0, meta

    ph_list = [float(canon_phase(canonical_phases, int(e))) for e in ep_list]
    ph_lo, ph_hi = float(min(ph_list)), float(max(ph_list))

    epoch_wf: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for e in ep_list:
        ph_e = float(canon_phase(canonical_phases, int(e)))
        w_e, f_e, _fe = composite_epoch_linear(
            X_full,
            y_full,
            yerr_full,
            gn,
            ph_e,
            phase_atol=float(phase_epoch_atol),
            epoch_of_row=epoch_of_row,
            epoch_id=int(e),
        )
        ok = np.isfinite(w_e) & np.isfinite(f_e) & (w_e > 0) & (f_e > 0)
        w_e = np.asarray(w_e, dtype=float)[ok]
        f_e = np.asarray(f_e, dtype=float)[ok]
        if w_e.size >= 2:
            o = np.argsort(w_e)
            w_e = w_e[o]
            f_e = f_e[o]
        epoch_wf[int(e)] = (w_e, f_e)

    def collect_pairs(
        *,
        phase_lo: Optional[float],
        phase_hi: Optional[float],
        endpoint_clamp: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        d_out: list[float] = []
        s_out: list[float] = []
        m_out: list[float] = []
        any_wl_endpoint_clamp = False
        x2_all = np.asarray(X_full[:, 1], dtype=float).ravel()
        for i in range(int(X_full.shape[0])):
            if not bool(phot_mask_full[i]):
                continue
            x2p = float(x2_all[i])
            if phase_lo is not None and phase_hi is not None:
                if not (np.isfinite(x2p) and float(phase_lo) <= x2p <= float(phase_hi)):
                    continue
            elif not np.isfinite(x2p):
                continue
            wl = float(wavelength_aa_from_x1_norm(np.asarray([X_full[i, 0]], dtype=float), gn)[0])
            d_lin = float(scaled_ln_to_linear(np.asarray([y_full[i]], dtype=float), gn)[0])
            sig = float(linear_flux_yerr(np.asarray([y_full[i]], dtype=float), np.asarray([yerr_full[i]], dtype=float), gn)[0])
            if not (np.isfinite(wl) and wl > 0 and np.isfinite(d_lin) and d_lin > 0 and np.isfinite(sig) and sig > 0):
                continue
            e_sorted = sorted(ep_list, key=lambda ee: abs(x2p - float(canon_phase(canonical_phases, int(ee)))))
            m_best = float("nan")
            for ep_try in e_sorted:
                wm, fm = epoch_wf[int(ep_try)]
                m_try, ucl = _interp_spec_linear_flux_positive(wm, fm, wl, endpoint_clamp=endpoint_clamp)
                if np.isfinite(m_try) and m_try > 0:
                    m_best = float(m_try)
                    if ucl:
                        any_wl_endpoint_clamp = True
                    break
            if np.isfinite(m_best) and m_best > 0:
                d_out.append(d_lin)
                s_out.append(sig)
                m_out.append(m_best)
        return (
            np.asarray(d_out, dtype=float),
            np.asarray(s_out, dtype=float),
            np.asarray(m_out, dtype=float),
            bool(any_wl_endpoint_clamp),
        )

    w0 = float(rough_phot_phase_window_norm)
    wmax = float(max(bundle_phot_pool_max_phase_window_norm, w0))
    windows: list[float] = [w0]
    wcur = w0
    step = max(w0 * 0.5, 0.02)
    while wcur + step <= wmax + 1e-12:
        wcur += step
        windows.append(float(min(wcur, wmax)))
    if wmax > windows[-1] + 1e-12:
        windows.append(wmax)
    windows = sorted({float(round(w, 12)) for w in windows})

    strategies: list[tuple[str, Optional[tuple[float, float]], bool]] = []
    for w in windows:
        strategies.append((f"phase_band_w={w:.6g}", (ph_lo - w, ph_hi + w), False))
    strategies.append(("all_phot_inrange", None, False))
    strategies.append(("all_phot_endpoint_clamp", None, True))

    min_pts = max(1, int(phot_anchor_min_points))
    for label, band, ep_clamp in strategies:
        if band is not None:
            d_a, sig_a, m_a, uclamp = collect_pairs(
                phase_lo=band[0], phase_hi=band[1], endpoint_clamp=ep_clamp
            )
        else:
            d_a, sig_a, m_a, uclamp = collect_pairs(phase_lo=None, phase_hi=None, endpoint_clamp=ep_clamp)
        if d_a.size < min_pts:
            continue
        s_star, chi2, npt = _linear_scale_chi2_ls(
            d_a, sig_a, m_a, s_lo=float(linear_s_clip_lo), s_hi=float(linear_s_clip_hi)
        )
        if npt < min_pts:
            continue
        meta["phot_anchor_ladder_stage"] = label
        meta["phot_anchor_endpoint_clamp_used"] = bool(uclamp)
        meta["phot_anchor_n_points"] = int(npt)
        meta["chi2_at_s_opt"] = float(chi2)
        meta["pooled_linear_scale_s_opt"] = float(s_star)
        meta["any_applied"] = True
        return float(s_star), meta

    meta["reason"] = "ladder_exhausted_no_chi2_pairs"
    return 1.0, meta


def estimate_bundle_log_scales_rough_phot_wavelength_points(
    X_full: np.ndarray,
    y_full: np.ndarray,
    yerr_full: np.ndarray,
    gn: dict,
    phot_mask_full: np.ndarray,
    canonical_phases: np.ndarray,
    epoch_of_row: np.ndarray,
    epochs_b: np.ndarray,
    *,
    phase_epoch_atol: float,
    rough_phot_phase_window_norm: float,
    phot_anchor_min_points: int = 1,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Per-epoch log scale from nearby-phase photometry at effective wavelengths (bundle fallback)."""
    dl_by_ep: dict[int, float] = {}
    per_ep: dict[str, Any] = {}
    for epi_inner in epochs_b.tolist():
        ph_ep = float(canon_phase(canonical_phases, int(epi_inner)))
        wl_ep, fl_ep, _fe_ep = composite_epoch_linear(
            X_full,
            y_full,
            yerr_full,
            gn,
            ph_ep,
            phase_atol=float(phase_epoch_atol),
            epoch_of_row=epoch_of_row,
            epoch_id=int(epi_inner),
        )
        dlog, meta = estimate_epoch_log_scale_from_phot_wavelength_points(
            X=X_full,
            y=y_full,
            yerr=yerr_full,
            gn=gn,
            phot_mask=phot_mask_full,
            ph_epoch=ph_ep,
            wl_spec=wl_ep,
            fl_spec=fl_ep,
            phot_phase_window_norm=float(rough_phot_phase_window_norm),
            min_points=int(phot_anchor_min_points),
        )
        dl_by_ep[int(epi_inner)] = float(dlog)
        per_ep[str(int(epi_inner))] = meta
    any_applied = any(bool(v.get("applied")) for v in per_ep.values())
    bm: dict[str, Any] = {
        "per_epoch": per_ep,
        "any_applied": any_applied,
        "anchor_mode": "rough_phot_wavelength_points",
    }
    return dl_by_ep, bm


def run_gp_shell(
    input_npz: str,
    output_dir: str,
    gp_tag: str,
    extra: list[str],
) -> None:
    cmd = [
        sys.executable,
        os.path.join(HERE, "run_gp.py"),
        "--input",
        os.path.abspath(input_npz),
        "--output-dir",
        os.path.abspath(output_dir),
        "--tag",
        gp_tag,
    ]
    cmd.extend(extra)
    print("[bundle_scale_pipeline]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def apply_scales_pipeline(
    input_npz: str,
    output_npz: str,
    gn: dict,
    *,
    max_bundle_minutes: float,
    phase_epoch_atol: float,
    seam_weight: float,
    overlap_grid: int,
    enrich: Optional[dict[str, np.ndarray]],
    phot_spec_threshold: int,
    pseudo_band_digits: int,
    filter_config_path: Optional[str],
    global_scale_iters: int,
    run_gp_flag: bool,
    runs_dir: str,
    gp_tag_prefix: str,
    gp_extra: list[str],
    phase_tolerance_norm_global: float,  # retained for CLI compatibility; phot GP uses exact epoch phase
    rough_phot_anchor: bool,
    rough_phot_phase_window_norm: float,
    bundle_phot_pool_max_phase_window_norm: float,
    phot_anchor_min_points: int,
    seam_band_half_width_aa: float,
    arm_gap_factor: float,
    arm_min_gap_norm: float,
    gap_veto_min_rel_gain: Optional[float] = None,
    gap_veto_edge_err_ratio: Optional[float] = None,
) -> dict[str, Any]:
    bd = np.load(input_npz, allow_pickle=False)
    X_full = np.asarray(bd["X"], dtype=float)
    y_full = np.asarray(bd["y"], dtype=float)
    yerr_full = np.asarray(bd["yerr"], dtype=float)

    train_obs_override = bd["train_obs_class"] if "train_obs_class" in bd.files else None
    point_class_full = gu.effective_point_class(X_full, train_obs_class=np.asarray(train_obs_override) if train_obs_override is not None else None, threshold=int(phot_spec_threshold))

    spec_mask_full = point_class_full == gu.SPEC
    canonical_phases, epoch_of_row = unique_spec_epochs(X_full, spec_mask_full)
    n_eps = canonical_phases.size
    bundle_of_epoch = np.full(n_eps, -1, dtype=np.int32) if n_eps else np.zeros(0, dtype=np.int32)
    spec_bundle_id_full = np.full(X_full.shape[0], -1, dtype=np.int32)

    report: dict[str, Any] = {
        "input": os.path.abspath(input_npz),
        "n_epochs": int(n_eps),
        "epochs": [],
    }

    labels_eps = np.zeros(0, dtype=np.int32)
    t_epoch = np.zeros(0, dtype=float)
    if n_eps > 0:
        t_epoch = time_per_epoch(X_full, canonical_phases, gn, enrich)
        labels_eps = sb.cluster_by_time(t_epoch, max_delta_minutes=float(max_bundle_minutes))

        for epi in range(n_eps):
            report["epochs"].append(
                {
                    "epoch": epi,
                    "bundle": int(labels_eps[epi]),
                    "canonical_phase_norm": float(canonical_phases[epi]),
                    "centroid_time_days": float(t_epoch[epi]),
                }
            )

        for epi in range(n_eps):
            bundle_of_epoch[epi] = int(labels_eps[epi])

    intra_mult_by_epoch = {int(epi): 1.0 for epi in range(n_eps)}
    if n_eps > 0:
        y_full, yerr_full = apply_intra_epoch_arm_scaling(
            X_full,
            y_full,
            yerr_full,
            gn,
            epoch_of_row,
            canonical_phases,
            phase_atol=float(phase_epoch_atol),
            seam_weight=float(seam_weight),
            overlap_grid=int(overlap_grid),
            seam_band_half_width_aa=float(seam_band_half_width_aa),
            arm_gap_factor=float(arm_gap_factor),
            arm_min_gap_norm=float(arm_min_gap_norm),
            gap_veto_min_rel_gain=gap_veto_min_rel_gain,
            gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
        )
        for bid in range(int(labels_eps.max()) + 1 if labels_eps.size else 0):
            ep_ids = np.flatnonzero(labels_eps == bid)
            mpart = intra_bundle_epoch_scales(
                X_full,
                y_full,
                yerr_full,
                gn,
                canonical_phases,
                ep_ids,
                epoch_of_row,
                phase_atol=phase_epoch_atol,
                seam_weight=seam_weight,
                overlap_grid_points=overlap_grid,
                seam_band_half_width_aa=float(seam_band_half_width_aa),
                gap_veto_min_rel_gain=gap_veto_min_rel_gain,
                gap_veto_edge_err_ratio=gap_veto_edge_err_ratio,
            )
            for k_ep, vv in mpart.items():
                intra_mult_by_epoch[k_ep] = float(vv)

        y_adj = np.asarray(y_full, dtype=float)
        yer_adj = np.asarray(yerr_full, dtype=float)
        for epi in range(n_eps):
            ridx = np.flatnonzero(epoch_of_row == epi)
            y_adj, yer_adj = apply_epoch_linear_multiplier(
                y_adj, yer_adj, gn, ridx, mult=intra_mult_by_epoch[epi]
            )
            sbid = bundle_of_epoch[epi]
            spec_bundle_id_full[ridx] = sbid

        y_full = y_adj
        yerr_full = yer_adj

    report["intra_linear_mult_by_epoch"] = {str(k): v for k, v in intra_mult_by_epoch.items()}

    trds_roots: list[str] = []
    band_aliases: dict[str, str] = {}
    synth_system = "ab"
    if filter_config_path and os.path.isfile(filter_config_path):
        trds_roots, band_aliases, synth_system = load_filter_config(filter_config_path)

    phot_mask_full = point_class_full == gu.PHOT
    band_groups = phot_band_training_groups(
        X_full,
        phot_mask_full,
        enrich,
        pseudo_digits=pseudo_band_digits,
    )

    bundles_try = (
        sorted({int(bundle_of_epoch[ep]) for ep in range(n_eps)}) if n_eps > 0 else []
    )

    iterations_meta: list[dict[str, Any]] = []
    bundle_log_accum = {str(b): 0.0 for b in bundles_try}

    def _write_bundle(path_out: str) -> None:
        payload: dict[str, Any] = {}
        for kk in bd.files:
            payload[kk] = np.asarray(bd[kk]).copy()
        payload["X"] = X_full
        payload["y"] = y_full
        payload["yerr"] = yerr_full
        payload["spec_bundle_id"] = spec_bundle_id_full
        np.savez(path_out, **payload)

    out_abs = os.path.abspath(output_npz)
    _write_bundle(out_abs)
    mir = mirror_bundle_meta_from_input(os.path.abspath(input_npz), out_abs)
    if mir:
        report["bundle_meta_written"] = mir

    have_filter = bool(filter_config_path) and os.path.isfile(str(filter_config_path))
    use_band_anchor = have_filter and (enrich is not None) and bool(run_gp_flag)
    use_rough_anchor = bool(rough_phot_anchor)
    want_global = bool(int(global_scale_iters) > 0 and n_eps > 0 and bundles_try and (use_band_anchor or use_rough_anchor))
    bands_try_main = sorted(band_groups.keys())
    gp_tag_latest = ""

    if want_global and use_band_anchor and not run_gp_flag:
        report["global_scale_warning"] = "`--global-scale-iters`>0 ignored without `--run-gp` for band-based anchor."
        want_global = False

    for it_i in range(int(global_scale_iters) if want_global else 0):
        mu_tr = None
        if use_band_anchor:
            if not bands_try_main:
                iterations_meta.append({"iter": it_i, "skipped": "no_phot_band_groups"})
                break
            gp_tag_latest = f"{gp_tag_prefix}_gscale_{it_i:d}"
            run_gp_shell(out_abs, runs_dir, gp_tag_latest, gp_extra)
            preds_path = os.path.join(os.path.abspath(runs_dir), gp_tag_latest, "predictions.npz")
            if not os.path.isfile(preds_path):
                iterations_meta.append({"iter": it_i, "error": f"missing_predictions={preds_path!r}"})
                break
            preds = np.load(preds_path, allow_pickle=False)
            try:
                mu_tr = preds["mu_train"] if "mu_train" in preds.files else None
                mu_tr = np.asarray(mu_tr, dtype=float).ravel() if mu_tr is not None else None
                if mu_tr is None or mu_tr.size == 0:
                    iterations_meta.append({"iter": it_i, "error": "predictions.mu_train_missing_or_bad_shape"})
                    break
                n_full_x = int(X_full.shape[0])
                if mu_tr.shape[0] != n_full_x:
                    mu_tr = scatter_train_vector_to_bundle(preds, mu_tr, n_full_x)
                    if mu_tr is None:
                        iterations_meta.append(
                            {"iter": it_i, "error": "predictions.mu_train_missing_or_bad_shape"}
                        )
                        break
            finally:
                preds.close()

        iter_bundle_log: dict[str, Any] = {}
        agg_epoch_factors = np.ones(n_eps, dtype=float)

        for bid_main in bundles_try:
            epochs_b = np.flatnonzero(labels_eps == bid_main).astype(int)
            med_wl_list: list[float] = []
            for e in epochs_b.tolist():
                w_e, _, _ = composite_epoch_linear(
                    X_full,
                    y_full,
                    yerr_full,
                    gn,
                    canon_phase(canonical_phases, int(e)),
                    phase_atol=float(phase_epoch_atol),
                    epoch_of_row=epoch_of_row,
                    epoch_id=int(e),
                )
                med_wl_list.append(float(np.nanmedian(w_e)) if w_e.size else float("inf"))
            rep_ep_bluest = int(epochs_b[int(np.argmin(med_wl_list))])
            rep_ph_bluest = float(canon_phase(canonical_phases, rep_ep_bluest))

            if use_band_anchor and mu_tr is not None:
                dl_by_ep, bm = estimate_bundle_per_epoch_log_scales(
                    X_full,
                    y_full,
                    yerr_full,
                    canonical_phases,
                    epoch_of_row,
                    epochs_b,
                    band_groups,
                    mu_tr,
                    gn,
                    bands_try=bands_try_main,
                    aliases=band_aliases,
                    trds_roots=trds_roots,
                    synth_system=synth_system,
                    phase_atol=float(phase_epoch_atol),
                )
                bm["anchor_mode"] = "band_synth_vs_phot_gp"
                if rough_phot_anchor and not bool(bm.get("any_applied")):
                    n_skip = len(bm.get("skipped_band_details_flat", []))
                    dl_by_ep, bm_rough = estimate_bundle_log_scales_rough_phot_wavelength_points(
                        X_full,
                        y_full,
                        yerr_full,
                        gn,
                        phot_mask_full,
                        canonical_phases,
                        epoch_of_row,
                        epochs_b,
                        phase_epoch_atol=float(phase_epoch_atol),
                        rough_phot_phase_window_norm=float(rough_phot_phase_window_norm),
                        phot_anchor_min_points=int(phot_anchor_min_points),
                    )
                    bm = {
                        **bm_rough,
                        "anchor_mode": "band_failed_no_synth_rough_fallback",
                        "band_anchor_prior_skipped_band_details_n": int(n_skip),
                    }
                    print(
                        f"[bundle_scale_pipeline] NOTE: bundle_{int(bid_main)}: band anchor applied to "
                        f"no epochs (e.g. no_synth); using rough phot-wavelength fallback "
                        f"(prior skipped_band_details={n_skip}).",
                        file=sys.stderr,
                    )
            else:
                dl_by_ep, bm = estimate_bundle_log_scales_rough_phot_wavelength_points(
                    X_full,
                    y_full,
                    yerr_full,
                    gn,
                    phot_mask_full,
                    canonical_phases,
                    epoch_of_row,
                    epochs_b,
                    phase_epoch_atol=float(phase_epoch_atol),
                    rough_phot_phase_window_norm=float(rough_phot_phase_window_norm),
                    phot_anchor_min_points=int(phot_anchor_min_points),
                )
            bm["representative_spec_epoch_bluest_median_wl"] = rep_ep_bluest
            bm["representative_phase_norm_bluest"] = rep_ph_bluest

            # Apply one global factor per bundle (not per-epoch), so relative intra-bundle
            # spectroscopic scaling remains intact after photometric anchoring.
            applied_dls: list[float] = []
            for epi_inner in epochs_b.tolist():
                dl_ep = float(dl_by_ep.get(int(epi_inner), 0.0))
                pep = bm["per_epoch"].get(str(int(epi_inner)), {})
                if bool(pep.get("applied")) and np.isfinite(dl_ep):
                    applied_dls.append(dl_ep)
            if applied_dls:
                d_bundle = float(np.median(np.asarray(applied_dls, dtype=float)))
                gm_bundle = float(np.exp(d_bundle))
                bundle_shared_applied = int(len(applied_dls))
            else:
                s_pool, pool_m = estimate_bundle_pooled_phot_chi2_linear_scale(
                    X_full,
                    y_full,
                    yerr_full,
                    gn,
                    phot_mask_full,
                    canonical_phases,
                    epoch_of_row,
                    epochs_b,
                    phase_epoch_atol=float(phase_epoch_atol),
                    rough_phot_phase_window_norm=float(rough_phot_phase_window_norm),
                    bundle_phot_pool_max_phase_window_norm=float(bundle_phot_pool_max_phase_window_norm),
                    phot_anchor_min_points=int(phot_anchor_min_points),
                )
                gm_bundle = float(np.clip(s_pool, 1e-300, 1e300))
                d_bundle = float(np.log(gm_bundle))
                bm = {**bm, **pool_m}
                bm["anchor_mode"] = str(pool_m.get("anchor_mode", "bundle_pooled_phot_chi2"))
                for epi_inner in epochs_b.tolist():
                    dl_by_ep[int(epi_inner)] = float(d_bundle)
                bundle_shared_applied = int(pool_m.get("phot_anchor_n_points", 0))
                if pool_m.get("any_applied"):
                    print(
                        f"[bundle_scale_pipeline] NOTE: bundle_{int(bid_main)}: pooled χ² phot anchor "
                        f"(n_points={bundle_shared_applied}, stage={pool_m.get('phot_anchor_ladder_stage')!r}).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[bundle_scale_pipeline] WARNING: bundle_{int(bid_main)}: no per-epoch anchor and "
                        f"pooled χ² exhausted ({pool_m.get('reason')!r}).",
                        file=sys.stderr,
                    )
            for epi_inner in epochs_b.tolist():
                agg_epoch_factors[int(epi_inner)] *= gm_bundle
            bundle_log_accum[f"{bid_main}"] = float(bundle_log_accum[str(bid_main)]) + float(d_bundle)
            iter_bundle_log[f"bundle_{int(bid_main)}"] = {
                **bm,
                "delta_log_scale_bundle_sum": d_bundle,
                "bundle_shared_log_scale": d_bundle,
                "bundle_shared_linear_scale": gm_bundle,
                "bundle_shared_applied_epochs": int(bundle_shared_applied),
            }

        for epi in range(n_eps):
            sel = np.flatnonzero(epoch_of_row == int(epi))
            y_full, yerr_full = apply_epoch_linear_multiplier(
                y_full, yerr_full, gn, sel, mult=float(agg_epoch_factors[epi])
            )

        iterations_meta.append({"iteration": int(it_i), "per_bundle_meta": iter_bundle_log})
        _write_bundle(out_abs)

    bd.close()

    report["global_scale_iterations"] = iterations_meta
    report["global_log_sigma_delta_summed_bundle"] = bundle_log_accum
    report["predictions_gp_tag_latest"] = gp_tag_latest if want_global else ""

    jpath = os.path.splitext(out_abs)[0] + "_scale_report.json"
    with open(jpath, "w", encoding="utf-8") as jfh:
        json.dump(report, jfh, indent=2)
        jfh.flush()

    report["scale_report_written"] = jpath
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--meta", default=None)
    p.add_argument(
        "--enrich",
        default=None,
        help="training-row-aligned enrich npz (mjd / band fields). If omitted, looks for "
        "<input_stem>_enrich.npz or enrich.npz beside --input.",
    )
    p.add_argument("--max-bundle-minutes", type=float, default=5.0)
    p.add_argument("--phase-epoch-atol", type=float, default=5e-6)
    p.add_argument("--seam-weight", type=float, default=1.0)
    p.add_argument("--overlap-grid-points", type=int, default=256)
    p.add_argument(
        "--seam-fit-half-width-aa",
        type=float,
        default=50.0,
        help="per-arm seam linear fit uses λ within this many Å of the overlap midpoint "
        "(~100 Å total across the two arms at default 50)",
    )
    p.add_argument(
        "--arm-gap-factor",
        type=float,
        default=35.0,
        help="split λ arms within one epoch when Δx₁_norm exceeds factor×median(Δx₁) (same spirit as plot segmenter)",
    )
    p.add_argument(
        "--arm-min-gap-norm",
        type=float,
        default=3e-3,
        help="minimum normalized x₁ gap (with --arm-gap-factor) to split intra-epoch arms",
    )
    p.add_argument(
        "--gap-veto-min-rel-gain",
        type=float,
        default=None,
        help="optional: if set (e.g. 0.05), reject gap seam scales when WLS line-χ² relative gain "
        "is below this threshold. Default: disabled (always apply finite line-based scale).",
    )
    p.add_argument(
        "--gap-veto-edge-err-ratio",
        type=float,
        default=None,
        help="optional: if set (e.g. 0.65), reject gap scale when median-edge log-ratio does not "
        "improve vs this fraction of the raw ratio. Default: disabled.",
    )
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument("--pseudo-band-digits", type=int, default=4)
    p.add_argument(
        "--filter-config",
        default=None,
        help="YAML with TRDS roots + band aliases. If omitted, uses configs/filter_pipeline.yaml "
        "or configs/filter_pipeline.example.yaml under the repo when present.",
    )
    p.add_argument(
        "--global-scale-iters",
        type=int,
        default=1,
        help="global photometric bundle-scale iterations (0 = relative-only). Default 1: uses "
        "band+synth inner GP when enrich and filter YAML are both present; otherwise rough / pooled χ² "
        "phot anchoring without enrich. Use --skip-global-phot-anchor to disable.",
    )
    p.add_argument(
        "--skip-global-phot-anchor",
        action="store_true",
        help="relative-only output: disable photometric global anchoring (forces effective "
        "global-scale-iters to 0 and skips inner run_gp for this stage).",
    )
    p.add_argument(
        "--rough-phot-anchor",
        dest="rough_phot_anchor",
        action="store_true",
        default=True,
        help="when band+synth anchoring is unavailable (no enrich, no filter, or no inner run_gp), "
        "use nearby-phase phot points at effective wavelengths plus pooled χ² fallback. Default on.",
    )
    p.add_argument("--no-rough-phot-anchor", dest="rough_phot_anchor", action="store_false")
    p.add_argument(
        "--rough-phot-phase-window-norm",
        type=float,
        default=0.06,
        help="for rough phot anchor, include phot points with |Δphase_norm| <= window (default 0.06).",
    )
    p.add_argument(
        "--bundle-phot-pool-max-phase-window-norm",
        type=float,
        default=0.35,
        help="when widening the pooled χ² phot anchor, do not exceed this normalized phase half-width "
        "around the bundle's spec phase span (default 0.35).",
    )
    p.add_argument(
        "--phot-anchor-min-points",
        type=int,
        default=1,
        help="minimum phot–spec pairs for rough per-epoch anchor and for pooled χ² anchor (default 1).",
    )
    p.add_argument(
        "--phase-tolerance-norm-global",
        type=float,
        default=0.06,
        help="legacy; global phot anchor evaluates a per-band smoother at each spec epoch's phase",
    )
    p.add_argument("--run-gp", action="store_true")
    p.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--gp-tag-prefix", default="bundle_scaled")
    p.add_argument(
        "gp_argv",
        nargs=argparse.REMAINDER,
        default=[],
        help="extras forwarded to run_gp.py after `--`, e.g. `-- --sigma-spec 0.01`",
    )
    ns = p.parse_args(argv)

    gp_extra_argv = [str(t) for t in ns.gp_argv if str(t).strip() != "--"]

    inp_abs = os.path.abspath(os.path.expanduser(str(ns.input).strip()))

    enrich_explicit = bool(ns.enrich and str(ns.enrich).strip())
    if enrich_explicit:
        enrich_path = _abspath_file(str(ns.enrich))
        if not os.path.isfile(enrich_path):
            print(
                f"[bundle_scale_pipeline] ERROR: --enrich file not found: {enrich_path!r}\n"
                "  Drop --enrich to auto-discover <input_stem>_enrich.npz or enrich.npz beside --input.",
                file=sys.stderr,
            )
            return 2
    else:
        enrich_path = discover_enrich_npz(inp_abs, None)

    enrich_env: Optional[dict[str, np.ndarray]] = None
    if enrich_path and os.path.isfile(enrich_path):
        ep = np.load(enrich_path, allow_pickle=True)
        enrich_env = {kk: np.asarray(ep[kk]) for kk in ep.files}
        ep.close()
        if not enrich_explicit:
            print(
                f"[bundle_scale_pipeline] NOTE: auto-discovered enrich npz {enrich_path!r}",
                file=sys.stderr,
            )

    filt_explicit = bool(ns.filter_config and str(ns.filter_config).strip())
    if filt_explicit:
        filt_path = _abspath_file(str(ns.filter_config))
        if not os.path.isfile(filt_path):
            print(
                f"[bundle_scale_pipeline] ERROR: --filter-config file not found: {filt_path!r}",
                file=sys.stderr,
            )
            return 2
    else:
        filt_path = discover_filter_config_yaml(None, HERE)

    filt_ok = bool(filt_path) and os.path.isfile(filt_path)
    if filt_ok and filt_path is not None and not filt_explicit:
        print(
            f"[bundle_scale_pipeline] NOTE: auto-discovered filter config {filt_path!r}",
            file=sys.stderr,
        )

    eff_gscale = int(ns.global_scale_iters)
    requested_gscale = eff_gscale
    eff_run_gp = bool(ns.run_gp)
    skip_ga = bool(ns.skip_global_phot_anchor)

    if skip_ga:
        if requested_gscale > 0:
            print(
                "[bundle_scale_pipeline] NOTE: --skip-global-phot-anchor → relative-only "
                f"(effective global-scale-iters 0; CLI had {requested_gscale}).",
                file=sys.stderr,
            )
        eff_gscale = 0
        eff_run_gp = False
    else:
        if not skip_ga and eff_gscale < 1:
            eff_gscale = 1
            if enrich_env is not None and filt_ok:
                print(
                    "[bundle_scale_pipeline] NOTE: enrich + filter config present; "
                    "using --global-scale-iters=1 (band+synth photometric anchor). "
                    "Use --skip-global-phot-anchor for relative-only output.",
                    file=sys.stderr,
                )
            else:
                print(
                    "[bundle_scale_pipeline] NOTE: using --global-scale-iters=1 (photometric anchor). "
                    "Without enrich + filter YAML, rough / pooled χ² phot anchoring is used (no enrich required). "
                    "Use --skip-global-phot-anchor for relative-only output.",
                    file=sys.stderr,
                )
        if enrich_env is not None and filt_ok and eff_gscale > 0 and not eff_run_gp:
            eff_run_gp = True
            print(
                "[bundle_scale_pipeline] NOTE: enabling --run-gp for inner global-scale GP.",
                file=sys.stderr,
            )

    if eff_gscale > 0 and (enrich_env is None or not filt_ok):
        if bool(ns.rough_phot_anchor):
            print(
                "[bundle_scale_pipeline] WARNING: missing enrich/filter for band-based phot anchor; "
                "falling back to --rough-phot-anchor using phot points at their effective wavelengths.",
                file=sys.stderr,
            )
            eff_run_gp = False
            filt_ok = False
        else:
            tried_e = (
                "pass --enrich /path/to/enrich.npz or place "
                f"{os.path.splitext(os.path.basename(inp_abs))[0]}_enrich.npz or enrich.npz beside the input bundle"
            )
            tried_f = (
                "pass --filter-config /path/to/filters.yaml or add configs/filter_pipeline.yaml "
                "(or rely on configs/filter_pipeline.example.yaml in this repo)"
            )
            print(
                "[bundle_scale_pipeline] ERROR: photometric anchoring is on "
                f"(global-scale-iters={requested_gscale}) but --no-rough-phot-anchor was set and "
                "band+synth prerequisites are missing.\n"
                f"  Enrich: {'ok' if enrich_env is not None else 'missing'} ({tried_e}).\n"
                f"  Filter YAML: {'ok' if filt_ok else 'missing'} ({tried_f}).\n"
                "  Fix: pass --skip-global-phot-anchor or --global-scale-iters 0 for relative-only, "
                "or supply enrich + filter YAML for band anchor, or omit --no-rough-phot-anchor (default).",
                file=sys.stderr,
            )
            return 2

    filt_arg_for_pipeline = filt_path if filt_ok else None

    gn_here = load_grid_norm(ns.input, meta_override=ns.meta)
    summary = apply_scales_pipeline(
        ns.input,
        ns.output,
        gn_here,
        max_bundle_minutes=float(ns.max_bundle_minutes),
        phase_epoch_atol=float(ns.phase_epoch_atol),
        seam_weight=float(ns.seam_weight),
        overlap_grid=int(ns.overlap_grid_points),
        enrich=enrich_env,
        phot_spec_threshold=int(ns.phot_spec_threshold),
        pseudo_band_digits=int(ns.pseudo_band_digits),
        filter_config_path=filt_arg_for_pipeline,
        global_scale_iters=eff_gscale,
        run_gp_flag=eff_run_gp,
        runs_dir=ns.runs_dir,
        gp_tag_prefix=str(ns.gp_tag_prefix),
        gp_extra=gp_extra_argv,
        phase_tolerance_norm_global=float(ns.phase_tolerance_norm_global),
        rough_phot_anchor=bool(ns.rough_phot_anchor),
        rough_phot_phase_window_norm=float(ns.rough_phot_phase_window_norm),
        bundle_phot_pool_max_phase_window_norm=float(ns.bundle_phot_pool_max_phase_window_norm),
        phot_anchor_min_points=int(ns.phot_anchor_min_points),
        seam_band_half_width_aa=float(ns.seam_fit_half_width_aa),
        arm_gap_factor=float(ns.arm_gap_factor),
        arm_min_gap_norm=float(ns.arm_min_gap_norm),
        gap_veto_min_rel_gain=ns.gap_veto_min_rel_gain,
        gap_veto_edge_err_ratio=ns.gap_veto_edge_err_ratio,
    )

    snippet = {"scale_report_written": summary.get("scale_report_written")}
    if summary.get("predictions_gp_tag_latest"):
        snippet["predictions_gp_tag_latest"] = summary["predictions_gp_tag_latest"]

    print("[bundle_scale_pipeline] wrote:")
    print(json.dumps(snippet, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
