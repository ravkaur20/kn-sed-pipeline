#!/usr/bin/env python3
"""Band-by-band photometry light curves + synthetic photometry; bundled spectra + GP slices.

Loads training bundle and optional ``runs/<tag>/predictions.npz`` for GP posterior on
training points (``mu_train``) and on the prediction grid (``mu``, ``X_fill``).

Photometry grouping (display only; ``run_gp`` fits **one likelihood per training row**
with that row's exact ``(x₁, x₂)`` — no wavelength binning in the fit):
  * If ``--enrich`` npz has ``band_name`` / ``band_id``, use real band labels.
  * Else ``--phot-pseudo-grouping rounded`` (default): merge photometry rows whose
    normalized ``X[:,0]`` agree to ``--pseudo-band-digits`` decimals (legacy panels).
  * Else ``--phot-pseudo-grouping unique_x1``: one panel per distinct photometry ``x₁``
    among training rows (optionally round keys with ``--phot-unique-x1-decimals``).

Synthetic photometry (optional): ``--filter-config`` YAML + ``filter_synthesis``;
for each photometric band, spectra at matching epoch (same norm-phase within tol)
are converted to λ (Å), optionally composited, convolved per bandpass. Missing
filters are skipped with a console summary.

Spectral bundles: if the bundle NPZ contains ``spec_bundle_id`` (from
``bundle_scale_pipeline``), panels group **exactly those rows** — matching intra-bundle
scaling. Otherwise phases are clustered in time using ``phase_days`` and
``--bundle-minutes`` (may disagree with the scaler when MJD-based times differ).
When predictions are present, GP overlays use ``LinearNDInterpolator`` on the
``X_fill`` grid so each exposure is compared to **μ(λ | its own phase)**, not a
single slice at a median/snapped grid phase (which inflates residuals when
``|dμ/dt|`` is large). Photometry band panels resample µ along time at
``--phot-lc-time-step-days`` (default **0.05** days or MJD step when enrich has
``mjd``) so light curves reveal true small-scale structure instead of chord
artifacts between sparse epochs. By default the dense curve uses ``x₁`` **tracked**
in time through each band's photometry (``--phot-lc-x1-mode track``) so it matches
the same normalized-wavelength path as the grey points; ``median`` fixes ``x₁`` at
the band median (legacy and can disagree with dots when the GP couples wavelength
and phase).

**Recommended offline order** for “truth as analyzed” overlays:
``bundle_preprocess`` (telluric / photoclass fixes) ``->`` ``bundle_scale_pipeline``
``->`` ``run_gp`` ``->`` this script with ``-b`` set to that same corrected ``*.npz``.

Spectral bundle panels omit ``telluric_bad_mask`` rows and preprocess-disabled pixels
(inflated ``yerr``); use ``--overlap-scale`` only for within-exposure λ-arm alignment
(off by default so pipeline-calibrated bundles are shown faithfully).

Writes PNGs under ``--output-dir`` (default ``<runs-dir>/<tag>/figs/overview`` when ``--tag``
is set; ``--runs-dir`` defaults to ``<repo>/runs`` and should match ``run_gp --output-dir``).
When predictions are loaded, each spectral bundle also writes ``spec_bundle_<id>_pairs.png``:
MST edges from ``intra_bundle_epoch_scale_trace`` (same solver inputs as
``intra_bundle_epoch_scales``): raw composites dashed, solid = ref / mov×``s`` **as passed to**
``solve_pair_scale``; narrow overlap + seam shading.

Examples::

    python plot_bands_gp_overview.py --bundle gp_minimal_bundle.npz --tag matern52_linear_opt

    python plot_bands_gp_overview.py -b gp_minimal_bundle.npz \\
        -p runs/my_run/predictions.npz --out-dir ./overview_figs \\
        --filter-config configs/filter_pipeline.example.yaml --enrich enrich.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Literal, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import bundle_meta as bmeta
import bundle_preprocess as bpre
import bundle_scale_pipeline as bsp
import gp_grid_interp as ggi
import gp_utils as gu
import spectrum_bundles as sb
from filter_synthesis import (
    load_filter_config,
    resolve_bandpasses,
    save_filter_report,
    summarize_report,
    synthesize_effstim,
)
from plot_results import (
    _build_scaled_spec_overlay_rows,
    denorm_ln_wavelength,
    linear_flux_yerr,
    norm_x2_from_phase_days,
    phase_days_from_norm_x2,
    scatter_train_vector_to_bundle,
    scaled_ln_to_linear,
    wl_linear_aa_to_plot_x1,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")
DEFAULT_RUNS = os.path.join(HERE, "runs")


def _wl_norm_support_union(subs: list[dict], *, pad: float) -> tuple[float, float]:
    """Union of normalized-log10-λ coverage for overlay segments (one spectroscopic phase)."""
    if not subs:
        return float("nan"), float("nan")
    lo = min(float(np.min(np.asarray(sg["wl_norm"], dtype=float))) for sg in subs)
    hi = max(float(np.max(np.asarray(sg["wl_norm"], dtype=float))) for sg in subs)
    span = max(hi - lo, 1e-9)
    p = float(pad) * span if float(pad) < 1.0 else float(pad)
    return lo - p, hi + p


def _latent_gp_interpolators_from_fill(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: Optional[np.ndarray],
) -> tuple[LinearNDInterpolator, Optional[LinearNDInterpolator]]:
    """Bilinear-style latent interpolation on the Cartesian ``X_fill`` grid (via triangulation).

    Used so spectral-bundle panels compare each exposure to **μ(λ | phase=that exposure)**,
    instead of a single GP slice at a median/snapped phase (which biases residuals when
    ``|dμ/dt|`` is large, e.g. early times).
    """
    xf = np.asarray(X_fill, dtype=float).reshape(-1, 2)
    m = np.asarray(mu, dtype=float).ravel()
    if m.shape[0] != xf.shape[0]:
        raise ValueError("mu length must match X_fill rows")
    it_mu = LinearNDInterpolator(xf, m, fill_value=np.nan)
    it_std: Optional[LinearNDInterpolator] = None
    if std is not None:
        s = np.asarray(std, dtype=float).ravel()
        if s.shape[0] == xf.shape[0]:
            it_std = LinearNDInterpolator(xf, s, fill_value=np.nan)
    return it_mu, it_std


def _epoch_ids_for_bundle_phases(
    canonical_phases: np.ndarray,
    phases_bundle: np.ndarray,
    *,
    spec_phase_decimals: int,
) -> list[int]:
    """Map rounded training phases in a bundle to spectroscopic epoch indices."""
    if int(spec_phase_decimals) < 0:
        return []
    digits = int(spec_phase_decimals)
    out: list[int] = []
    for ph in np.asarray(phases_bundle, dtype=float).ravel():
        rq = float(np.round(float(ph), digits))
        hit: int | None = None
        for e in range(int(canonical_phases.size)):
            if float(np.round(float(canonical_phases[e]), digits)) == rq:
                hit = int(e)
                break
        if hit is not None:
            out.append(hit)
    uniq: list[int] = []
    for e in out:
        if e not in uniq:
            uniq.append(e)
    return uniq


def _shade_pair_scale_regions(ax: Any, geom: dict[str, Any], gn: dict, *, with_labels: bool) -> None:
    """Shade overlap (narrow band for χ² locality) + seam affine bands; x matches spectral overview."""
    lbl_ov = "overlap χ² (±seam half-width, clipped)" if with_labels else "_nolegend_"
    lbl_rb = "ref seam band (affine)" if with_labels else "_nolegend_"
    lbl_mb = "mov seam band (affine)" if with_labels else "_nolegend_"
    ov = geom.get("overlap_shade_aa")
    if ov is not None and isinstance(ov, (tuple, list)) and float(ov[1]) > float(ov[0]) + 1e-9:
        xs = wl_linear_aa_to_plot_x1(np.asarray([float(ov[0]), float(ov[1])], dtype=float), gn)
        ax.axvspan(float(np.min(xs)), float(np.max(xs)), color="green", alpha=0.18, label=lbl_ov)
    rb = geom["ref_seam_band_aa"]
    mb = geom["mov_seam_band_aa"]
    xrb = wl_linear_aa_to_plot_x1(np.asarray([float(rb[0]), float(rb[1])], dtype=float), gn)
    ax.axvspan(float(np.min(xrb)), float(np.max(xrb)), color="orange", alpha=0.16, label=lbl_rb)
    xmb = wl_linear_aa_to_plot_x1(np.asarray([float(mb[0]), float(mb[1])], dtype=float), gn)
    ax.axvspan(float(np.min(xmb)), float(np.max(xmb)), color="cyan", alpha=0.13, label=lbl_mb)


def _flux_ylim_from_series(ax: Any, series: list[np.ndarray]) -> None:
    """Avoid clipping sparse arms: use min/max of all plotted flux (tiny padding)."""
    parts = [np.asarray(s, dtype=float).ravel() for s in series if np.asarray(s, dtype=float).size]
    if not parts:
        return
    yy = np.concatenate(parts)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return
    lo, hi = float(np.nanmin(yy)), float(np.nanmax(yy))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    pad = 0.04 * (hi - lo + 1e-30)
    ax.set_ylim(lo - pad, hi + pad)


def _write_spec_bundle_pair_scaling_figure(
    *,
    bundle_id: int,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    canonical_phases: np.ndarray,
    epoch_of_row: np.ndarray,
    phases_bundle: np.ndarray,
    out_dir: str,
    phase_epoch_atol: float,
    seam_weight: float,
    overlap_grid_points: int,
    seam_fit_half_width_aa: float,
    spec_phase_decimals: int,
) -> None:
    """MST propagation panels from ``intra_bundle_epoch_scale_trace`` (exact solver inputs).

    Sequential raw pairwise scaling is wrong for n≥3 because later edges use **cumulative**
    multiplier on one side ``intra_bundle_epoch_scales``); this matches the scaler.
    """
    if int(spec_phase_decimals) < 0:
        return
    e_ids = _epoch_ids_for_bundle_phases(
        canonical_phases, phases_bundle, spec_phase_decimals=int(spec_phase_decimals)
    )
    if len(e_ids) < 2:
        return
    _, trace, elist_o, _ = bsp.intra_bundle_epoch_scale_trace(
        X,
        y,
        yerr,
        gn,
        canonical_phases,
        np.asarray(e_ids, dtype=int),
        epoch_of_row,
        phase_atol=float(phase_epoch_atol),
        seam_weight=float(seam_weight),
        overlap_grid_points=int(overlap_grid_points),
        seam_band_half_width_aa=float(seam_fit_half_width_aa),
    )
    if not trace:
        return

    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ee in elist_o:
        w, f, er = bsp.composite_epoch_linear(
            X,
            y,
            yerr,
            gn,
            bsp.canon_phase(canonical_phases, int(ee)),
            phase_atol=float(phase_epoch_atol),
            epoch_of_row=epoch_of_row,
            epoch_id=int(ee),
        )
        data[int(ee)] = (w, f, er)

    n_trace = len(trace)
    n_p = n_trace
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, max(len(elist_o), 2)))
    ep_to_ci = {int(elist_o[i]): i for i in range(len(elist_o))}

    fig, axes = plt.subplots(n_p, 1, figsize=(11, 4.0 * n_p), sharex=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    for pi, row in enumerate(trace):
        ax = axes[pi]
        e_lo = int(row["epoch_bluer"])
        e_hi = int(row["epoch_redder"])
        wl_b = np.asarray(row["wl_bluer_aa"], dtype=float)
        f_bs = np.asarray(row["f_bluer_in"], dtype=float).ravel()
        wl_r = np.asarray(row["wl_redder_aa"], dtype=float)
        fr_in = np.asarray(row["f_redder_in"], dtype=float).ravel()
        s_ij = float(row["s_ij"])
        case = str(row.get("case", ""))
        geom = row["geometry"]
        mode = str(geom["mode"])

        wl_raw_lo, f_raw_lo, _ = data[e_lo]
        wl_raw_hi, f_raw_hi, _ = data[e_hi]
        c_lo = colors[ep_to_ci[e_lo]]
        c_hi = colors[ep_to_ci[e_hi]]

        x_rlo = wl_linear_aa_to_plot_x1(wl_raw_lo, gn)
        x_rhi = wl_linear_aa_to_plot_x1(wl_raw_hi, gn)
        x_sb = wl_linear_aa_to_plot_x1(wl_b, gn)
        x_sr = wl_linear_aa_to_plot_x1(wl_r, gn)

        ax.plot(x_rlo, f_raw_lo, "--", color=c_lo, lw=1.15, alpha=0.5, label=f"epoch {e_lo} composite (raw)")
        ax.plot(x_rhi, f_raw_hi, "--", color=c_hi, lw=1.15, alpha=0.5, label=f"epoch {e_hi} composite (raw)")
        ax.plot(
            x_sb,
            f_bs,
            "-",
            color=c_lo,
            lw=2.2,
            alpha=0.95,
            label=f"epoch {e_lo} ref (as in solver)",
        )
        ax.plot(
            x_sr,
            fr_in * s_ij,
            "-",
            color=c_hi,
            lw=2.2,
            alpha=0.95,
            label=f"epoch {e_hi} mov×s (solver)",
        )
        _shade_pair_scale_regions(ax, geom, gn, with_labels=(pi == 0))
        ax.set_ylabel("flux (linear)")
        ax.set_title(
            f"Spectral bundle {bundle_id} — MST edge {pi + 1}/{n_trace}: {e_lo}→{e_hi} ({case}); "
            f"s={s_ij:.6g}; mode={mode}"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        _flux_ylim_from_series(ax, [f_raw_lo, f_raw_hi, f_bs, fr_in * s_ij])
        print(
            f"[overview] spec_bundle_{bundle_id}_pairs.png  panel {pi + 1}: MST {e_lo}→{e_hi}  "
            f"s={s_ij:.12g}  mode={mode}  case={case}  overlap_extent_aa={geom.get('overlap_aa')}  "
            f"overlap_shade_aa={geom.get('overlap_shade_aa')}"
        )

    axes[-1].set_xlabel("log10(wavelength)")
    fig.tight_layout()
    pair_path = os.path.join(out_dir, f"spec_bundle_{int(bundle_id)}_pairs.png")
    fig.savefig(pair_path, dpi=150)
    plt.close(fig)
    print(f"[overview] wrote {pair_path}")


def _canonical_phases(vals: np.ndarray, *, atol: float = 5e-6) -> np.ndarray:
    arr = np.sort(np.asarray(vals, dtype=float).ravel())
    out: list[float] = []
    for v in arr:
        if not out or all(abs(float(v) - u) > float(atol) for u in out):
            out.append(float(v))
    return np.asarray(out, dtype=float)


def _load_gn(bundle_path: str, meta_path: Optional[str], config_path: Optional[str]) -> dict:
    """Prefer meta tied to the **bundle** for ln-flux denorm; fall back to run ``config.json``.

    Using ``runs/<tag>/config.json`` first breaks ``--bundle`` vs ``--tag`` mismatches: training
    ``y`` is always written for the bundle's collaborator scaling, while an old run config can
    embed a different ``grid_norm_info`` snapshot.
    """
    babs = os.path.abspath(os.path.expanduser(str(bundle_path)))
    bdir = os.path.dirname(babs)
    bname = os.path.splitext(os.path.basename(babs))[0]
    meta_resolved = meta_path
    if not meta_resolved:
        candidates = [
            os.path.join(bdir, f"{bname}_meta.json"),
            os.path.join(HERE, f"{bname}_meta.json"),
            os.path.join(HERE, "gp_scaled_bundle_meta.json"),
            os.path.join(HERE, "gp_minimal_bundle_meta.json"),
            os.path.join(HERE, "gp_bundle_corrected_meta.json"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                meta_resolved = cand
                break
    gn_bundle = bmeta.grid_norm_from_bundle_or_meta(bundle_path, meta_path=meta_resolved)
    if not gn_bundle.get("_normalized_only"):
        return gn_bundle
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("grid_norm_info"), dict):
            gn = dict(cfg["grid_norm_info"])
            gn["_normalized_only"] = False
            return gn
    return gn_bundle


def _load_enrich(path: Optional[str]) -> Optional[dict[str, np.ndarray]]:
    if not path:
        return None
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        print(f"[overview] WARNING: enrich missing {p!r}", file=sys.stderr)
        return None
    d = np.load(p, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def _phot_band_labels(enrich: Optional[dict], n: int) -> np.ndarray:
    labels = np.full(n, "", dtype=object)
    if enrich is None:
        return labels
    if "band_name" in enrich:
        for i in range(min(n, enrich["band_name"].shape[0])):
            labels[i] = str(enrich["band_name"][i])
    elif "band_id" in enrich:
        for i in range(min(n, enrich["band_id"].shape[0])):
            labels[i] = f"id_{int(enrich['band_id'][i])}"
    return labels


def _time_axis(X: np.ndarray, gn: dict, enrich: Optional[dict]) -> np.ndarray:
    if enrich is not None and "mjd" in enrich:
        m = np.asarray(enrich["mjd"], dtype=float).ravel()
        if m.size >= X.shape[0]:
            return m[: X.shape[0]]
    return phase_days_from_norm_x2(X[:, 1], gn)


def _pseudo_band_key(x1: np.ndarray, nd: int = 4) -> np.ndarray:
    return np.round(np.asarray(x1, dtype=float), nd)


def photometry_pseudo_wavelength_groups(
    X: np.ndarray,
    phot_row_indices: np.ndarray,
    gn: dict,
    *,
    grouping: str,
    pseudo_band_digits: int,
    unique_x1_decimals: int,
    max_unique_panels: int,
) -> dict[str, np.ndarray]:
    """Build photometry panel groups when enrich band labels are absent.

    ``grouping='rounded'``: legacy — merge rows whose ``X[:,0]`` agree to ``pseudo_band_digits`` decimals.

    ``grouping='unique_x1'``: one panel per distinct training ``x₁`` (normalized log₁₀ λ proxy) among
    photometry rows — matches how ``run_gp`` sees each design row. Optional rounding before ``np.unique``
    only affects the **panel key** (``unique_x1_decimals``), not the stored ``X`` values.
    """
    phot_row_indices = np.asarray(phot_row_indices, dtype=int).ravel()
    if phot_row_indices.size == 0:
        return {}
    if grouping == "rounded":
        pk = _pseudo_band_key(X[phot_row_indices, 0], int(pseudo_band_digits))
        groups_dd: dict[float, list[int]] = defaultdict(list)
        for pos, r in enumerate(phot_row_indices.tolist()):
            groups_dd[float(pk[pos])].append(int(r))
        return {f"log10λ_norm≈{k:.4f}": np.asarray(v, dtype=int) for k, v in groups_dd.items()}

    if grouping != "unique_x1":
        raise ValueError(f"grouping must be 'rounded' or 'unique_x1'; got {grouping!r}")

    x1 = np.asarray(X[phot_row_indices, 0], dtype=float).ravel()
    if int(unique_x1_decimals) >= 0:
        x1k = np.round(x1, int(unique_x1_decimals))
    else:
        x1k = x1.copy()
    u, inv = np.unique(x1k, return_inverse=True)
    if int(max_unique_panels) > 0 and u.size > int(max_unique_panels):
        print(
            f"[overview] WARN: unique_x1 grouping yields {u.size} photometry panels "
            f"(>{max_unique_panels}); consider --phot-unique-x1-decimals 4 or use enrich band labels",
            file=sys.stderr,
        )
    out: dict[str, np.ndarray] = {}
    for j in range(u.size):
        lp = float(denorm_ln_wavelength(np.asarray([u[j]]), gn)[0])
        lab = f"log10λ_phys≈{lp:.6f} x1_norm={u[j]:.12g}"
        out[lab] = phot_row_indices[inv == j]
    return out


def _smooth_sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return x, y
    o = np.argsort(x)
    return x[o], y[o]


def _norm_x2_for_dense_time_axis(
    t_dense: np.ndarray,
    tt_band: np.ndarray,
    x2_band: np.ndarray,
    gn: dict,
    enrich: Optional[dict],
) -> np.ndarray:
    """Map dense time-axis values (MJD or phase days) to normalized phase column ``x₂``."""
    td = np.asarray(t_dense, dtype=float).ravel()
    tt_b = np.asarray(tt_band, dtype=float).ravel()
    x2_b = np.asarray(x2_band, dtype=float).ravel()
    if enrich is not None and "mjd" in enrich:
        o = np.argsort(tt_b)
        return np.interp(td, tt_b[o], x2_b[o], left=float("nan"), right=float("nan"))
    return norm_x2_from_phase_days(td, gn)


def _x1_on_dense_time_grid(t_dense: np.ndarray, tt: np.ndarray, x1_band: np.ndarray) -> np.ndarray:
    """Interpolate normalized log-λ ``x₁`` onto ``t_dense`` along the band's photometry in time.

    Duplicate times (same ``tt``) are collapsed by averaging ``x₁`` so ``np.interp`` is well-defined.
    """
    td = np.asarray(t_dense, dtype=float).ravel()
    tt_b = np.asarray(tt, dtype=float).ravel()
    x1_b = np.asarray(x1_band, dtype=float).ravel()
    ok = np.isfinite(tt_b) & np.isfinite(x1_b)
    tt_b, x1_b = tt_b[ok], x1_b[ok]
    if tt_b.size == 0:
        return np.full(td.shape[0], np.nan, dtype=float)
    o = np.argsort(tt_b)
    tt_s, x1_s = tt_b[o], x1_b[o]
    tu, inv = np.unique(tt_s, return_inverse=True)
    if tu.size == tt_s.size:
        x1u = x1_s
    else:
        sums = np.zeros(tu.size, dtype=float)
        cnts = np.zeros(tu.size, dtype=int)
        for j in range(tt_s.size):
            sums[inv[j]] += x1_s[j]
            cnts[inv[j]] += 1
        x1u = sums / np.maximum(cnts, 1)
    return np.interp(td, tu, x1u, left=float(x1u[0]), right=float(x1u[-1]))


def _dense_phot_gp_slice(
    X: np.ndarray,
    idx: np.ndarray,
    tt: np.ndarray,
    preds_npz: Any,
    gn: dict,
    enrich: Optional[dict],
    *,
    posterior_kind: PosteriorKind,
    time_step: float,
    x1_mode: str = "track",
) -> Optional[dict[str, np.ndarray]]:
    """Dense (t, µ_linear) along the band's time axis via 2D latent GP interpolation.

    ``x1_mode='track'`` (default): at each dense time, use ``x₁`` interpolated from the band's
    photometry in time so the curve lies on the same (phase, wavelength) manifold as the grey
    points. ``x1_mode='median'`` fixes ``x₁`` to the band median (legacy; can disagree with dots
    when the GP is correlated across normalized wavelength).
    """
    if time_step <= 0 or preds_npz is None or idx.size == 0:
        return None
    fs = preds_npz.files
    tlo = float(np.nanmin(tt))
    thi = float(np.nanmax(tt))
    if not (np.isfinite(tlo) and np.isfinite(thi) and thi > tlo):
        return None
    step = float(time_step)
    t_dense = np.arange(tlo, thi + 0.5 * step, step, dtype=float)
    if t_dense.size < 2:
        t_dense = np.asarray([tlo, thi], dtype=float)

    x2_q = _norm_x2_for_dense_time_axis(t_dense, tt, X[idx, 1], gn, enrich)
    ok = np.isfinite(x2_q)
    if not np.any(ok):
        return None
    t_dense = t_dense[ok]
    x2_q = x2_q[ok]
    x1_all = np.asarray(X[idx, 0], dtype=float)
    if str(x1_mode).lower() == "median":
        x1_col = np.full(t_dense.shape[0], float(np.median(x1_all)), dtype=float)
    else:
        x1_col = _x1_on_dense_time_grid(t_dense, tt, x1_all)
    X_query = np.column_stack([x1_col, x2_q])

    try:
        if posterior_kind == "train":
            if "mu_train" not in fs:
                return None
            mu_ref = scatter_train_vector_to_bundle(preds_npz, preds_npz["mu_train"], int(X.shape[0]))
            if mu_ref is None:
                return None
            ok_rows = np.isfinite(mu_ref)
            if not np.any(ok_rows):
                return None
            mu_lat = _interp_latent_gp_on_rows(X[ok_rows], mu_ref[ok_rows], X_query)
        else:
            if "X_fill" not in fs:
                return None
            X_fill = np.asarray(preds_npz["X_fill"], dtype=float)
            mu_key = "mu_raw" if posterior_kind == "grid_raw" and "mu_raw" in fs else "mu"
            if mu_key not in fs:
                return None
            mu_fill = np.asarray(preds_npz[mu_key], dtype=float).ravel()
            mu_lat = _interp_latent_gp_on_rows(X_fill, mu_fill, X_query)

        sig_lat: Optional[np.ndarray] = None
        if posterior_kind == "train":
            sig_key = "std_train" if "std_train" in fs else ("sigma_eff_train" if "sigma_eff_train" in fs else None)
            if sig_key is not None:
                sig_ref = scatter_train_vector_to_bundle(preds_npz, preds_npz[sig_key], int(X.shape[0]))
                if sig_ref is None:
                    return None
                ok_s = ok_rows & np.isfinite(sig_ref)
                if np.any(ok_s):
                    sig_lat = _interp_latent_gp_on_rows(X[ok_s], sig_ref[ok_s], X_query)
        else:
            X_fill = np.asarray(preds_npz["X_fill"], dtype=float)
            if posterior_kind == "grid_raw" and "std_raw" in fs:
                sig_fill = np.asarray(preds_npz["std_raw"], dtype=float).ravel()
            elif "std" in fs:
                sig_fill = np.asarray(preds_npz["std"], dtype=float).ravel()
            else:
                sig_fill = None
            if sig_fill is not None and sig_fill.shape[0] == X_fill.shape[0]:
                sig_lat = _interp_latent_gp_on_rows(X_fill, sig_fill, X_query)

    except Exception as exc:  # noqa: BLE001 — Delaunay / degenerate hull in pathological bands
        print(f"[overview] WARN: dense phot GP skipped ({exc})", file=sys.stderr)
        return None

    gp_lin_dense = scaled_ln_to_linear(mu_lat, gn)
    out: dict[str, np.ndarray] = {"t": t_dense, "gp_lin": gp_lin_dense}
    if sig_lat is not None:
        sd = np.asarray(sig_lat, dtype=float).ravel()
        sd[~np.isfinite(sd)] = 1e-6
        out["sig_lin"] = linear_flux_yerr(mu_lat.astype(float), sd, gn).ravel()
    return out


PosteriorKind = Literal["train", "grid_pp", "grid_raw"]


def _interp_latent_gp_on_rows(X_fill: np.ndarray, latent_vec: np.ndarray, X_rows: np.ndarray) -> np.ndarray:
    out, _n_nn = ggi.interp_latent_gp_at_fill_rows(X_fill, latent_vec, X_rows)
    return out


def interp_latent_gp_linear_only(
    X_sites: np.ndarray,
    latent_vec: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    """``LinearNDInterpolator`` on ``(X_sites, latent_vec)`` with no nearest-neighbor fill.

    Values are finite on the convex hull of ``X_sites`` and NaN outside (or on degenerate
    triangulation). Used by diagnostics to detect hull-exit artifacts on dense phot curves.
    """
    latent_vec = np.asarray(latent_vec, dtype=float).ravel()
    Xs = np.asarray(X_sites, dtype=float).reshape(-1, 2)
    if Xs.shape[0] != latent_vec.shape[0]:
        raise ValueError("X_sites rows must match latent_vec length")
    Xq = np.asarray(X_query, dtype=float).reshape(-1, 2)
    lut = LinearNDInterpolator(Xs, latent_vec, fill_value=np.nan)
    z = lut(Xq)
    return np.asarray(z.ravel(), dtype=float)


def build_comparison_mu_lin(
    preds_npz: Any,
    X: np.ndarray,
    gn: dict,
    *,
    posterior_kind: PosteriorKind,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return linear-flux µ and ±1σ surrogates for residuals at every training row."""
    if preds_npz is None or X.shape[0] == 0:
        return None, None
    fs = preds_npz.files

    def _lat_lin(mu_lat: np.ndarray, sig_lat_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sd = np.asarray(sig_lat_hint, dtype=float).ravel().copy()
        sd[~np.isfinite(sd)] = 1e-6
        mu_lin_arr = scaled_ln_to_linear(mu_lat.astype(float), gn).ravel()
        sig_lin_arr = linear_flux_yerr(mu_lat.astype(float), sd, gn).ravel()
        return mu_lin_arr, sig_lin_arr

    if posterior_kind == "train":
        if "mu_train" not in fs:
            print("[overview] WARN: posterior-kind=train but mu_train missing", file=sys.stderr)
            return None, None

        mt = np.asarray(preds_npz["mu_train"], dtype=float).reshape(-1)
        sig_lat = preds_npz["std_train"] if "std_train" in fs else preds_npz["sigma_eff_train"]

        sigma_lat_any = np.asarray(sig_lat, dtype=float).reshape(-1)
        if sigma_lat_any.size != mt.size:
            print("[overview] WARN: posterior-kind=train latent σ length mismatch; skip residual mapping", file=sys.stderr)
            return None, None
        if "train_row_index_orig" in fs and int(mt.size) != int(X.shape[0]):
            oi = np.asarray(preds_npz["train_row_index_orig"], dtype=np.int64).ravel()
            if oi.size != mt.size or int(np.min(oi)) < 0 or int(np.max(oi)) >= X.shape[0]:
                print(
                    "[overview] WARN: train_row_index_orig inconsistent with bundle X; skip residual mapping",
                    file=sys.stderr,
                )
                return None, None
            mfull = np.full(X.shape[0], np.nan, dtype=float)
            sfull = np.full(X.shape[0], np.nan, dtype=float)
            mfull[oi] = mt
            sfull[oi] = sigma_lat_any
            mt, sigma_lat_any = mfull, sfull
        return _lat_lin(mt, sigma_lat_any)

    if posterior_kind != "train" and ("X_fill" not in fs or "mu" not in fs):
        print("[overview] WARN: grid posterior-kind needs X_fill+mu predictions", file=sys.stderr)
        return None, None

    X_fill = np.asarray(preds_npz["X_fill"], dtype=float)
    latent_mu = preds_npz["mu_raw"] if posterior_kind == "grid_raw" and "mu_raw" in fs else preds_npz["mu"]
    latent_mu = np.asarray(latent_mu, dtype=float).ravel()
    mu_lat_interp = _interp_latent_gp_on_rows(X_fill, latent_mu, X)

    if posterior_kind == "grid_raw" and "std_raw" in fs:
        sig_field = np.asarray(preds_npz["std_raw"], dtype=float).ravel()
    else:
        if "std" not in fs:
            print("[overview] WARN: missing std alongside grid µ; residuals omitted", file=sys.stderr)
            return None, None
        sig_field = np.asarray(preds_npz["std"], dtype=float).ravel()

    sig_lat_interp = _interp_latent_gp_on_rows(X_fill, sig_field, X.astype(float))

    return _lat_lin(mu_lat_interp, sig_lat_interp)


def _spectrum_composite_at_phase(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    phase_norm: float,
    spec_mask: np.ndarray,
    *,
    phase_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wave_angstrom sorted, linear flux) for all spec rows near phase_norm."""
    m = spec_mask & np.isfinite(X[:, 0]) & np.isfinite(y)
    m &= np.isclose(X[:, 1], phase_norm, rtol=0.0, atol=phase_tol)
    if not np.any(m):
        return np.array([]), np.array([])
    rows = np.nonzero(m)[0]
    wave_log_phys = denorm_ln_wavelength(X[rows, 0], gn)
    wave_aa = np.power(10.0, wave_log_phys)
    f_lin = scaled_ln_to_linear(y[rows], gn)
    o = np.argsort(wave_aa)
    return wave_aa[o], f_lin[o]



def plot_photometry_bands(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    enrich: Optional[dict],
    mu_train: Optional[np.ndarray],
    out_dir: str,
    *,
    filter_cfg_path: Optional[str],
    synth_phase_tol: float,
    pseudo_band_digits: int,
    filter_report_path: Optional[str],
    comparison_mu_lin: Optional[np.ndarray] = None,
    comparison_sigma_lin: Optional[np.ndarray] = None,
    plot_residual_vs_gp: bool = False,
    posterior_legend_suffix: str = "",
    preds_npz: Any = None,
    posterior_kind: PosteriorKind = "train",
    phot_lc_time_step: float = 0.05,
    phot_lc_x1_mode: str = "track",
    phot_pseudo_grouping: str = "rounded",
    phot_unique_x1_decimals: int = 12,
    phot_max_unique_panels_warn: int = 500,
) -> None:
    phot_m = point_class == gu.PHOT
    if not np.any(phot_m):
        print("[overview] no photometry rows")
        return

    labels = _phot_band_labels(enrich, X.shape[0])
    use_real = enrich is not None and np.any(labels[phot_m] != "")

    groups: dict[str, np.ndarray]
    if use_real:
        groups = defaultdict(list)
        for i in np.nonzero(phot_m)[0]:
            lab = str(labels[i]) if labels[i] else "unknown"
            groups[lab].append(i)
        groups = {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    else:
        phot_rows = np.nonzero(phot_m)[0]
        groups = photometry_pseudo_wavelength_groups(
            X,
            phot_rows,
            gn,
            grouping=str(phot_pseudo_grouping),
            pseudo_band_digits=int(pseudo_band_digits),
            unique_x1_decimals=int(phot_unique_x1_decimals),
            max_unique_panels=int(phot_max_unique_panels_warn),
        )

    band_aliases: dict[str, str] = {}
    trds_roots: list[str] = []
    synth_system = "ab"
    bandpasses: dict[str, Any] = {}
    if filter_cfg_path and os.path.isfile(filter_cfg_path):
        trds_roots, band_aliases, synth_system = load_filter_config(filter_cfg_path)
        all_band_keys = list(groups.keys())
        bandpasses, rep0 = resolve_bandpasses(all_band_keys, band_aliases=band_aliases, trds_roots=trds_roots)
        if filter_report_path:
            save_filter_report(filter_report_path, rep0)
            print(summarize_report(rep0))

    spec_m = point_class == gu.SPEC

    for lab, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if idx.size == 0:
            continue
        t_ax = _time_axis(X, gn, enrich)
        flux = scaled_ln_to_linear(y[idx], gn)
        ferr = linear_flux_yerr(y[idx], yerr[idx], gn)
        tt = t_ax[idx]

        want_resid = bool(plot_residual_vs_gp and comparison_mu_lin is not None)
        if want_resid:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(10, 7.6),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.72]},
            )
            ax_top, ax_bottom = axes[0], axes[1]
        else:
            fig, ax_top = plt.subplots(figsize=(10, 4.5))
            ax_bottom = None

        ax_top.errorbar(tt, flux, yerr=ferr, fmt="o", ms=5, color="0.35", alpha=0.65, label="photometry")

        dense_pack = None
        if phot_lc_time_step > 0 and preds_npz is not None:
            dense_pack = _dense_phot_gp_slice(
                X,
                idx,
                tt,
                preds_npz,
                gn,
                enrich,
                posterior_kind=posterior_kind,
                time_step=phot_lc_time_step,
                x1_mode=phot_lc_x1_mode,
            )

        gp_lin = None
        leg = "GP"
        if comparison_mu_lin is not None:
            gp_lin = np.asarray(comparison_mu_lin[idx], dtype=float)
            leg = f"GP ({posterior_legend_suffix})" if posterior_legend_suffix else "GP (comparison)"
        elif mu_train is not None:
            gp_lin = scaled_ln_to_linear(mu_train[idx], gn)
            leg = "GP (mu_train)"

        if dense_pack is not None:
            ax_top.plot(
                dense_pack["t"],
                dense_pack["gp_lin"],
                "-",
                color="steelblue",
                lw=2.0,
                alpha=0.95,
                label=leg,
            )
        elif gp_lin is not None:
            txs, gpy = _smooth_sorted(tt, gp_lin)
            ax_top.plot(txs, gpy, "-", color="steelblue", lw=2.0, alpha=0.95, label=leg)

        bp = bandpasses.get(lab)
        if bp is not None:
            ts_syn: list[float] = []
            mag_syn: list[float] = []
            uniq_spec_ph = np.unique(X[spec_m, 1])
            for i_row in idx:
                ph_n = float(X[i_row, 1])
                if uniq_spec_ph.size == 0:
                    break
                j = int(np.argmin(np.abs(uniq_spec_ph - ph_n)))
                if np.abs(uniq_spec_ph[j] - ph_n) > synth_phase_tol:
                    continue
                w_aa, f_lin = _spectrum_composite_at_phase(
                    X,
                    y,
                    yerr,
                    gn,
                    float(uniq_spec_ph[j]),
                    spec_m,
                    phase_tol=max(synth_phase_tol * 0.25, 1e-9),
                )
                if w_aa.size < 3:
                    continue
                val, err = synthesize_effstim(w_aa, f_lin, bp, system=synth_system)
                if val is None or err is not None:
                    continue
                ts_syn.append(float(t_ax[i_row]))
                mag_syn.append(float(val))
            if ts_syn:
                ax_syn = ax_top.twinx()
                txs_m, mag_s = _smooth_sorted(np.asarray(ts_syn), np.asarray(mag_syn))
                ax_syn.plot(txs_m, mag_s, "s-", color="darkgreen", ms=4, lw=1.5, alpha=0.9, label="synth (spectra)")
                ax_syn.set_ylabel(f"synthetic phot ({synth_system.upper()} mag)", color="darkgreen")
                ax_syn.tick_params(axis="y", labelcolor="darkgreen")
                ax_syn.invert_yaxis()

        ax_top.set_ylabel("flux (linear)")
        ax_top.set_title(f"Photometry — {lab[:90]}")
        ax_top.grid(alpha=0.25)
        ax_top.legend(fontsize=8, loc="best")

        xlabel = "MJD" if enrich is not None and "mjd" in enrich else "phase (days)"
        if ax_bottom is not None and gp_lin is not None:
            o = np.argsort(tt)
            tt_s = tt[o]
            flux_s = flux[o]
            gp_s = gp_lin[o]
            if dense_pack is not None:
                t_d = np.asarray(dense_pack["t"], dtype=float)
                gp_d = np.asarray(dense_pack["gp_lin"], dtype=float)
                flux_d = np.interp(t_d, tt_s, flux_s, left=float("nan"), right=float("nan"))
                tiny = np.finfo(float).tiny
                rat_d = (flux_d - gp_d) / np.maximum(np.abs(gp_d), np.fmax(np.abs(flux_d) * 0.05, tiny))
                m = np.isfinite(rat_d) & np.isfinite(flux_d)
                ax_bottom.plot(
                    t_d[m],
                    rat_d[m],
                    "-",
                    color="indianred",
                    lw=1.4,
                    alpha=0.9,
                    label="(data−µ)/|µ|",
                )
                norm_sig_d: Optional[np.ndarray] = None
                if dense_pack.get("sig_lin") is not None:
                    sig_ld = np.asarray(dense_pack["sig_lin"], dtype=float)
                    norm_sig_d = sig_ld / np.maximum(np.abs(gp_d), tiny)
                elif comparison_sigma_lin is not None:
                    sig_c = np.asarray(comparison_sigma_lin[idx], dtype=float)[o]
                    norm_sig = sig_c / np.maximum(np.abs(gp_s), tiny)
                    norm_sig_d = np.interp(t_d, tt_s, norm_sig, left=float("nan"), right=float("nan"))
                if norm_sig_d is not None:
                    mm = m & np.isfinite(norm_sig_d)
                    ax_bottom.fill_between(
                        t_d[mm],
                        -norm_sig_d[mm],
                        norm_sig_d[mm],
                        color="indianred",
                        alpha=0.12,
                        label="±σ/|µ|",
                    )
            else:
                rat = (flux_s - gp_s) / np.maximum(np.abs(gp_s), np.fmax(np.abs(flux_s) * 0.05, np.finfo(float).tiny))
                txs_r, rat_s = _smooth_sorted(tt_s, rat)
                ax_bottom.plot(txs_r, rat_s, "-", color="indianred", lw=1.4, alpha=0.9, label="(data−µ)/|µ|")
                if comparison_sigma_lin is not None:
                    sig_c = np.asarray(comparison_sigma_lin[idx], dtype=float)[o]
                    norm_sig = sig_c / np.maximum(np.abs(gp_s), np.finfo(float).tiny)
                    t_sig, sig_s = _smooth_sorted(tt_s, norm_sig)
                    sig_on_r = np.interp(txs_r, t_sig, sig_s)
                    sig_on_r = np.nan_to_num(sig_on_r, nan=0.0, posinf=0.0, neginf=0.0)
                    ax_bottom.fill_between(
                        txs_r,
                        -sig_on_r,
                        sig_on_r,
                        color="indianred",
                        alpha=0.12,
                        label="±σ/|µ|",
                    )
            ax_bottom.axhline(0.0, color="0.5", lw=0.6)
            ax_bottom.set_ylabel("relative residual")
            ax_bottom.set_xlabel(xlabel)
            ax_bottom.grid(alpha=0.25)
            ax_bottom.legend(fontsize=7, loc="best")
        else:
            ax_top.set_xlabel(xlabel)

        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in lab)[:55]
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phot_band_{safe}.png"), dpi=150)
        plt.close(fig)
        print(f"[overview] wrote phot_band_{safe}.png")


def _bundle_phase_groups_for_spec_panels(
    X: np.ndarray,
    point_class: np.ndarray,
    prm: np.ndarray,
    gn: dict,
    spec_bundle_id: Optional[np.ndarray],
    *,
    bundle_minutes: float,
    spec_phase_decimals: int,
    min_spec_rows_per_phase: int,
    phase_match_atol: float,
) -> tuple[list[tuple[int, np.ndarray]], str]:
    """``[(bundle_id, sorted_phases_in_bundle), ...]`` plus short provenance for plot titles."""
    spec_m = point_class == gu.SPEC
    sm_use = spec_m & prm

    def _filter_phases(phases: np.ndarray) -> np.ndarray:
        if min_spec_rows_per_phase <= 0 or phases.size == 0:
            return phases
        kept: list[float] = []
        for ph in phases.tolist():
            if int(spec_phase_decimals) >= 0:
                rq = float(np.round(float(ph), int(spec_phase_decimals)))
                n_sp = int(
                    np.count_nonzero(
                        sm_use & (np.round(X[:, 1], int(spec_phase_decimals)) == rq)
                    )
                )
            else:
                n_sp = int(
                    np.count_nonzero(sm_use & np.isclose(X[:, 1], float(ph), rtol=0.0, atol=phase_match_atol))
                )
            if n_sp >= min_spec_rows_per_phase:
                kept.append(float(ph))
            elif n_sp > 0:
                print(
                    f"[overview] skipping thin spectroscopic phase {float(ph):g} "
                    f"(N_spec_after_mask={n_sp} < --min-spec-rows-per-phase={min_spec_rows_per_phase})",
                    file=sys.stderr,
                )
        return np.sort(np.asarray(kept, dtype=float))

    if spec_bundle_id is not None and int(spec_bundle_id.shape[0]) == int(X.shape[0]):
        sbid = np.asarray(spec_bundle_id, dtype=np.int32).ravel()
        bids = sorted({int(b) for b in np.unique(sbid[sm_use]).tolist() if int(b) >= 0})
        out: list[tuple[int, np.ndarray]] = []
        for bid in bids:
            mb = sm_use & (sbid == int(bid))
            if not np.any(mb):
                continue
            if int(spec_phase_decimals) >= 0:
                phs = np.unique(np.round(X[mb, 1], int(spec_phase_decimals)))
            else:
                phs = _canonical_phases(X[mb, 1], atol=phase_match_atol)
            phs = np.sort(np.asarray(phs, dtype=float))
            phs = _filter_phases(phs)
            if phs.size:
                out.append((int(bid), phs))
        return out, "spec_bundle_id"

    if int(spec_phase_decimals) >= 0:
        uniq_phases = np.sort(np.unique(np.round(X[sm_use, 1], int(spec_phase_decimals))))
    else:
        uniq_phases = _canonical_phases(X[spec_m, 1], atol=phase_match_atol)
    uniq_phases = _filter_phases(uniq_phases)
    if uniq_phases.size == 0:
        return [], "phase_time"
    t_centers = phase_days_from_norm_x2(uniq_phases, gn)
    labels = sb.cluster_by_time(t_centers, max_delta_minutes=float(bundle_minutes))
    out2: list[tuple[int, np.ndarray]] = []
    for j in range(int(labels.max()) + 1):
        mask_ph = labels == j
        phb = np.sort(uniq_phases[mask_ph])
        if phb.size:
            out2.append((int(j), phb))
    return out2, "phase_time"


def plot_spectral_bundles_gp(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    X_fill: np.ndarray,
    mu_grid: np.ndarray,
    std_grid: Optional[np.ndarray],
    out_dir: str,
    *,
    bundle_minutes: float,
    overlap_scale: bool,
    gap_factor: float,
    min_gap_norm: float,
    plot_ratio_vs_gp: bool = False,
    plot_row_mask: Optional[np.ndarray] = None,
    min_spec_rows_per_phase: int = 0,
    train_obs_class_labeled: bool = True,
    phase_match_atol: float = 5e-6,
    spec_phase_decimals: int = 9,
    spec_bundle_id: Optional[np.ndarray] = None,
    canonical_phases: Optional[np.ndarray] = None,
    epoch_of_row: Optional[np.ndarray] = None,
    pair_scale_phase_atol: float = 5e-6,
    pair_scale_seam_weight: float = 1.0,
    pair_scale_overlap_grid: int = 256,
    pair_scale_seam_half_width_aa: float = 50.0,
    bundle_gp_phase_coalesce_norm: float = 2e-5,
    only_spec_bundle_ids: Optional[set[int]] = None,
) -> None:
    spec_m = point_class == gu.SPEC
    if not np.any(spec_m):
        print("[overview] no spectroscopy rows")
        return

    prm = (
        np.asarray(plot_row_mask, dtype=bool).ravel()
        if plot_row_mask is not None
        else np.ones(X.shape[0], dtype=bool)
    )
    if prm.shape[0] != X.shape[0]:
        raise ValueError("plot_row_mask length must match X")

    sm_use = spec_m & prm
    bundle_phase_groups, bundle_src = _bundle_phase_groups_for_spec_panels(
        X,
        point_class,
        prm,
        gn,
        spec_bundle_id,
        bundle_minutes=bundle_minutes,
        spec_phase_decimals=int(spec_phase_decimals),
        min_spec_rows_per_phase=int(min_spec_rows_per_phase),
        phase_match_atol=float(phase_match_atol),
    )
    if not bundle_phase_groups:
        print("[overview] no spectroscopy rows after thin-phase filter", file=sys.stderr)
        return
    if bundle_src == "spec_bundle_id":
        print("[overview] spectral bundle panels: grouping by spec_bundle_id (matches bundle_scale_pipeline)")
    else:
        print(
            "[overview] spectral bundle panels: grouping by Δt on phase timeline "
            "(no spec_bundle_id column — may disagree with bundle_scale_pipeline when enrich MJD differs)",
            file=sys.stderr,
        )

    wl_fill_sorted = np.sort(np.unique(X_fill[:, 0]))
    interp_mu, interp_std = _latent_gp_interpolators_from_fill(X_fill, mu_grid, std_grid)

    for bundle_id, phases_bundle in bundle_phase_groups:
        if only_spec_bundle_ids is not None and int(bundle_id) not in only_spec_bundle_ids:
            continue
        if phases_bundle.size == 0:
            continue

        scaled_rows, n_ov = _build_scaled_spec_overlay_rows(
            X,
            y,
            yerr,
            gn,
            phases_bundle,
            spec_m,
            overlap_scale=overlap_scale,
            gap_factor=gap_factor,
            min_abs_gap_norm=min_gap_norm,
            plot_row_mask=prm,
            phase_match_atol=phase_match_atol,
            spec_phase_decimals=int(spec_phase_decimals),
        )

        if plot_ratio_vs_gp:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7.8),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.65]},
            )
            ax_top, ax_bot = axes
        else:
            fig, ax_top = plt.subplots(figsize=(11, 5))
            ax_bot = None

        colors = plt.cm.viridis(np.linspace(0.05, 0.92, max(phases_bundle.size, 1)))
        by_phase = defaultdict(list)
        nd_ph = int(spec_phase_decimals)
        for rw in scaled_rows:
            phk = float(np.round(float(rw["sp_phase"]), nd_ph)) if nd_ph >= 0 else float(rw["sp_phase"])
            by_phase[phk].append(rw)

        ph_lbl_nd = max(6, min(14, nd_ph))
        dx_tol = float(bundle_gp_phase_coalesce_norm)
        ph_ptp = float(np.ptp(phases_bundle)) if phases_bundle.size > 1 else 0.0
        coalesce_gp_ph = bool(dx_tol > 0.0 and phases_bundle.size >= 2 and ph_ptp <= dx_tol)
        gp_ph_coalesced = float(np.median(phases_bundle)) if coalesce_gp_ph else None
        if coalesce_gp_ph:
            print(
                f"[overview] spec_bundle_{bundle_id}_gp.png: coalescing GP μ to median norm x₂="
                f"{gp_ph_coalesced:.{ph_lbl_nd}f} (Δx₂ span={ph_ptp:.3e} ≤ {dx_tol:g})",
                file=sys.stderr,
            )

        def _gp_slice_for_wl(ph_use: float, subs_list: list[dict], line_color: Any, lbl: str) -> None:
            wl_u = wl_fill_sorted.astype(float)
            if subs_list:
                lo_n, hi_n = _wl_norm_support_union(subs_list, pad=0.04)
                if np.isfinite(lo_n) and np.isfinite(hi_n):
                    wl_u = wl_u[(wl_u >= lo_n) & (wl_u <= hi_n)]
            else:
                wl_u = wl_u[:0]
            if not wl_u.size:
                return
            pts = np.column_stack([wl_u, np.full(wl_u.size, float(ph_use), dtype=float)])
            mu_lat = interp_mu(pts)
            w_line = denorm_ln_wavelength(wl_u, gn)
            o = np.argsort(w_line)
            w_sorted = w_line[o]
            mu_lin = scaled_ln_to_linear(mu_lat[o], gn)
            okm = np.isfinite(mu_lin)
            if np.any(okm):
                ax_top.plot(
                    w_sorted[okm],
                    mu_lin[okm],
                    "--",
                    color=line_color,
                    lw=1.85,
                    alpha=0.88,
                    label=lbl,
                )
            if interp_std is not None and std_grid is not None:
                st_lat = interp_std(pts)[o]
                sig_lin = linear_flux_yerr(mu_lat[o], st_lat, gn)
                okb = okm & np.isfinite(sig_lin)
                if np.any(okb):
                    ax_top.fill_between(
                        w_sorted[okb],
                        (mu_lin - sig_lin)[okb],
                        (mu_lin + sig_lin)[okb],
                        color=line_color,
                        alpha=0.09,
                    )

        if coalesce_gp_ph and gp_ph_coalesced is not None:
            all_subs_co: list[dict] = []
            for ph_b in phases_bundle:
                pk = float(np.round(float(ph_b), nd_ph)) if nd_ph >= 0 else float(ph_b)
                all_subs_co.extend(by_phase.get(pk, []))
            _gp_slice_for_wl(
                gp_ph_coalesced,
                all_subs_co,
                "0.15",
                f"GP μ (norm x₂≈{gp_ph_coalesced:.{ph_lbl_nd}f}, coalesced Δx₂={ph_ptp:.2e})",
            )
        else:
            for k_sp, ph_b in enumerate(phases_bundle):
                color = colors[k_sp % len(colors)]
                ph_key = float(np.round(float(ph_b), nd_ph)) if nd_ph >= 0 else float(ph_b)
                subs = by_phase.get(ph_key, [])
                _gp_slice_for_wl(
                    float(ph_b),
                    subs,
                    color,
                    f"GP μ (norm x₂={float(ph_b):.{ph_lbl_nd}f})",
                )

        ph_for_resid = float(gp_ph_coalesced) if coalesce_gp_ph and gp_ph_coalesced is not None else None
        for k_sp, ph_b in enumerate(phases_bundle):
            color = colors[k_sp % len(colors)]
            ph_key = float(np.round(float(ph_b), nd_ph)) if nd_ph >= 0 else float(ph_b)
            subs = by_phase.get(ph_key, [])
            for sg in subs:
                wl = denorm_ln_wavelength(sg["wl_norm"], gn)
                yp = sg["flux_lin"] * float(sg["scale"])
                lw = 2.2 if phases_bundle.size <= 4 else 1.2
                ax_top.plot(wl, yp, "-", color=color, lw=lw, alpha=0.85)
                if ax_bot is not None:
                    wl_n = np.asarray(sg["wl_norm"], dtype=float).ravel()
                    ph_use = float(ph_for_resid) if ph_for_resid is not None else float(sg["sp_phase"])
                    pts_s = np.column_stack([wl_n, np.full(wl_n.size, ph_use, dtype=float)])
                    mu_lat_s = interp_mu(pts_s)
                    mu_interp = scaled_ln_to_linear(mu_lat_s, gn)
                    okr = np.isfinite(mu_interp) & (mu_interp != 0) & np.isfinite(yp)
                    if np.any(okr):
                        rr = np.full_like(yp, np.nan)
                        rr[okr] = (yp[okr] / np.maximum(mu_interp[okr], 1e-30)) - 1.0
                        ax_bot.plot(wl, rr, "-", color=color, lw=min(lw, 1.1), alpha=0.85)

        ax_top.set_ylabel("flux (linear)")
        gp_title = (
            "GP μ on fill grid (median x₂ when Δx₂≤coalesce tol), λ clipped to data support"
            if coalesce_gp_ph
            else "GP μ on fill grid, clipped in λ to each exposure's data support"
        )
        ax_top.set_title(
            f"Spectral bundle {bundle_id} — {phases_bundle.size} spec phase(s), "
            f"Δt≤{bundle_minutes:g} min; {gp_title} "
            f"(n_ov={n_ov} intra-exposure λ-arm links only)"
        )
        ax_top.legend(fontsize=8, loc="upper right")
        ax_top.grid(alpha=0.25)
        if ax_bot is not None:
            ax_bot.axhline(0.0, color="k", lw=0.6, alpha=0.45)
            ax_bot.set_ylabel("spec/µ_gp − 1")
            ax_bot.set_xlabel("log10(wavelength)")
            ax_bot.grid(alpha=0.25)
        else:
            ax_top.set_xlabel("log10(wavelength)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"spec_bundle_{bundle_id}_gp.png"), dpi=150)
        plt.close(fig)
        print(f"[overview] wrote spec_bundle_{bundle_id}_gp.png")

        if canonical_phases is not None and epoch_of_row is not None:
            _write_spec_bundle_pair_scaling_figure(
                bundle_id=int(bundle_id),
                X=X,
                y=y,
                yerr=yerr,
                gn=gn,
                canonical_phases=np.asarray(canonical_phases, dtype=float).ravel(),
                epoch_of_row=np.asarray(epoch_of_row, dtype=np.int32).ravel(),
                phases_bundle=phases_bundle,
                out_dir=out_dir,
                phase_epoch_atol=float(pair_scale_phase_atol),
                seam_weight=float(pair_scale_seam_weight),
                overlap_grid_points=int(pair_scale_overlap_grid),
                seam_fit_half_width_aa=float(pair_scale_seam_half_width_aa),
                spec_phase_decimals=int(spec_phase_decimals),
            )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", "-b", default=DEFAULT_BUNDLE)
    p.add_argument("--meta", default=None)
    p.add_argument("--predictions", "-p", default=None, help="predictions.npz with mu_train, X_fill, mu")
    p.add_argument(
        "--runs-dir",
        default=None,
        help="parent of <tag>/ when using --tag without --predictions (default: <repo>/runs). "
        "Must match run_gp --output-dir if you did not use the default.",
    )
    p.add_argument(
        "--tag",
        "-t",
        default=None,
        help="use <runs-dir>/<tag>/predictions.npz if --predictions omitted",
    )
    p.add_argument("--run-config", default=None, help="runs/<tag>/config.json for grid_norm_info")
    p.add_argument("--output-dir", "-o", default=None)
    p.add_argument("--enrich", default=None)
    p.add_argument("--filter-config", default=None)
    p.add_argument("--filter-report", default=None, help="write JSON report for synth filters")
    p.add_argument("--synth-phase-tol", type=float, default=0.06, help="match phot epoch to spec phase (norm x2)")
    p.add_argument("--bundle-minutes", type=float, default=5.0)
    p.add_argument("--pseudo-band-digits", type=int, default=4)
    p.add_argument(
        "--phot-pseudo-grouping",
        choices=("rounded", "unique_x1"),
        default="rounded",
        help="when enrich lacks band labels: merge photometry into panels by rounded x₁ (legacy) "
        "or one panel per distinct training x₁ (matches per-row design in run_gp; display only).",
    )
    p.add_argument(
        "--phot-unique-x1-decimals",
        type=int,
        default=12,
        metavar="N",
        help="with --phot-pseudo-grouping unique_x1: round x₁ to N decimals before grouping keys "
        "(-1 = exact float keys; many panels if phot λ values differ slightly).",
    )
    p.add_argument(
        "--phot-max-unique-panels-warn",
        type=int,
        default=500,
        metavar="M",
        help="with unique_x1: warn on stderr if the number of photometry panels exceeds M (0 = never).",
    )
    p.add_argument(
        "--overlap-scale",
        action="store_true",
        help=(
            "Display-only: median-ratio overlap alignment of λ segments **within each "
            "spectroscopic exposure** (instrument arms only). Default off — use for raw "
            "visual alignment; corrected bundles usually need this off."
        ),
    )
    p.add_argument(
        "--no-overlap-scale",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--phot-spec-threshold",
        type=int,
        default=50,
        help="when bundle lacks train_obs_class: heuristic spec vs phot (unique λ per phase; "
        "same meaning as run_gp --phot-spec-threshold, default 50)",
    )
    p.add_argument(
        "--min-spec-rows-per-phase",
        type=int,
        default=0,
        metavar="N",
        help="omit spectral phases with fewer than N spec rows after telluric/yerr masking "
        "(0 = off; useful to suppress tiny fragmented exposures)",
    )
    p.add_argument(
        "--expect-pipeline-bundle",
        action="store_true",
        help="require spec_bundle_id in bundle (written by bundle_scale_pipeline); exit if missing",
    )
    p.add_argument("--posterior-kind", choices=("train", "grid_pp", "grid_raw"), default="train")
    p.add_argument(
        "--plot-residuals-vs-gp",
        action="store_true",
        help="second panel per phot band showing (flux−µ)/|µ| vs time (µ from --posterior-kind)",
    )
    p.add_argument(
        "--phot-lc-time-step-days",
        type=float,
        default=0.05,
        metavar="DT",
        help="with predictions, resample GP μ on the photometry time axis at this spacing (same units as "
        "the x-axis: MJD if enrich has mjd, else phase in days). Interpolates the latent GP in (x₁,x₂). "
        "Use 0 to draw straight segments only between training rows (legacy).",
    )
    p.add_argument(
        "--phot-lc-x1-mode",
        choices=("track", "median"),
        default="track",
        help="how dense phot GP sets normalized log-λ x₁ along time: 'track' interpolates x₁(t) "
        "from the band's photometry (recommended); 'median' uses a single band-median x₁ (legacy).",
    )
    p.add_argument(
        "--spec-ratio-vs-gp",
        action="store_true",
        help="second panel per spectral bundle with (scaled spec / µ_gp at slice − 1)",
    )
    p.add_argument(
        "--spec-phase-decimals",
        type=int,
        default=9,
        help="treat spectroscopic rows as distinct exposures when round(x₂, N) differs "
        "(match bundle_scale_pipeline; -1 = legacy merge by atol)",
    )
    p.add_argument(
        "--bundle-gp-phase-coalesce-norm",
        type=float,
        default=2e-5,
        metavar="DX2",
        help="if max(x₂)−min(x₂) in a spectral bundle (normalized log10 phase) ≤ DX2, draw a single "
        "GP μ curve at median x₂ instead of interpolating at each rounded exposure (avoids large "
        "Δμ_latent across ~1e⁻⁶ phase steps). Set 0 to disable.",
    )
    p.add_argument(
        "--only-spec-bundle-ids",
        default=None,
        metavar="IDS",
        help="comma-separated spec_bundle_id integers: only write spec_bundle_<id>_*.png panels "
        "(phot band PNGs are still written)",
    )
    args = p.parse_args(argv)

    overlap_disp = bool(args.overlap_scale)
    if args.no_overlap_scale:
        overlap_disp = False

    only_spec: Optional[Set[int]] = None
    if args.only_spec_bundle_ids and str(args.only_spec_bundle_ids).strip():
        only_spec = set(int(x.strip()) for x in str(args.only_spec_bundle_ids).split(",") if x.strip())

    runs_root = os.path.abspath(os.path.expanduser(args.runs_dir or DEFAULT_RUNS))

    pred_path = args.predictions
    if pred_path:
        pred_path = os.path.abspath(os.path.expanduser(str(pred_path)))
    tag = str(args.tag).strip() if args.tag is not None else ""
    tag = tag or None
    if tag and not pred_path:
        pred_path = os.path.join(runs_root, tag, "predictions.npz")
    run_cfg = args.run_config
    if run_cfg:
        run_cfg = os.path.abspath(os.path.expanduser(str(run_cfg)))
    elif tag:
        cand = os.path.join(runs_root, tag, "config.json")
        if os.path.isfile(cand):
            run_cfg = cand

    gn = _load_gn(args.bundle, args.meta, run_cfg)
    enrich = _load_enrich(args.enrich)

    bundle = np.load(args.bundle, allow_pickle=False)
    try:
        if args.expect_pipeline_bundle and "spec_bundle_id" not in bundle.files:
            print(
                "[overview] ERROR: --expect-pipeline-bundle set but bundle has no "
                "spec_bundle_id (run bundle_scale_pipeline on this npz first).",
                file=sys.stderr,
            )
            return 3

        X = np.asarray(bundle["X"], dtype=float)
        y = np.asarray(bundle["y"], dtype=float)
        yerr = np.asarray(bundle["yerr"], dtype=float)
        spec_bundle_id_arr: Optional[np.ndarray] = (
            np.asarray(bundle["spec_bundle_id"], dtype=np.int32)
            if "spec_bundle_id" in bundle.files
            else None
        )

        obs = np.asarray(bundle["train_obs_class"]) if "train_obs_class" in bundle.files else None
        if obs is None:
            print(
                "[overview] WARNING: bundle has no train_obs_class; spectroscopy vs photometry "
                "classification uses heuristic only (same --phot-spec-threshold as run_gp). "
                "For reliable spec panels run bundle_preprocess or add train_obs_class before plotting.",
                file=sys.stderr,
            )
        labeled = obs is not None
        point_class = gu.effective_point_class(
            X,
            train_obs_class=obs,
            threshold=args.phot_spec_threshold,
        )

        plot_row_good = np.ones(X.shape[0], dtype=bool)
        if "telluric_bad_mask" in bundle.files:
            tm = np.asarray(bundle["telluric_bad_mask"])
            tm = tm.reshape(-1).astype(bool)
            if tm.shape[0] != X.shape[0]:
                print(
                    f"[overview] WARNING: telluric_bad_mask length {tm.shape[0]} != N={X.shape[0]}; ignoring",
                    file=sys.stderr,
                )
            else:
                plot_row_good &= ~tm
        ydl = float(bpre.YERR_DISABLED)
        plot_row_good &= np.isfinite(yerr.ravel())
        plot_row_good &= yerr.ravel() < ydl
    finally:
        bundle.close()

    mu_train = X_fill = mu_grid = std_grid = None
    pr = None
    if pred_path and os.path.isfile(pred_path):
        pr = np.load(pred_path, allow_pickle=False)
        if "mu_train" in pr.files:
            mt0 = np.asarray(pr["mu_train"], dtype=float)
            mu_al = scatter_train_vector_to_bundle(pr, mt0, int(X.shape[0]))
            if mu_al is None and mt0.size != int(X.shape[0]):
                print(
                    "[overview] ERROR: predictions mu_train length does not match bundle rows and "
                    "train_row_index_orig is missing or invalid — omit GP-at-train overlay",
                    file=sys.stderr,
                )
                mu_train = None
            else:
                mu_train = mu_al if mu_al is not None else mt0
        if "X_fill" in pr.files and "mu" in pr.files:
            X_fill = np.asarray(pr["X_fill"], dtype=float)
            mu_grid = np.asarray(pr["mu"], dtype=float)
            std_grid = np.asarray(pr["std"], dtype=float) if "std" in pr.files else None
        print(f"[overview] loaded predictions {pred_path!r}")
    else:
        print("[overview] WARNING: no predictions.npz — GP curves omitted", file=sys.stderr)

    legend_suffix_map = {"train": "train", "grid_pp": "grid_postprocessed", "grid_raw": "grid_raw"}
    comp_mu_lin = comp_sig_lin = None
    if pr is not None and args.plot_residuals_vs_gp:
        comp_mu_lin, comp_sig_lin = build_comparison_mu_lin(pr, X, gn, posterior_kind=args.posterior_kind)

    out_dir = args.output_dir
    if out_dir:
        out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
    elif tag:
        out_dir = os.path.join(runs_root, tag, "figs", "overview")
    else:
        out_dir = os.path.join(HERE, "overview_plots")
    os.makedirs(out_dir, exist_ok=True)

    fr_path = args.filter_report or os.path.join(out_dir, "filter_synth_report.json")

    plot_photometry_bands(
        X,
        y,
        yerr,
        point_class,
        gn,
        enrich,
        mu_train,
        out_dir,
        filter_cfg_path=args.filter_config,
        synth_phase_tol=args.synth_phase_tol,
        pseudo_band_digits=args.pseudo_band_digits,
        filter_report_path=fr_path if args.filter_config else None,
        comparison_mu_lin=comp_mu_lin,
        comparison_sigma_lin=comp_sig_lin,
        plot_residual_vs_gp=args.plot_residuals_vs_gp,
        posterior_legend_suffix=legend_suffix_map[str(args.posterior_kind)],
        preds_npz=pr,
        posterior_kind=args.posterior_kind,
        phot_lc_time_step=float(args.phot_lc_time_step_days),
        phot_lc_x1_mode=str(args.phot_lc_x1_mode),
        phot_pseudo_grouping=str(args.phot_pseudo_grouping),
        phot_unique_x1_decimals=int(args.phot_unique_x1_decimals),
        phot_max_unique_panels_warn=int(args.phot_max_unique_panels_warn),
    )

    if X_fill is not None and mu_grid is not None:
        spec_m_epochs = point_class == gu.SPEC
        canon_phases, epoch_of_row = bsp.unique_spec_epochs(X, spec_m_epochs)
        plot_spectral_bundles_gp(
            X,
            y,
            yerr,
            point_class,
            gn,
            X_fill,
            mu_grid,
            std_grid,
            out_dir,
            bundle_minutes=args.bundle_minutes,
            overlap_scale=overlap_disp,
            gap_factor=35.0,
            min_gap_norm=3e-3,
            plot_ratio_vs_gp=bool(args.spec_ratio_vs_gp),
            plot_row_mask=plot_row_good,
            min_spec_rows_per_phase=args.min_spec_rows_per_phase,
            train_obs_class_labeled=labeled,
            spec_phase_decimals=int(args.spec_phase_decimals),
            spec_bundle_id=spec_bundle_id_arr,
            canonical_phases=canon_phases,
            epoch_of_row=epoch_of_row,
            bundle_gp_phase_coalesce_norm=float(args.bundle_gp_phase_coalesce_norm),
            only_spec_bundle_ids=only_spec,
        )

    if pr is not None:
        try:
            pr.close()
        except Exception:
            pass

    print(f"[overview] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
