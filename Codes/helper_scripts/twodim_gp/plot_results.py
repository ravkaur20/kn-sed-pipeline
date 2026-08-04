"""Plot diagnostics for a single GP run.

Usage:
    python plot_results.py --tag matern52_linear_opt
    python plot_results.py --tag matern32_nearest_baseline_jitter

Reads:
    runs/<tag>/predictions.npz  (written by run_gp.py)
    runs/<tag>/config.json
    gp_minimal_bundle.npz       (training X/y/yerr)
    <bundle_stem>_meta.json     optional; default path next to bundle (e.g. gp_minimal_bundle_meta.json).
                                Supplies ``grid_norm_info`` for physical axes and ln-flux → linear flux.

Writes:
    runs/<tag>/figs/*.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import cycle
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

try:
    from . import bundle_meta as bmeta
    from . import bundle_preprocess as bpre
    from . import gp_utils as gu
except ImportError:
    import bundle_meta as bmeta
    import bundle_preprocess as bpre
    import gp_utils as gu


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "runs")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def scaled_ln_to_linear(mu: np.ndarray, gn: dict) -> np.ndarray:
    """Training / GP outputs are collaborator-normalized ln-flux; map to linear flux when meta exists."""
    m = np.asarray(mu, dtype=float)
    if gn.get("_normalized_only"):
        return m
    off = float(gn["offset"])
    sf = float(gn["scale_factor"])
    return np.exp(m * sf + off)


def phase_days_from_norm_x2(x2: np.ndarray, gn: dict) -> np.ndarray:
    """Invert z-score on column 1 (normalized log10 phase) to phase in days."""
    u = np.asarray(x2, dtype=float)
    if gn.get("_normalized_only"):
        return u
    x2m = float(gn["x2_mean"])
    x2s = float(gn["x2_std"])
    log10_phase = x2m + x2s * u
    return np.power(10.0, log10_phase)


def norm_x2_from_phase_days(phase_days: np.ndarray, gn: dict) -> np.ndarray:
    """Map phase in days to normalized ``X[:, 1]`` (inverse of ``phase_days_from_norm_x2``).

    For ``_normalized_only`` grids, ``phase_days`` is already the stored x₂ coordinate.
    """
    t = np.asarray(phase_days, dtype=float)
    if gn.get("_normalized_only"):
        return t
    tiny = float(np.finfo(float).tiny)
    lp = np.log10(np.maximum(t, tiny))
    return (lp - float(gn["x2_mean"])) / float(gn["x2_std"])


def denorm_ln_wavelength(x1_norm: np.ndarray, gn: dict) -> np.ndarray:
    """Physical log10(wavelength) from normalized column 0."""
    u = np.asarray(x1_norm, dtype=float)
    if gn.get("_normalized_only"):
        return u
    return float(gn["x1_mean"]) + float(gn["x1_std"]) * u


def wl_linear_aa_to_plot_x1(wl_aa: np.ndarray, gn: dict) -> np.ndarray:
    """Horizontal axis matching spectroscopy overlays: ``composite_epoch_linear`` returns ``wav`` Å with
    ``wav = 10 ** wl_log_phys`` and ``wl_log_phys == denorm_ln_wavelength(X[:,0], gn)`` per row.

    In ``_normalized_only`` mode this is already the original spectral coordinate and is
    returned unchanged.

    Must **not** clip ``wav`` to ≥1 Å: with ``grid_norm_info`` / normalized column 0, negative ``x₁``
    gives ``wav < 1`` and clipping would collapse every point to ``log₁₀(1) = 0`` (vertical spike).
    """
    if gn.get("_normalized_only"):
        return np.asarray(wl_aa, dtype=float)
    wav = np.asarray(wl_aa, dtype=float)
    tiny = np.finfo(float).tiny
    wl_log_phys = np.log10(np.maximum(wav, tiny))
    u_norm = (wl_log_phys - float(gn["x1_mean"])) / float(gn["x1_std"])
    return np.asarray(denorm_ln_wavelength(u_norm, gn), dtype=float)


def denorm_ln_phase_days(x2_norm: np.ndarray, gn: dict) -> np.ndarray:
    """Physical log10(phase days) from normalized column 1."""
    u = np.asarray(x2_norm, dtype=float)
    if gn.get("_normalized_only"):
        return u
    return float(gn["x2_mean"]) + float(gn["x2_std"]) * u


def linear_flux_yerr(mu_or_y: np.ndarray, yerr: np.ndarray, gn: dict) -> np.ndarray:
    """Symmetric std on linear flux (first-order delta from latent ln estimate). Ignored when plotting in normalized units."""
    e = np.asarray(yerr, dtype=float)
    if gn.get("_normalized_only"):
        return e
    f = scaled_ln_to_linear(mu_or_y, gn)
    sf = float(gn["scale_factor"])
    return np.maximum(np.abs(f) * sf * e, np.finfo(float).tiny)


def scatter_train_vector_to_bundle(
    preds: Any,
    vec: np.ndarray,
    bundle_n: int,
    *,
    key_orig: str = "train_row_index_orig",
) -> Optional[np.ndarray]:
    """Place per-fit-row ``vec`` (length K) into length ``bundle_n`` at ``train_row_index_orig``; else NaN."""
    vec = np.asarray(vec, dtype=float).ravel()
    if vec.size == bundle_n:
        return vec
    if key_orig not in preds.files:
        print(
            f"[plot_results] WARN: vector length {vec.size} != bundle N={bundle_n} and no "
            f"{key_orig!r} in predictions — cannot align to full bundle",
            file=sys.stderr,
        )
        return None
    oi = np.asarray(preds[key_orig], dtype=np.int64).ravel()
    if oi.size != vec.size:
        print(
            f"[plot_results] WARN: {key_orig} length {oi.size} != vector {vec.size}",
            file=sys.stderr,
        )
        return None
    if oi.size == 0 or int(np.min(oi)) < 0 or int(np.max(oi)) >= bundle_n:
        print(
            f"[plot_results] WARN: {key_orig} out of range for bundle_n={bundle_n}",
            file=sys.stderr,
        )
        return None
    out = np.full(bundle_n, np.nan, dtype=float)
    out[oi] = vec
    return out


def _spec_segments_sorted(
    wl_norm: np.ndarray,
    y_seg: np.ndarray,
    yer_seg: np.ndarray,
    *,
    gn: dict,
    gap_factor: float,
    min_abs_gap_norm: float,
) -> list[dict]:
    """Split one phase's spectroscopy rows into contiguous λ chunks (instrument arms etc.)."""
    wl_norm = np.asarray(wl_norm, dtype=float).ravel()
    y_seg = np.asarray(y_seg, dtype=float).ravel()
    yer_seg = np.asarray(yer_seg, dtype=float).ravel()
    ok = np.isfinite(wl_norm) & np.isfinite(y_seg)
    wl_norm = wl_norm[ok]
    y_seg = y_seg[ok]
    yer_seg = yer_seg[ok]
    if wl_norm.size == 0:
        return []
    order = np.argsort(wl_norm)
    wl = wl_norm[order]
    ys = y_seg[order]
    ye = yer_seg[order]
    fl = scaled_ln_to_linear(ys, gn)
    ef = linear_flux_yerr(ys, ye, gn)
    dx = np.diff(wl)
    med = float(np.median(dx[dx > 1e-12])) if np.any(dx > 1e-12) else 0.002
    thr = max(gap_factor * med, float(min_abs_gap_norm))
    cuts = [0]
    for ii in range(dx.size):
        if dx[ii] > thr:
            cuts.append(ii + 1)
    cuts.append(wl.size)
    out = []
    for a in range(len(cuts) - 1):
        i0, i1 = cuts[a], cuts[a + 1]
        if i1 <= i0:
            continue
        seg_wl = wl[i0:i1]
        out.append(
            {
                "wl_norm": seg_wl.copy(),
                "flux_lin": fl[i0:i1].copy(),
                "eflux_lin": ef[i0:i1].copy(),
                "y_norm": ys[i0:i1].copy(),
                "yer_norm": ye[i0:i1].copy(),
            }
        )
    return out


def _overlap_median_fi_over_fj(
    wl_i: np.ndarray,
    fi: np.ndarray,
    wl_j: np.ndarray,
    fj: np.ndarray,
    *,
    n_grid: int = 320,
    min_overlap_grid: int = 10,
) -> Optional[float]:
    """Return median λ-grid sample of ``f_i / f_j`` in overlap so ``sj = si * ratio`` aligns ``sj*fj≈si*fi``."""
    lo = max(np.min(wl_i), np.min(wl_j))
    hi = min(np.max(wl_i), np.max(wl_j))
    if hi - lo <= np.finfo(float).eps * (1 + abs(lo) + abs(hi)):
        return None
    grid = np.linspace(lo, hi, n_grid)
    ii = np.interp(grid, wl_i, fi, left=np.nan, right=np.nan)
    jj = np.interp(grid, wl_j, fj, left=np.nan, right=np.nan)
    mask = np.isfinite(ii) & np.isfinite(jj) & (ii > 0) & (jj > 0)
    if mask.sum() < min_overlap_grid:
        return None
    r = ii[mask] / jj[mask]
    med = float(np.median(np.clip(r, 1e-20, 1e20)))
    return med if np.isfinite(med) and med > 0 else None


def _solve_relative_linear_scales(
    segments: list[dict],
) -> tuple[np.ndarray, int]:
    """Starting from longest segment, repeatedly assign scales via overlap ratios until saturation."""
    n = len(segments)
    if n <= 1:
        return np.ones(max(n, 1), dtype=float), 0
    pts = np.array([sg["flux_lin"].size for sg in segments], dtype=float)
    ref = int(np.argmax(pts))
    scales = np.ones(n, dtype=float)
    assigned = np.zeros(n, dtype=bool)
    assigned[ref] = True

    wl_list = [sg["wl_norm"] for sg in segments]
    f_list = [sg["flux_lin"] for sg in segments]

    n_edges_used = 0
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not assigned[i]:
                continue
            wi, fi = wl_list[i], f_list[i]
            for j in range(n):
                if assigned[j]:
                    continue
                wj, fj_seg = wl_list[j], f_list[j]
                rij = _overlap_median_fi_over_fj(wi, fi, wj, fj_seg)
                if rij is None:
                    continue
                scales[j] = float(np.clip(scales[i] * rij, 1e-3, 1e3))
                assigned[j] = True
                changed = True
                n_edges_used += 1

    scales[~assigned] = 1.0
    return scales, n_edges_used


def _build_scaled_spec_overlay_rows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    gn: dict,
    near_spec_phases: np.ndarray,
    spec_mask: np.ndarray,
    *,
    overlap_scale: bool,
    gap_factor: float,
    min_abs_gap_norm: float,
    plot_row_mask: Optional[np.ndarray] = None,
    phase_match_atol: float = 5e-6,
    spec_phase_decimals: int = 9,
) -> tuple[list[dict], int]:
    """Collect λ segments per near-sim spec phase and attach overlap ``scale`` **per exposure only**.

    Median-ratio overlap scaling is solved **within each** ``sp_phase`` (instrument arms),
    never across different spectroscopic epochs—linking phases together previously
    distorted relative calibration and could draw spurious chords between unrelated data.

    When ``spec_phase_decimals`` ≥ 0, spectroscopic rows are grouped by ``round(x₂, decimals)``
    (same convention as ``bundle_scale_pipeline``), so nearly-coincident exposures are not
    merged into one polyline sorted only by wavelength.
    """
    segments: list[dict] = []
    edges_total = 0
    prm = (
        np.asarray(plot_row_mask, dtype=bool).ravel()
        if plot_row_mask is not None
        else None
    )
    if prm is not None and prm.shape[0] != X_train.shape[0]:
        raise ValueError("plot_row_mask length must match X_train")

    near_ph = np.sort(np.asarray(near_spec_phases, dtype=float).ravel())
    if int(spec_phase_decimals) >= 0:
        phase_list = [float(x) for x in np.unique(np.round(near_ph, int(spec_phase_decimals)))]
    else:
        phase_list = []
        for ph in near_ph:
            if not phase_list or all(abs(float(ph) - q) > float(phase_match_atol) for q in phase_list):
                phase_list.append(float(ph))

    for sp_phase in phase_list:
        if int(spec_phase_decimals) >= 0:
            rq = float(np.round(float(sp_phase), int(spec_phase_decimals)))
            m = (
                np.isfinite(X_train[:, 0])
                & spec_mask
                & (np.round(X_train[:, 1], int(spec_phase_decimals)) == rq)
            )
        else:
            m = np.isfinite(X_train[:, 0]) & spec_mask & np.isclose(
                X_train[:, 1], sp_phase, rtol=0.0, atol=phase_match_atol
            )
        if prm is not None:
            m &= prm
        if not np.any(m):
            continue
        wl = X_train[m, 0]
        ys = y_train[m]
        ye = yerr_train[m]
        chunks = _spec_segments_sorted(
            wl,
            ys,
            ye,
            gn=gn,
            gap_factor=gap_factor,
            min_abs_gap_norm=min_abs_gap_norm,
        )
        phase_rows: list[dict] = [{"sp_phase": float(sp_phase), **ch} for ch in chunks]
        if not phase_rows:
            continue
        edges_ph = 0
        if overlap_scale and len(phase_rows) > 1:
            scales_ph, edges_ph = _solve_relative_linear_scales(phase_rows)
        else:
            scales_ph = np.ones(len(phase_rows), dtype=float)
            edges_ph = 0
        edges_total += int(edges_ph)
        for sg, sc in zip(phase_rows, scales_ph):
            sg["scale"] = float(sc)
            segments.append(sg)
    return segments, edges_total


def _make_wavelength_slice_figure(
    x1_fill: np.ndarray,
    x2_fill: np.ndarray,
    mu_fill: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    *,
    use_log10_phase_axis: bool,
    log_y: bool,
    save_path: str,
    suptitle: Optional[str] = None,
    overlay_training: bool = True,
) -> None:
    x1m, x1s = float(gn["x1_mean"]), float(gn["x1_std"])
    x2m, x2s = float(gn["x2_mean"]), float(gn["x2_std"])

    fit_wls = np.unique(x1_fill)[::10]
    len_wls = len(fit_wls)
    if len_wls < 4:
        print(f"[plot_results]   only {len_wls} slices; skipping {save_path}")
        return

    color_iter = cycle(plt.cm.gnuplot(np.linspace(0.05, 0.95, len_wls)))

    fig = plt.figure(figsize=(11, 7))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=9, y=1.02)

    quarter = len_wls // 4
    panel_slices = [
        (1, slice(0, quarter)),
        (2, slice(quarter, 2 * quarter)),
        (3, slice(2 * quarter, 3 * quarter)),
        (4, slice(3 * quarter, len_wls)),
    ]

    for panel_idx, sl in panel_slices:
        ax = plt.subplot(2, 2, panel_idx)
        wls_in_panel = fit_wls[sl]
        if wls_in_panel.size == 0:
            continue
        ax.set_title(
            "log10(wl): %.3f-%.3f"
            % (
                min(x1m + x1s * wls_in_panel),
                max(x1m + x1s * wls_in_panel),
            ),
            fontsize=10,
        )
        for i in wls_in_panel:
            mask = x1_fill == i
            if not mask.any():
                continue
            xv = (x2m + x2s * x2_fill[mask]) if use_log10_phase_axis else phase_days_from_norm_x2(x2_fill[mask], gn)
            yv = scaled_ln_to_linear(mu_fill[mask], gn)
            order = np.argsort(xv)
            ax.plot(xv[order], yv[order], color=next(color_iter), lw=0.8, alpha=0.8)
        if overlay_training:
            wls_min = wls_in_panel.min()
            wls_max = wls_in_panel.max()
            wls_pad = 0.5 * (np.unique(x1_fill).max() - np.unique(x1_fill).min()) / max(len(np.unique(x1_fill)) - 1, 1)
            wls_mask = (X_train[:, 0] >= wls_min - wls_pad) & (X_train[:, 0] <= wls_max + wls_pad)
            for cls_name, marker, alpha, label in (
                (gu.PHOT, "o", 0.9, "phot"),
                (gu.SPEC, ".", 0.5, "spec"),
            ):
                cm = wls_mask & (point_class == cls_name)
                if not cm.any():
                    continue
                xv = (x2m + x2s * X_train[cm, 1]) if use_log10_phase_axis else phase_days_from_norm_x2(X_train[cm, 1], gn)
                yv = scaled_ln_to_linear(y_train[cm], gn)
                ax.errorbar(
                    xv,
                    yv,
                    yerr=linear_flux_yerr(y_train[cm], yerr_train[cm], gn),
                    fmt=marker,
                    ms=3 if cls_name == gu.PHOT else 1.5,
                    lw=0.5,
                    elinewidth=0.4,
                    color="k",
                    alpha=alpha,
                    label=label,
                )
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("log10(phase days)" if use_log10_phase_axis else "Phase (days)")
        ax.set_ylabel("flux (linear)" if not gn.get("_normalized_only") else "y (normalized latent)")
        if log_y:
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.92] if suptitle else None)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_heatmap(
    X_fill: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    save_path: str,
    cmap: str = "viridis",
    overlay_training_phases: Optional[np.ndarray] = None,
    gn: Optional[dict] = None,
    *,
    linearize_values: bool = False,
) -> None:
    gn = gn or bmeta.identity_grid_norm()
    wls = np.unique(X_fill[:, 0])
    phases = np.unique(X_fill[:, 1])
    grid = np.full((wls.size, phases.size), np.nan, dtype=float)
    wls_idx = {v: i for i, v in enumerate(wls)}
    phase_idx = {v: i for i, v in enumerate(phases)}
    for k in range(X_fill.shape[0]):
        i = wls_idx[X_fill[k, 0]]
        j = phase_idx[X_fill[k, 1]]
        grid[i, j] = values[k]

    x_plot = denorm_ln_phase_days(phases, gn)
    y_plot = denorm_ln_wavelength(wls, gn)
    z_plot = scaled_ln_to_linear(grid, gn) if linearize_values else grid

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x_plot, y_plot, z_plot, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=cbar_label)
    if overlay_training_phases is not None and overlay_training_phases.size:
        for p in np.unique(overlay_training_phases):
            ax.axvline(denorm_ln_phase_days(np.array([float(p)]), gn)[0], color="white", lw=0.2, alpha=0.5)
    if gn.get("_normalized_only"):
        ax.set_xlabel("normalized log10(phase days)")
        ax.set_ylabel("normalized log10(wavelength)")
    else:
        ax.set_xlabel("log10(phase days)")
        ax.set_ylabel("log10(wavelength)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_training_coverage(
    X: np.ndarray,
    y: np.ndarray,
    point_class: np.ndarray,
    X_fill: np.ndarray,
    save_path: str,
    gn: Optional[dict] = None,
) -> None:
    gn = gn or bmeta.identity_grid_norm()
    fig, ax = plt.subplots(figsize=(10, 5))
    spec_mask = point_class == gu.SPEC
    phot_mask = point_class == gu.PHOT
    x_s = denorm_ln_phase_days(X[spec_mask, 1], gn)
    x_p = denorm_ln_phase_days(X[phot_mask, 1], gn)
    w_s = denorm_ln_wavelength(X[spec_mask, 0], gn)
    w_p = denorm_ln_wavelength(X[phot_mask, 0], gn)
    c_spec = scaled_ln_to_linear(y[spec_mask], gn)
    c_phot = scaled_ln_to_linear(y[phot_mask], gn)
    sc = ax.scatter(x_s, w_s, c=c_spec, s=3, cmap="viridis", label=f"spec (n={spec_mask.sum()})")
    ax.scatter(x_p, w_p, c=c_phot, s=14, cmap="viridis",
               edgecolors="red", linewidths=0.6, label=f"phot (n={phot_mask.sum()})")
    fig.colorbar(sc, ax=ax, label=("y (normalized)" if gn.get("_normalized_only") else "flux (linear)"))
    ax.set_xlim(denorm_ln_phase_days(X_fill[:, 1], gn).min(), denorm_ln_phase_days(X_fill[:, 1], gn).max())
    ax.set_ylim(denorm_ln_wavelength(X_fill[:, 0], gn).min(), denorm_ln_wavelength(X_fill[:, 0], gn).max())
    if gn.get("_normalized_only"):
        ax.set_xlabel("normalized log10(phase days)")
        ax.set_ylabel("normalized log10(wavelength)")
    else:
        ax.set_xlabel("log10(phase days)")
        ax.set_ylabel("log10(wavelength)")
    ax.set_title("Training coverage (red rim = phot, dot = spec)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_residual_histograms(
    y: np.ndarray,
    mu_train: np.ndarray,
    sigma_eff: np.ndarray,
    point_class: np.ndarray,
    save_path: str,
) -> None:
    m_fit = np.isfinite(mu_train) & np.isfinite(y) & np.isfinite(sigma_eff)
    if not np.any(m_fit):
        print("[plot_results]   skip training_residuals.png (no finite mu_train rows)", file=sys.stderr)
        return
    y = y[m_fit]
    mu_train = mu_train[m_fit]
    sigma_eff = sigma_eff[m_fit]
    point_class = point_class[m_fit]
    resid = mu_train - y
    norm = resid / np.maximum(sigma_eff, 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(resid, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("mu_train - y")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"raw residuals (mean={resid.mean():.3g}, std={resid.std():.3g})")

    bins = np.linspace(-6, 6, 80)
    for cls, color, label in (
        (gu.PHOT, "indianred", "phot"),
        (gu.SPEC, "steelblue", "spec"),
    ):
        m = point_class == cls
        if not m.any():
            continue
        axes[1].hist(
            np.clip(norm[m], -6, 6),
            bins=bins,
            histtype="step",
            lw=1.5,
            color=color,
            label=f"{label} (n={m.sum()}, std={norm[m].std():.2f})",
        )
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("(mu_train - y) / sigma_eff")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"normalized residuals (target std=1)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_spectrum_figure(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    requested_phases: np.ndarray,
    save_path: str,
    gn: Optional[dict] = None,
    near_sim_tol: float = 0.05,
    *,
    overlap_scale_display: bool = True,
    segment_gap_factor: float = 35.0,
    min_segment_gap_norm: float = 3e-3,
    spec_phase_decimals: int = 9,
) -> None:
    """Spectrum (mu vs wavelength) at the closest available training spectra,
    overlaying *all* near-simultaneous spectra.

    For each requested normalized-log10(phase) value:

    1. Snap to the *nearest spec training phase* (an actual spectrum that
       exists in the data, even if it isn't exactly at the requested time).
    2. Find every other spec training phase within ``near_sim_tol`` of that
       chosen spec phase. These count as "near-simultaneous" spectra and
       are all overlaid (with different colors), since some spectra are
       very close in time but cover non-overlapping wavelength ranges.
    3. Snap the chosen spec phase to the nearest X_fill grid phase and plot
       the GP prediction (mu +/- 1 sigma) at that grid phase.
    4. Optionally overlay phot training points within ``near_sim_tol`` of
       the chosen spec phase for cross-context.

    If ``overlap_scale_display`` is True, overlapping λ arms **within each
    spectroscopic phase** receive **multiplicative** display constants (median
    flux ratio on overlaps) so arm joins are softened **for plotting only** —
    training data fed to the GP is unchanged. Scales never link across exposures.
    """
    phases_pred = np.unique(X_fill[:, 1])
    if phases_pred.size < 2:
        return

    gn = gn or bmeta.identity_grid_norm()

    spec_mask = point_class == gu.SPEC
    plot_row_good = np.isfinite(yerr_train) & (np.asarray(yerr_train, dtype=float) < float(bpre.YERR_DISABLED))
    spec_phases_train = np.unique(X_train[spec_mask, 1]) if spec_mask.any() else np.array([])
    if spec_phases_train.size == 0:
        print("[plot_results]   no spec training data; skipping spectrum figure")
        return

    n = len(requested_phases)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, requested in zip(axes, requested_phases):
        spec_idx = int(np.argmin(np.abs(spec_phases_train - requested)))
        spec_phase = float(spec_phases_train[spec_idx])

        # Every spec phase within near_sim_tol of the chosen spec phase.
        near_spec_phases = np.sort(
            spec_phases_train[np.abs(spec_phases_train - spec_phase) <= near_sim_tol]
        )

        scaled_rows, n_ov = _build_scaled_spec_overlay_rows(
            X_train,
            y_train,
            yerr_train,
            gn,
            near_spec_phases,
            spec_mask,
            overlap_scale=overlap_scale_display,
            gap_factor=segment_gap_factor,
            min_abs_gap_norm=min_segment_gap_norm,
            plot_row_mask=plot_row_good,
            spec_phase_decimals=int(spec_phase_decimals),
        )
        by_phase: dict[float, list[dict]] = defaultdict(list)
        nd_sp = int(spec_phase_decimals)
        for rw in scaled_rows:
            pk = float(np.round(float(rw["sp_phase"]), nd_sp)) if nd_sp >= 0 else float(rw["sp_phase"])
            by_phase[pk].append(rw)

        phase_keys = sorted(by_phase.keys())
        spec_colors = plt.cm.viridis(
            np.linspace(0.05, 0.85, max(len(phase_keys), 1))
        )

        def _wl_norm_support_union_pr(subs: list[dict], *, pad: float = 0.04) -> tuple[float, float]:
            if not subs:
                return float("nan"), float("nan")
            lo = min(float(np.min(np.asarray(sg["wl_norm"], dtype=float))) for sg in subs)
            hi = max(float(np.max(np.asarray(sg["wl_norm"], dtype=float))) for sg in subs)
            span = max(hi - lo, 1e-9)
            return lo - pad * span, hi + pad * span

        xf = np.asarray(X_fill, dtype=float)
        interp_mu = LinearNDInterpolator(xf, np.asarray(mu, dtype=float).ravel(), fill_value=np.nan)
        interp_std = LinearNDInterpolator(xf, np.asarray(std, dtype=float).ravel(), fill_value=np.nan)
        wl_u_full = np.sort(np.unique(xf[:, 0]))
        ph_nd = max(6, min(14, nd_sp))
        _gp_coalesce_dx2 = 2e-5
        ph_span_keys = float(np.ptp(np.asarray(phase_keys, dtype=float))) if len(phase_keys) > 1 else 0.0
        coalesce_gp = len(phase_keys) >= 2 and ph_span_keys <= _gp_coalesce_dx2
        gp_ph_med = float(np.median(np.asarray(phase_keys, dtype=float))) if coalesce_gp else None

        def _plot_gp_slice(sp_ph: float, subs_k: list, color: Any, lab: str) -> None:
            wl_u = wl_u_full.astype(float)
            if subs_k:
                lo_n, hi_n = _wl_norm_support_union_pr(subs_k)
                if np.isfinite(lo_n) and np.isfinite(hi_n):
                    wl_u = wl_u[(wl_u >= lo_n) & (wl_u <= hi_n)]
            if not wl_u.size:
                return
            pts = np.column_stack([wl_u, np.full(wl_u.size, float(sp_ph), dtype=float)])
            mu_lat = interp_mu(pts)
            st_lat = interp_std(pts)
            wls_line = denorm_ln_wavelength(wl_u, gn)
            o = np.argsort(wls_line)
            w_ord = wls_line[o]
            mu_lin = scaled_ln_to_linear(mu_lat[o], gn)
            sig_lin = linear_flux_yerr(mu_lat[o], st_lat[o], gn)
            ok = np.isfinite(mu_lin) & np.isfinite(sig_lin)
            if np.any(ok):
                ax.fill_between(w_ord[ok], (mu_lin - sig_lin)[ok], (mu_lin + sig_lin)[ok], color=color, alpha=0.18)
                ax.plot(
                    w_ord[ok],
                    mu_lin[ok],
                    color=color,
                    lw=1.35,
                    ls="--",
                    alpha=0.9,
                    label=lab,
                )

        if coalesce_gp and gp_ph_med is not None:
            subs_all: list[dict] = []
            for pk in phase_keys:
                subs_all.extend(by_phase.get(pk, []))
            _plot_gp_slice(
                gp_ph_med,
                subs_all,
                "0.2",
                f"GP μ (norm x₂≈{gp_ph_med:.{ph_nd}f}, coalesced Δx₂={ph_span_keys:.2e})",
            )
        else:
            for k_sp, sp_phase in enumerate(phase_keys):
                subs_k = by_phase[sp_phase]
                c = spec_colors[k_sp % len(spec_colors)]
                _plot_gp_slice(
                    float(sp_phase),
                    subs_k,
                    c,
                    f"GP μ (norm x₂={float(sp_phase):.{ph_nd}f})",
                )
        n_spec_pts = sum(r["flux_lin"].size for r in scaled_rows)
        n_seg_total = len(scaled_rows)
        for k_sp, sp_phase in enumerate(phase_keys):
            subs = by_phase.get(sp_phase, [])
            if not subs:
                continue
            sp_color = spec_colors[k_sp % len(spec_colors)]
            n_at = sum(s["flux_lin"].size for s in subs)
            delta = sp_phase - spec_phase
            for si, sg in enumerate(subs):
                sc = float(sg["scale"])
                wl_p = denorm_ln_wavelength(sg["wl_norm"], gn)
                y_p = sg["flux_lin"] * sc
                y_e = sg["eflux_lin"] * sc
                lbl = (
                    f"spec phase {sp_phase:.4g} (?={delta:+.3g}, n={n_at})"
                    + (" (overlap-scaled plot)" if overlap_scale_display and n_ov > 0 else "")
                    if si == 0
                    else None
                )
                ax.errorbar(
                    wl_p,
                    y_p,
                    yerr=y_e,
                    fmt=".",
                    ms=2,
                    lw=0.4,
                    elinewidth=0.4,
                    color=sp_color,
                    alpha=0.85,
                    label=lbl,
                )

        phot_mask = point_class == gu.PHOT
        phot_near = phot_mask & (np.abs(X_train[:, 1] - spec_phase) <= near_sim_tol)
        n_phot_at = int(phot_near.sum())
        if n_phot_at:
            ax.errorbar(
                denorm_ln_wavelength(X_train[phot_near, 0], gn),
                scaled_ln_to_linear(y_train[phot_near], gn),
                yerr=linear_flux_yerr(y_train[phot_near], yerr_train[phot_near], gn),
                fmt="o",
                ms=3,
                lw=0.4,
                elinewidth=0.4,
                mfc="none",
                mec="red",
                ecolor="red",
                alpha=0.7,
                label=f"phot within ?={near_sim_tol:g} (n={n_phot_at})",
            )
        ov_txt = ""
        if overlap_scale_display and n_ov > 0:
            ov_txt = f"; overlap-ratio scales applied ({n_ov} segment links)"
        ax.set_title(
            f"requested log10(phase)={requested:.3g} -> spec phase {spec_phase:.4g} "
            f"(?req={spec_phase - requested:+.3g}); "
            f"{near_spec_phases.size} near-simultaneous spec phase(s) within ?={near_sim_tol:g}, "
            f"{n_spec_pts} spec pts / {n_seg_total} λ-segments"
            f"{ov_txt}",
            fontsize=8,
        )
        ax.set_ylabel("flux (linear)" if not gn.get("_normalized_only") else "mu")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="best", ncol=1)
    axes[-1].set_xlabel(("log10(wavelength)" if not gn.get("_normalized_only") else "normalized log10(wavelength)"))
    sub = ""
    if not gn.get("_normalized_only"):
        sub = ". Axes denormalized from bundle meta."
    disp = ""
    if overlap_scale_display:
        disp = "\nSpec points: contiguous λ segments linked by overlap median ratios (display scaling only)."
    fig.suptitle(
        "Spectra at the nearest available training-spectrum phase (GP +/- 1 sigma)\n"
        "near-simultaneous spec phases in viridis; dashed GP μ linearly interpolated on X_fill to each phase; phot in red"
        f"{sub}{disp}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_phase_profile_figure(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    save_path: str,
    gn: Optional[dict] = None,
    n_wls: int = 6,
) -> None:
    """Phase profile (mu vs phase, +/- std band) at a few fixed wavelengths.

    Picks ``n_wls`` evenly-spaced wavelengths from X_fill and overlays the
    training points whose normalized wavelength is the closest of all
    prediction wavelengths.
    """
    gn = gn or bmeta.identity_grid_norm()
    wls_pred = np.unique(X_fill[:, 0])
    pick = wls_pred[np.linspace(0, wls_pred.size - 1, n_wls).astype(int)]

    fig, axes = plt.subplots(n_wls, 1, figsize=(10, 1.6 * n_wls), sharex=True)
    if n_wls == 1:
        axes = [axes]

    for ax, wls in zip(axes, pick):
        mask = X_fill[:, 0] == wls
        ph = denorm_ln_phase_days(X_fill[mask, 1], gn)
        m = scaled_ln_to_linear(mu[mask], gn)
        sspread = linear_flux_yerr(mu[mask], std[mask], gn)
        order = np.argsort(ph)
        ph = ph[order]
        m = m[order]
        sspread = sspread[order]
        ax.fill_between(ph, m - sspread, m + sspread, color="steelblue", alpha=0.25, label="+/- 1 sigma")
        ax.plot(ph, m, color="steelblue", lw=1.0, label="mu")

        # Pick training points whose nearest wls (in the full X_fill grid) is this one.
        if X_train.size:
            nearest = wls_pred[np.argmin(np.abs(wls_pred[None, :] - X_train[:, [0]]), axis=1)]
            sel = nearest == wls
            if sel.any():
                for cls, marker, color in (
                    (gu.PHOT, "o", "red"),
                    (gu.SPEC, ".", "k"),
                ):
                    cm = sel & (point_class == cls)
                    if not cm.any():
                        continue
                    ax.errorbar(
                        denorm_ln_phase_days(X_train[cm, 1], gn),
                        scaled_ln_to_linear(y_train[cm], gn),
                        yerr=linear_flux_yerr(y_train[cm], yerr_train[cm], gn),
                        fmt=marker,
                        ms=3 if cls == gu.PHOT else 2,
                        lw=0.4,
                        elinewidth=0.4,
                        color=color,
                        alpha=0.7,
                    )
        wlab = denorm_ln_wavelength(np.array([wls]), gn)[0]
        ax.set_ylabel(
            ("log10 wl=%.4f" % wlab if not gn.get("_normalized_only") else f"norm wls={wls:.3f}"),
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel(
        "log10(phase days)" if not gn.get("_normalized_only") else "normalized log10(phase days)"
    )
    fig.suptitle("Phase profiles at fixed wavelengths (mu +/- 1 sigma, training overlaid)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="matern52_linear_opt")
    p.add_argument("--bundle", default=DEFAULT_BUNDLE)
    p.add_argument(
        "--meta",
        default=None,
        help="bundle meta JSON (default: sibling <stem>_meta.json next to --bundle)",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"parent of <tag>/ (same as run_gp --output-dir). Default: {DEFAULT_OUTPUT_DIR!r}",
    )
    p.add_argument(
        "--spectrum-phases",
        type=str,
        default="-2 -1 0 0.5 1",
        help="space-separated list of normalized log10(phase) values for spectrum plots",
    )
    p.add_argument(
        "--spectrum-tolerance",
        type=float,
        default=0.05,
        help="tolerance (normalized log10(phase) units) for treating other "
             "spec training phases as 'near-simultaneous' to the chosen phase. "
             "All such spectra are overlaid in the spectrum panel.",
    )
    p.add_argument(
        "--no-spec-overlap-scale",
        action="store_true",
        default=False,
        help="disable display-only λ-segment flux rescaling across wavelength overlaps "
             "(spectra overlays use raw calibrated flux)",
    )
    p.add_argument(
        "--spec-segment-gap-factor",
        type=float,
        default=35.0,
        help="split spectroscopy rows into segments when Δlog10 λ exceeds this × median Δ within a nominal phase",
    )
    p.add_argument(
        "--spec-min-gap-norm",
        type=float,
        default=3e-3,
        help="minimum normalized log10-wavelength gap to force a new segment regardless of median spacing",
    )
    p.add_argument(
        "--spec-phase-decimals",
        type=int,
        default=9,
        help="group spectroscopy rows by round(x₂, N) in spectrum overlays (match bundle_scale_pipeline; "
        "-1 = legacy merge with a single atol)",
    )
    p.add_argument(
        "--heatmap-raw",
        action="store_true",
        default=False,
        help="also write gp_mu_heatmap_raw.png from predictions['mu_raw'] when present "
        "(diagnostic: stripes here are from the GP, not early-time post-processing).",
    )
    p.add_argument(
        "--heatmap-normalized",
        action="store_true",
        default=False,
        help="also write gp_mu_heatmap_normalized.png (and raw variant with --heatmap-raw) "
        "on normalized GP coordinates with mu latent colorbar.",
    )
    args = p.parse_args(argv)

    args.output_dir = os.path.abspath(os.path.expanduser(str(args.output_dir)))

    tag = str(args.tag).strip()
    if not tag:
        print(
            "[plot_results] ERROR: --tag is empty. Pass the same name as ``run_gp.py -t`` "
            f"(default tag is auto-generated from flags if you omit ``-t``). "
            f"Outputs live under ``<output-dir>/<tag>/predictions.npz`` (default output-dir is {DEFAULT_OUTPUT_DIR!r}).",
            file=sys.stderr,
        )
        return 1

    run_dir = os.path.join(args.output_dir, tag)
    pred_path = os.path.join(run_dir, "predictions.npz")
    config_path = os.path.join(run_dir, "config.json")
    figs_dir = os.path.join(run_dir, "figs")

    if not os.path.exists(pred_path):
        print(
            f"[plot_results] ERROR: {pred_path} not found. Run run_gp.py first with matching "
            f"``-o {args.output_dir!r}`` and ``-t {tag!r}``.\n"
            "If you used ``run_gp -o runs`` from another working directory, pass that directory "
            "as ``plot_results.py --output-dir`` (resolved absolute paths must match).",
            file=sys.stderr,
        )
        return 1
    _ensure_dir(figs_dir)

    print(f"[plot_results] loading {pred_path}")
    preds = np.load(pred_path, allow_pickle=False)
    X_fill = preds["X_fill"]
    mu = preds["mu"]
    std = preds["std"]
    point_class = preds["point_class_train"]
    sigma_eff = preds["sigma_eff_train"] if "sigma_eff_train" in preds.files else None
    mu_train = preds["mu_train"] if "mu_train" in preds.files else None

    print(f"[plot_results] loading {args.bundle}")
    bundle = np.load(args.bundle, allow_pickle=False)
    X = np.asarray(bundle["X"], dtype=float)
    y = np.asarray(bundle["y"], dtype=float)
    yerr = np.asarray(bundle["yerr"], dtype=float)
    if "train_row_index_orig" in preds.files:
        oi = np.asarray(preds["train_row_index_orig"], dtype=np.int64).ravel()
        if "mu_train" in preds.files and int(preds["mu_train"].shape[0]) != int(oi.size):
            print(
                "[plot_results] ERROR: train_row_index_orig length does not match mu_train",
                file=sys.stderr,
            )
            return 1
        if int(oi.max()) >= X.shape[0] or int(oi.min()) < 0:
            print("[plot_results] ERROR: train_row_index_orig out of range for bundle X", file=sys.stderr)
            return 1
        X, y, yerr = X[oi], y[oi], yerr[oi]
        print(
            f"[plot_results] sliced bundle rows to GP training subset: N={X.shape[0]} "
            f"(full bundle N={int(bundle['X'].shape[0])}) via train_row_index_orig"
        )

    cfg: Optional[dict] = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        suptitle = (
            f"tag={tag} | log_lik={cfg.get('log_likelihood_at_compute', float('nan')):.2f} | "
            f"sigma_phot={cfg['config'].get('sigma_phot', float('nan')):.4g} | "
            f"sigma_spec={cfg['config'].get('sigma_spec', float('nan')):.4g}"
        )
    else:
        suptitle = f"tag={tag}"
    print(f"[plot_results] {suptitle}")

    gn = bmeta.grid_norm_from_bundle_or_meta(args.bundle, meta_path=args.meta)
    mp = args.meta or bmeta.bundle_meta_json_path(args.bundle)
    if gn.get("_normalized_only") and cfg and isinstance(cfg.get("grid_norm_info"), dict):
        gn = dict(cfg["grid_norm_info"])
        gn["_normalized_only"] = False
        print(
            f"[plot_results] grid scaling from saved run config ({config_path} grid_norm_info); "
            f"bundle meta {mp!r} had no grid_norm_info"
        )
    elif gn.get("_normalized_only"):
        print(
            f"[plot_results] WARNING: no usable grid_norm_info; tried {mp!r} and run config",
            file=sys.stderr,
        )
    else:
        print(f"[plot_results] grid scaling from bundle meta ({mp!r}); preferred over run config for ln-flux denorm")

    print(f"[plot_results] mu range [{mu.min():.4g}, {mu.max():.4g}], "
          f"std range [{std.min():.4g}, {std.max():.4g}]")

    _make_wavelength_slice_figure(
        X_fill[:, 0], X_fill[:, 1], mu,
        X, y, yerr, point_class, gn,
        use_log10_phase_axis=True,
        log_y=True,
        save_path=os.path.join(figs_dir, "gp_results_wavelength_slices.pdf"),
        suptitle=suptitle,
    )
    phase_note = ""
    if not gn.get("_normalized_only"):
        phase_note = " (phase axis in days)"
    _make_wavelength_slice_figure(
        X_fill[:, 0], X_fill[:, 1], mu,
        X, y, yerr, point_class, gn,
        use_log10_phase_axis=False,
        log_y=False,
        save_path=os.path.join(figs_dir, "gp_results_wavelength_slices_linear_phase_linear_flux.pdf"),
        suptitle=suptitle + phase_note,
    )

    train_phases = np.unique(X[:, 1])
    latent_label = r"posterior $\sigma$ (normalized latent)"
    _make_heatmap(
        X_fill,
        mu,
        title=f"GP posterior mean (linear flux) — {tag}" if not gn.get("_normalized_only") else f"GP posterior mean (mu) — {tag}",
        cbar_label="flux (linear)" if not gn.get("_normalized_only") else "mu (normalized latent)",
        save_path=os.path.join(figs_dir, "gp_mu_heatmap.png"),
        overlay_training_phases=train_phases,
        gn=gn,
        linearize_values=bool(not gn.get("_normalized_only")),
    )
    if args.heatmap_raw and "mu_raw" in preds.files:
        mu_raw = preds["mu_raw"]
        _make_heatmap(
            X_fill,
            mu_raw,
            title=(
                (f"GP posterior mean raw (pre early-time post) — {tag}")
                if not gn.get("_normalized_only")
                else (f"GP posterior mean raw (mu) — {tag}")
            ),
            cbar_label="flux (linear)" if not gn.get("_normalized_only") else "mu (normalized latent)",
            save_path=os.path.join(figs_dir, "gp_mu_heatmap_raw.png"),
            overlay_training_phases=train_phases,
            gn=gn,
            linearize_values=bool(not gn.get("_normalized_only")),
        )
    if args.heatmap_normalized:
        gn_norm = bmeta.identity_grid_norm()
        _make_heatmap(
            X_fill,
            mu,
            title=f"GP posterior mean (mu) — {tag}",
            cbar_label="mu (normalized latent)",
            save_path=os.path.join(figs_dir, "gp_mu_heatmap_normalized.png"),
            overlay_training_phases=train_phases,
            gn=gn_norm,
            linearize_values=False,
        )
        if args.heatmap_raw and "mu_raw" in preds.files:
            _make_heatmap(
                X_fill,
                preds["mu_raw"],
                title=f"GP posterior mean raw (mu) — {tag}",
                cbar_label="mu (normalized latent)",
                save_path=os.path.join(figs_dir, "gp_mu_heatmap_raw_normalized.png"),
                overlay_training_phases=train_phases,
                gn=gn_norm,
                linearize_values=False,
            )
    _make_heatmap(
        X_fill,
        std,
        title=f"GP posterior std — {tag}",
        cbar_label=latent_label if not gn.get("_normalized_only") else "std",
        save_path=os.path.join(figs_dir, "gp_std_heatmap.png"),
        cmap="magma",
        overlay_training_phases=train_phases,
        gn=gn,
        linearize_values=False,
    )
    _make_training_coverage(X, y, point_class, X_fill, os.path.join(figs_dir, "training_coverage.png"), gn=gn)

    _make_phase_profile_figure(
        X_fill, mu, std, X, y, yerr, point_class,
        save_path=os.path.join(figs_dir, "gp_mu_phase_profiles.png"),
        gn=gn,
    )

    requested_phases = np.array(
        [float(s) for s in args.spectrum_phases.split() if s.strip()],
        dtype=float,
    )
    if requested_phases.size:
        _make_spectrum_figure(
            X_fill,
            mu,
            std,
            X,
            y,
            yerr,
            point_class,
            requested_phases=requested_phases,
            save_path=os.path.join(figs_dir, "gp_spectra.png"),
            gn=gn,
            near_sim_tol=args.spectrum_tolerance,
            overlap_scale_display=not args.no_spec_overlap_scale,
            segment_gap_factor=args.spec_segment_gap_factor,
            min_segment_gap_norm=args.spec_min_gap_norm,
            spec_phase_decimals=int(args.spec_phase_decimals),
        )

    if mu_train is not None and sigma_eff is not None:
        _make_residual_histograms(
            y, mu_train, sigma_eff, point_class,
            save_path=os.path.join(figs_dir, "training_residuals.png"),
        )

    print("[plot_results] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
