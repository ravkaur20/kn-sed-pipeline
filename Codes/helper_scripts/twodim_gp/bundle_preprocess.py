#!/usr/bin/env python3
"""Post-load editing of training bundle: phot/spec overrides, telluric spike repair.

Run **after** you have a collaborator ``*.npz`` (and meta JSON for physical units).
Writes a new ``*.npz`` with updated ``y``, ``yerr``, optional ``train_obs_class``,
``telluric_bad_mask``, ``train_obs_class``.

Telluric cleaning (per spectroscopic phase): median-filter residual threshold in linear flux,
interpolate bad λ pixels from neighbours, set ``yerr`` to a huge finite value so GP ignores them.

Examples::

    # Find training rows near a phase / physical log10 λ (Å-scale column as plotted)
    python bundle_preprocess.py find \\
        --bundle gp_minimal_bundle.npz \\
        --norm-phase -1.822522 --log10-wavelength 3.73

    # Mark misclassified photometry rows + repair telluric spikes on one exposure
    python bundle_preprocess.py \\
        --input gp_minimal_bundle.npz \\
        --output gp_bundle_edited.npz \\
        --phot-indices 1234 \\
        --telluric --telluric-phases 0.595732

    # Repair all spectroscopic phases (can be slow)
    python bundle_preprocess.py -i gp_minimal_bundle.npz -o gp_edit.npz --telluric --telluric-all-spec

    # Reclassify mis-ID “spectra” by plot index + telluric on a few exposures only (see configs/)
    python bundle_preprocess.py preprocess \\
        -i gp_minimal_bundle.npz -o gp_bundle_fixed.npz \\
        --corrections-json configs/gp_minimal_bundle_plot_index_fixes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

try:
    from . import bundle_meta as bmeta
    from . import gp_utils as gu
except ImportError:
    import bundle_meta as bmeta
    import gp_utils as gu

# Finite stand-in for "infinite" noise so ``gp.compute`` stays numerical.
YERR_DISABLED = np.float64(1e30)


def canonical_sorted_phases(phase_col: np.ndarray, *, atol: float) -> np.ndarray:
    """Sorted unique exposure phases matching ``plot_separated_training_data`` grouping.

    ``atol`` is the same meaning as that script's ``--phase-match-atol`` (merge within atol).
    """
    arr = np.sort(np.asarray(phase_col, dtype=float).ravel())
    out: list[float] = []
    for v in arr:
        if not out or all(abs(float(v) - u) > float(atol) for u in out):
            out.append(float(v))
    return np.asarray(out, dtype=float)


def _load_gn(bundle_path: str) -> dict:
    meta_path = bmeta.bundle_meta_json_path(bundle_path)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta.get("grid_norm_info"), dict):
            gn = dict(meta["grid_norm_info"])
            gn["_normalized_only"] = False
            return gn
    return bmeta.identity_grid_norm()


def linear_flux_from_latent(y: np.ndarray, gn: dict) -> np.ndarray:
    m = np.asarray(y, dtype=float)
    if gn.get("_normalized_only"):
        return m
    off = float(gn["offset"])
    sf = float(gn["scale_factor"])
    return np.exp(m * sf + off)


def latent_from_linear(f_lin: np.ndarray, gn: dict) -> np.ndarray:
    fl = np.maximum(np.asarray(f_lin, dtype=float), 1e-300)
    if gn.get("_normalized_only"):
        return np.log(fl)
    off = float(gn["offset"])
    sf = float(gn["scale_factor"])
    return (np.log(fl) - off) / sf


def empirical_bad_pixel_mask(
    f_lin: np.ndarray,
    *,
    median_size: int = 21,
    sigma_k: float = 7.0,
    spike_quantile: float = 99.7,
    spike_factor: float = 2.5,
    dilate: int = 4,
) -> np.ndarray:
    """Flag pixels where robust residual vs median spectrum or extreme spikes."""
    n = f_lin.size
    if n < 5:
        return np.zeros(n, dtype=bool)
    k = min(int(median_size) | 1, n)
    if k % 2 == 0:
        k -= 1
    k = max(3, min(k, n - (1 - n % 2)))
    med = median_filter(f_lin.astype(float), size=k, mode="nearest")
    resid = f_lin.astype(float) - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    robust = 1.4826 * mad + 1e-30
    bad = np.abs(resid) > sigma_k * robust
    thr = np.percentile(f_lin, spike_quantile)
    bad |= f_lin > np.maximum(thr * spike_factor, thr + 5 * robust)
    if dilate > 0:
        bad = binary_dilation(bad, iterations=int(dilate))
    return bad


def repair_phase_rows(
    idx_rows: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gn: dict,
    *,
    median_size: int,
    sigma_k: float,
    dilate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return updated y, yerr and bad_mask slice for indices idx_rows (spectral rows at one phase)."""
    if idx_rows.size < 3:
        return y, yerr, np.zeros_like(y, dtype=bool)

    order = np.argsort(X[idx_rows, 0])
    idx_sorted = idx_rows[order]
    w = X[idx_sorted, 0]
    y_sub = y[idx_sorted].copy()
    ye_sub = yerr[idx_sorted].copy().astype(float)
    f_lin = linear_flux_from_latent(y_sub, gn)

    bad_local = empirical_bad_pixel_mask(
        f_lin,
        median_size=median_size,
        sigma_k=sigma_k,
        dilate=dilate,
    )
    good = ~bad_local
    full_bad = np.zeros_like(y, dtype=bool)
    full_bad[idx_sorted] = bad_local

    if good.sum() < 2:
        return y, yerr, full_bad

    f_new = f_lin.copy()
    f_new[bad_local] = np.interp(w[bad_local], w[good], f_lin[good])
    y_new_sub = latent_from_linear(f_new, gn)
    ye_new_sub = ye_sub.copy()
    ye_new_sub[bad_local] = YERR_DISABLED

    y_out = y.copy()
    yerr_out = yerr.copy().astype(float)
    y_out[idx_sorted] = y_new_sub
    yerr_out[idx_sorted] = ye_new_sub
    return y_out, yerr_out, full_bad


def cmd_find(args: argparse.Namespace) -> int:
    gn = _load_gn(args.bundle)
    d = np.load(args.bundle, allow_pickle=False)
    X = np.asarray(d["X"], dtype=float)
    pc = gu.effective_point_class(
        X,
        threshold=args.threshold,
        train_obs_class=np.asarray(d["train_obs_class"]) if "train_obs_class" in d.files else None,
    )

    # Physical log10 λ from normalized X[:,0]
    x1m, x1s = float(gn["x1_mean"]), float(gn["x1_std"])
    if gn.get("_normalized_only"):
        logwl_phys = X[:, 0]
    else:
        logwl_phys = x1m + x1s * X[:, 0]

    ph_tgt = float(args.norm_phase)
    wl_tgt = float(args.log10_wavelength)
    dt_ph = float(args.phase_tolerance)
    dt_wl = float(args.log10_wl_tolerance)

    m = (
        np.abs(X[:, 1] - ph_tgt) <= dt_ph
        ) & (
        np.abs(logwl_phys - wl_tgt) <= dt_wl
    )
    hits = np.nonzero(m)[0]
    print(f"[find] bundle={args.bundle!r} meta_grid={'no' if gn.get('_normalized_only') else 'yes'}")
    print(f"[find] phase_norm {ph_tgt:g} ± {dt_ph:g}, log10λ {wl_tgt:g} ± {dt_wl:g}")
    print(f"[find] matched rows: {hits.size}")
    for i in hits:
        print(
            f"  idx={int(i):5d}  class={pc[i]!r}  "
            f"x1_norm={X[i, 0]:.6g} x2_norm={X[i, 1]:.6g}  "
            f"log10λ_phys={logwl_phys[i]:.6g}"
        )
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    gn = _load_gn(args.input)
    raw = np.load(args.input, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float).copy()
    yerr = np.asarray(data["yerr"], dtype=float).copy()

    n = X.shape[0]
    corrections: dict = {}
    if str(args.corrections_json).strip():
        cpath = os.path.abspath(os.path.expanduser(str(args.corrections_json).strip()))
        if not os.path.isfile(cpath):
            print(f"[preprocess] ERROR: corrections json not found: {cpath!r}", file=sys.stderr)
            return 2
        with open(cpath, encoding="utf-8") as cf:
            corrections = json.load(cf)
        if not isinstance(corrections, dict):
            print("[preprocess] ERROR: corrections json must be an object", file=sys.stderr)
            return 2

    thr = int(corrections.get("phot_spec_threshold", args.threshold))
    plot_atol = float(corrections.get("spec_plot_phase_atol", args.spec_plot_phase_atol))

    force_list = corrections.get("force_phot_spec_plot_indices")
    if force_list is None:
        force_list = _parse_int_list(args.force_phot_spec_plot_indices or "")
    else:
        force_list = [int(x) for x in force_list]

    tell_idx_list = corrections.get("telluric_spec_plot_indices")
    if tell_idx_list is None:
        tell_idx_list = _parse_int_list(args.telluric_spec_plot_indices or "")
    else:
        tell_idx_list = [int(x) for x in tell_idx_list]

    if args.phot_indices.strip():
        if "train_obs_class" in data:
            obs = np.asarray(data["train_obs_class"]).astype("<U8").copy()
        else:
            obs = np.asarray(gu.classify_points(X, threshold=thr)).astype("<U8")
        for k in _parse_int_list(args.phot_indices):
            if 0 <= k < n:
                obs[k] = gu.PHOT
        data["train_obs_class"] = obs
        print(f"[preprocess] train_obs_class: forced phot at row indices {args.phot_indices}")

    if "train_obs_class" in data:
        obs = np.asarray(data["train_obs_class"]).astype("<U8").copy()
    else:
        obs = np.asarray(gu.classify_points(X, threshold=thr)).astype("<U8")

    pc_snapshot = gu.effective_point_class(X, threshold=thr, train_obs_class=obs)
    spec_m0 = pc_snapshot == gu.SPEC
    if not np.any(spec_m0):
        print("[preprocess] WARNING: no spec rows in initial classification", file=sys.stderr)
        phases_snap = np.zeros(0, dtype=float)
    else:
        phases_snap = canonical_sorted_phases(X[spec_m0, 1], atol=plot_atol)
        print(
            f"[preprocess] spec_plot_phase_atol={plot_atol:g} -> {phases_snap.size} spec exposure(s) "
            f"(same ordering as plot_separated_training_data spec_NNN)"
        )

    for ii in force_list:
        if ii < 0 or ii >= phases_snap.size:
            print(
                f"[preprocess] WARNING: force_phot spec_plot index {ii} out of range 0..{phases_snap.size - 1}",
                file=sys.stderr,
            )
            continue
        ph = float(phases_snap[ii])
        m = np.isclose(X[:, 1], ph, rtol=0.0, atol=plot_atol)
        nhit = int(m.sum())
        obs[m] = gu.PHOT
        print(f"[preprocess] force_phot spec_plot_index={ii} phase_norm≈{ph:.10g} rows={nhit}")

    if force_list:
        data["train_obs_class"] = obs

    pc_after = gu.effective_point_class(X, threshold=thr, train_obs_class=obs)
    spec_m1 = pc_after == gu.SPEC

    tell_phase_tol = max(float(args.phase_match_tol), float(plot_atol))

    phases_to_clean: list[float] = []
    if tell_idx_list:
        seen_ph: set[float] = set()
        for ii in tell_idx_list:
            if ii < 0 or ii >= phases_snap.size:
                print(
                    f"[preprocess] WARNING: telluric spec_plot index {ii} out of range 0..{phases_snap.size - 1}",
                    file=sys.stderr,
                )
                continue
            ph = float(phases_snap[ii])
            if ph not in seen_ph:
                seen_ph.add(ph)
                phases_to_clean.append(ph)
        print(f"[preprocess] telluric: {len(phases_to_clean)} phase(s) from spec_plot indices only")
    elif args.telluric:
        spec_phases = np.unique(X[spec_m1, 1])
        if args.telluric_all_spec:
            phases_to_clean = [float(p) for p in spec_phases]
        else:
            req = _parse_float_list(args.telluric_phases)
            if not req:
                print(
                    "[preprocess] ERROR: --telluric requires --telluric-phases or --telluric-all-spec",
                    file=sys.stderr,
                )
                return 2
            sp_list = np.asarray(spec_phases, dtype=float)
            seen_ph = set()
            for raw_ph in req:
                j = int(np.argmin(np.abs(sp_list - raw_ph)))
                snapped = float(sp_list[j])
                if np.abs(snapped - raw_ph) > 0.05:
                    print(
                        f"[preprocess] WARNING: requested phase {raw_ph:g} far from nearest spec phase {snapped:g}"
                    )
                if snapped not in seen_ph:
                    seen_ph.add(snapped)
                    phases_to_clean.append(snapped)

    bad_global = np.zeros(n, dtype=bool)

    for ph in phases_to_clean:
        rows = np.nonzero(
            (pc_after == gu.SPEC) & np.isclose(X[:, 1], ph, rtol=0.0, atol=tell_phase_tol)
        )[0]
        if rows.size < 3:
            print(f"[preprocess] telluric: skip phase {ph:g} (only {rows.size} spec rows after reclass)")
            continue
        y, yerr, bad_this = repair_phase_rows(
            rows,
            X,
            y,
            yerr,
            gn,
            median_size=args.median_window,
            sigma_k=args.telluric_sigma,
            dilate=args.telluric_dilate,
        )
        bad_global |= bad_this
        n_bad = int(bad_this.sum())
        print(f"[preprocess] telluric phase={ph:.8g}: flagged {n_bad} pixels")

    data["train_obs_class"] = obs
    data["y"] = y
    data["yerr"] = yerr
    if bad_global.any():
        data["telluric_bad_mask"] = bad_global

    if "y_compute" in data:
        yc = np.asarray(data["y_compute"], dtype=float).copy()
        yc[bad_global] = YERR_DISABLED
        data["y_compute"] = yc

    out_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **{k: data[k] for k in sorted(data.keys())})
    print(f"[preprocess] wrote {out_path} keys={sorted(data.keys())}")
    return 0


def _parse_int_list(s: str) -> list[int]:
    out = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_float_list(s: str) -> list[float]:
    out = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("find", help="Locate training rows by phase + log10 wavelength")
    pf.add_argument("--bundle", required=True)
    pf.add_argument("--norm-phase", type=float, required=True, dest="norm_phase")
    pf.add_argument("--log10-wavelength", type=float, required=True, dest="log10_wavelength")
    pf.add_argument("--phase-tolerance", type=float, default=5e-4)
    pf.add_argument("--log10-wl-tolerance", type=float, default=0.07)
    pf.add_argument("--threshold", type=int, default=50)
    pf.set_defaults(func=cmd_find)

    pp = sub.add_parser("preprocess", help="Edit bundle and save new npz")
    pp.add_argument("--input", "-i", required=True)
    pp.add_argument("--output", "-o", required=True)
    pp.add_argument(
        "--phot-indices",
        default="",
        help="comma-separated train indices to force photometry (writes train_obs_class)",
    )
    pp.add_argument("--threshold", type=int, default=50)
    pp.add_argument("--telluric", action="store_true", help="enable telluric spike repair")
    pp.add_argument(
        "--telluric-phases",
        default="",
        help="comma-separated normalized phase column values (X[:,1]) to clean",
    )
    pp.add_argument("--telluric-all-spec", action="store_true", help="clean every spec phase")
    pp.add_argument(
        "--corrections-json",
        default="",
        help="JSON with spec_plot_phase_atol, phot_spec_threshold, force_phot_spec_plot_indices, "
        "telluric_spec_plot_indices (same numbering as plot_separated_training_data spec_NNN)",
    )
    pp.add_argument(
        "--force-phot-spec-plot-indices",
        default="",
        help="comma-separated spec plot indices (overridden by corrections-json if set there)",
    )
    pp.add_argument(
        "--telluric-spec-plot-indices",
        default="",
        help="comma-separated spec plot indices for telluric repair only (implies telluric on those phases; "
        "overridden by corrections-json)",
    )
    pp.add_argument(
        "--spec-plot-phase-atol",
        type=float,
        default=0.0,
        help="0 = strict X[:,1] equality (matches one plot per np.unique spec phase). "
        "Must match plot_separated_training_data --phase-match-atol when resolving spec_NNN indices",
    )
    pp.add_argument("--phase-match-tol", type=float, default=1e-7)
    pp.add_argument("--median-window", type=int, default=21)
    pp.add_argument("--telluric-sigma", type=float, default=7.0)
    pp.add_argument("--telluric-dilate", type=int, default=4)
    pp.set_defaults(func=cmd_preprocess)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
