"""Build 2D GP training grid and run 2D GP inference for one outer-loop iteration."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import pipeline_config as pconf
import GP2dim_utils_iter as gpiter
from gp_full_spectra_export import export_full_gp_spectra
from iter_gp_predictions_export import save_iter_gp_config_json, save_iter_predictions_npz
from twodim_grid_prep import FullMangledSeries_Class


def resolve_gp2dim_module():
    """Return merged GP2dim helper module."""
    import GP2dim_utils as GP2dim

    return GP2dim
    import GP2dim_utils as GP2dim

    return GP2dim


def _prior_path(coco_path: str) -> tuple[Optional[str], str]:
    rel = getattr(pconf, "ITER_GP_PRIOR_FILE", None)
    if not rel:
        return None, ""
    base = (coco_path or pconf.COCO_PATH).rstrip(os.sep)
    folder = os.path.join(base, "Inputs", "2DIM_priors")
    return rel, folder


def _apply_iter_dense_grid_flags(spec_class) -> None:
    """Enable log-uniform phase columns on X_fill for iter runs when configured."""
    use_iter_dense = bool(getattr(pconf, "ITER_GP_DENSE_PREDICT_GRID", True))
    if use_iter_dense:
        spec_class.gp_predict_dense_log_phase = True
        spec_class.gp_predict_dense_log_phase_n = int(
            getattr(pconf, "ITER_GP_DENSE_PREDICT_GRID_N", 100)
        )
    else:
        spec_class.gp_predict_dense_log_phase = False
        spec_class.gp_predict_dense_log_phase_n = 64


def _iter_grid_delta() -> float:
    return float(getattr(pconf, "ITER_GP_GRID_DELTA", 50.0))


def _build_iter_spec_class(
    snname: str,
    *,
    t0_fix: float,
    mode: str,
    mangled_spectra_dir: str,
    gp_runs_dir: str,
    output_dir: str,
    filters_dir: Optional[str] = None,
    csp_sne: Sequence[str] | None = None,
    warm_start_config_path: str | None = None,
    verbose: bool = False,
    grid_delta: float | None = None,
) -> FullMangledSeries_Class:
    spec_class = FullMangledSeries_Class(
        snname,
        t0_fix,
        mode=mode,
        output_dir=output_dir,
        filters_dir=filters_dir,
        mangled_spectra_dir=mangled_spectra_dir,
        prepare_output_dir=False,
        verbose=verbose,
        DELTA=float(grid_delta if grid_delta is not None else _iter_grid_delta()),
    )
    spec_class.save_plot_path = gp_runs_dir
    spec_class.plot_grid_rebin = bool(getattr(pconf, "PLOT_GRID_REBIN", False))
    spec_class.csp_sne = tuple(
        csp_sne if csp_sne is not None else getattr(pconf, "CSP_SNE", ())
    )
    spec_class.gp_export_minimal = bool(getattr(pconf, "GP_EXPORT_MINIMAL", True))
    _apply_iter_dense_grid_flags(spec_class)
    spec_class.extrapolate_log_phase_span_dex = None
    spec_class.gp_predict_progress = verbose
    spec_class.gp_print_training_size = verbose
    if warm_start_config_path and os.path.isfile(warm_start_config_path):
        spec_class.warm_start_config_json = warm_start_config_path
    return spec_class


def _prepare_iter_grid_arrays(
    snname: str,
    spec_class: FullMangledSeries_Class,
) -> tuple[Any, ...]:
    GP2dim = resolve_gp2dim_module()
    raw_numbers, raw_numbers_err, off_xa, off_ya, grid_ext_columns = GP2dim.prepare_grid(
        snname, spec_class
    )
    y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm = GP2dim.transform2LOG_reshape(
        spec_class, raw_numbers, raw_numbers_err, off_xa, off_ya
    )
    return (
        y_data_nonan,
        y_data_nonan_err,
        x1_data_norm,
        x2_data_norm,
        grid_ext_columns,
    )


def _load_iter_prior(coco_path: str, spec_class: FullMangledSeries_Class) -> tuple[np.ndarray, np.ndarray, bool]:
    GP2dim = resolve_gp2dim_module()
    prior_file, prior_folder = _prior_path(coco_path)
    if prior_file:
        Xprior, yprior = GP2dim.setPRIOR(
            spec_class, PRIOR_file=prior_file, PRIOR_folder=prior_folder
        )
    else:
        Xprior, yprior = np.array([]), np.array([])
    return Xprior, yprior, bool(prior_file)


def run_iter_gp_grid_export_only(
    snname: str,
    *,
    t0_fix: float,
    mode: str,
    mangled_spectra_dir: str,
    export_dir: str,
    output_dir: str,
    coco_path: str,
    filters_dir: Optional[str] = None,
    kernel_wls_scale: Optional[float] = None,
    kernel_time_scale: Optional[float] = None,
    csp_sne: Sequence[str] | None = None,
    grid_delta: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build GP training grid and export ``gp_minimal_export/`` without fitting."""
    os.makedirs(export_dir, exist_ok=True)
    delta = float(grid_delta if grid_delta is not None else _iter_grid_delta())
    spec_class = _build_iter_spec_class(
        snname,
        t0_fix=t0_fix,
        mode=mode,
        mangled_spectra_dir=mangled_spectra_dir,
        gp_runs_dir=export_dir,
        output_dir=output_dir,
        filters_dir=filters_dir,
        csp_sne=csp_sne,
        verbose=verbose,
        grid_delta=delta,
    )
    y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm, grid_ext_columns = (
        _prepare_iter_grid_arrays(snname, spec_class)
    )
    Xprior, yprior, has_prior = _load_iter_prior(coco_path, spec_class)
    kwls = float(kernel_wls_scale or getattr(pconf, "ITER_GP_KERNEL_WLS_SCALE", 0.01))
    ktime = float(kernel_time_scale or getattr(pconf, "ITER_GP_KERNEL_TIME_SCALE", 0.04))
    prep = gpiter.prepare_training_bundle(
        spec_class,
        y_data_nonan,
        y_data_nonan_err,
        x1_data_norm,
        x2_data_norm,
        kwls,
        ktime,
        np.asarray(grid_ext_columns, dtype=float),
        prior=has_prior,
        points=Xprior,
        values=yprior,
    )
    gpiter.export_minimal_bundle(
        spec_class,
        prep,
        prior=has_prior,
        points=Xprior,
        values=yprior,
    )
    bundle_path = os.path.join(export_dir, "gp_minimal_export", "gp_minimal_bundle.npz")
    gn = dict(spec_class.grid_norm_info)
    if verbose:
        print(
            f"[run_iter_gp_grid_export_only] DELTA={delta:.1f} N={prep['n_train']} "
            f"scale_factor={gn.get('scale_factor', float('nan')):.4g} -> {bundle_path}",
            flush=True,
        )
    return {
        "bundle_path": bundle_path,
        "export_dir": export_dir,
        "grid_norm_info": gn,
        "n_train": int(prep["n_train"]),
        "grid_delta": delta,
        "spec_class": spec_class,
        "prep": prep,
    }


def run_iter_gp_fit(
    snname: str,
    *,
    t0_fix: float,
    mode: str,
    mangled_spectra_dir: str,
    gp_runs_dir: str,
    output_dir: str,
    coco_path: str,
    filters_dir: Optional[str] = None,
    kernel_wls_scale: Optional[float] = None,
    kernel_time_scale: Optional[float] = None,
    csp_sne: Sequence[str] | None = None,
    warm_start_config_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build grid from ``mangled_spectra_dir``, run 2D GP, write ``predictions.npz``."""
    os.makedirs(gp_runs_dir, exist_ok=True)

    spec_class = _build_iter_spec_class(
        snname,
        t0_fix=t0_fix,
        mode=mode,
        mangled_spectra_dir=mangled_spectra_dir,
        gp_runs_dir=gp_runs_dir,
        output_dir=output_dir,
        filters_dir=filters_dir,
        csp_sne=csp_sne,
        warm_start_config_path=warm_start_config_path,
        verbose=verbose,
    )
    y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm, grid_ext_columns = (
        _prepare_iter_grid_arrays(snname, spec_class)
    )
    Xprior, yprior, has_prior = _load_iter_prior(coco_path, spec_class)

    kwls = float(kernel_wls_scale or getattr(pconf, "ITER_GP_KERNEL_WLS_SCALE", 0.01))
    ktime = float(kernel_time_scale or getattr(pconf, "ITER_GP_KERNEL_TIME_SCALE", 0.04))

    x1_fill, x2_fill, mu_fill, std_fill, mu_fill_raw = gpiter.run_2DGP_GRID_iter(
        spec_class,
        y_data_nonan,
        y_data_nonan_err,
        x1_data_norm,
        x2_data_norm,
        kwls,
        ktime,
        np.asarray(grid_ext_columns, dtype=float),
        prior=has_prior,
        points=Xprior,
        values=yprior,
    )

    merged = getattr(spec_class, "_twodim_gp_merged", None) or {}

    gn = dict(spec_class.grid_norm_info)
    wls_min = float(np.min(spec_class.grids[0]))
    wls_max = float(np.max(spec_class.grids[0]))
    span_wl = max(wls_max - wls_min, 1e-9)
    n_wl = int(getattr(spec_class, "gp_predict_n_wavelength", 300))
    wl_step = float(getattr(spec_class, "gp_predict_wl_step", 0.01))
    n_wl_use = max(2, min(n_wl, int(np.ceil(span_wl / wl_step)) + 1))
    wls_log_grid = np.linspace(wls_min, wls_max, n_wl_use)
    phase_cols = np.asarray(grid_ext_columns, dtype=float)

    pred_path = os.path.join(gp_runs_dir, "predictions.npz")
    save_iter_predictions_npz(
        pred_path,
        mu=mu_fill,
        mu_raw=mu_fill_raw,
        std=std_fill,
        x1_fill=x1_fill,
        x2_fill=x2_fill,
        grid_norm_info=gn,
        merged=merged,
        wls_log_grid=wls_log_grid,
        phase_log10_columns=phase_cols,
    )
    save_iter_gp_config_json(
        os.path.join(gp_runs_dir, "config.json"),
        merged=merged,
        grid_norm_info=gn,
    )

    mu_key = str(getattr(pconf, "GP_PREDICT_MU_KEY", "mu"))
    mu_for_export = mu_fill_raw if mu_key == "mu_raw" else mu_fill
    exported_paths: list[str] = []
    if bool(getattr(pconf, "ITER_GP_EXPORT_FULL_GP", True)):
        exported_paths = export_full_gp_spectra(
            spec_class,
            x1_fill=x1_fill,
            x2_fill=x2_fill,
            mu_fill=mu_for_export,
            std_fill=std_fill,
            grid_ext_columns=grid_ext_columns,
            y_data_nonan=y_data_nonan,
            out_dir=gp_runs_dir,
            mu_key=mu_key,
        )

    return {
        "predictions_path": pred_path,
        "mu": mu_fill,
        "mu_raw": mu_fill_raw,
        "std": std_fill,
        "x1_fill": x1_fill,
        "x2_fill": x2_fill,
        "X_fill": merged.get("X_fill", np.column_stack([x1_fill, x2_fill])),
        "wls_log_grid": wls_log_grid,
        "phase_log10_columns": phase_cols,
        "grid_norm_info": gn,
        "spec_class": spec_class,
        "grid_ext_columns": np.asarray(grid_ext_columns, dtype=float),
        "y_data_nonan": y_data_nonan,
        "y_data_nonan_err": y_data_nonan_err,
        "x1_data_norm": x1_data_norm,
        "x2_data_norm": x2_data_norm,
        "exported_spectra_paths": exported_paths,
        "gp_runs_dir": gp_runs_dir,
        "gp_merged": merged,
        "point_class_train": merged.get("point_class_train"),
        "mu_train": merged.get("mu_train"),
    }


def load_predictions_npz(path: str) -> dict[str, Any]:
    data = dict(np.load(path, allow_pickle=True))
    gn_raw = data.get("grid_norm_info")
    if isinstance(gn_raw, np.ndarray) and gn_raw.dtype.kind in ("U", "S", "O"):
        gn = json.loads(str(gn_raw.item() if gn_raw.ndim == 0 else gn_raw[0]))
    elif isinstance(gn_raw, (str, bytes)):
        gn = json.loads(gn_raw)
    else:
        gn = dict(gn_raw) if gn_raw is not None else {}
    data["grid_norm_info"] = gn
    if "X_fill" not in data and "x1_fill" in data and "x2_fill" in data:
        data["X_fill"] = np.column_stack(
            [np.asarray(data["x1_fill"]), np.asarray(data["x2_fill"])]
        )
    return data


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Iter GP grid: full fit or export-only bundle.")
    sub = p.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export-only", help="Build grid and export gp_minimal_bundle (no GP fit)")
    exp.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    exp.add_argument("--t0-fix", type=float, default=None)
    exp.add_argument("--mode", default="extrapolate_spectra")
    exp.add_argument("--mangled-spectra-dir", required=True)
    exp.add_argument("--export-dir", required=True)
    exp.add_argument("--output-dir", default=None)
    exp.add_argument("--coco-path", default=None)
    exp.add_argument("--grid-delta", type=float, default=None)
    exp.add_argument("--quiet", action="store_true")

    args = p.parse_args()
    coco = (args.coco_path or pconf.COCO_PATH).rstrip(os.sep) + os.sep
    output_dir = args.output_dir or os.path.join(coco, "Outputs")
    t0 = args.t0_fix
    if t0 is None:
        t0 = float(getattr(pconf, "SN_EXPLOSION_MJD", {}).get(args.snname, 0.0))

    if args.cmd == "export-only":
        run_iter_gp_grid_export_only(
            args.snname,
            t0_fix=t0,
            mode=args.mode,
            mangled_spectra_dir=args.mangled_spectra_dir,
            export_dir=args.export_dir,
            output_dir=output_dir,
            coco_path=coco,
            grid_delta=args.grid_delta,
            verbose=not args.quiet,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
