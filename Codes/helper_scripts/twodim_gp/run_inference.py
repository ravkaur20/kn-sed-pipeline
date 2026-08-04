"""Collaborator GP: fit hyperparameters + predict on ``X_fill`` (library API).

This module refactors logic from ``twodim_gp/run_gp.py`` so PyCoCo can call it in-process
without writing CLI ``runs/<tag>/`` artifacts (optional JSON via ``write_run_json``).

Axis convention matches ``gp_minimal_bundle`` exports: ``X[:,0]`` = normalized log10(wavelength),
``X[:,1]`` = normalized log10(phase days).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Mapping, Optional, Union

import numpy as np
from scipy.optimize import minimize

import george

from . import gp_utils as gu

PredictChunk = Union[int, str]
BundleInput = Union[str, os.PathLike, np.lib.npyio.NpzFile, Mapping[str, Any]]

PREDICT_CHUNK_DEFAULT = 10_000

# Mirrors ``gp_rjf/run_gp.py`` argparse defaults aligned with collaborator WRITEUP “v5 style” starter.
DEFAULT_KWARGS: dict[str, Any] = {
    "kernel_time": "matern52",
    "kernel_wls": "matern52",
    "additive_time": False,
    "additive_wls": False,
    "mean": "linear",
    "phot_spec_threshold": 50,
    "lw": None,
    "lt": None,
    "lw2": 16.0,
    "lt2": 16.0,
    "lw_short": 0.005,
    "lt_short": 0.04,
    "w_short_w": 0.4,
    "w_short_t": 0.4,
    "log_amp": None,
    "sigma_phot": 0.02,
    "sigma_spec": 0.005,
    "enforce_mono_early": True,
    "enforce_blue_early": True,
    "early_time_cutoff": -4.0,
    "mono_floor_fraction": 0.5,
    "mono_min_slope": 0.005,
    "mono_smoothing_scale": 0.3,
    "optimize": True,
    "max_iter": 60,
    "optimize_subsample": 2500,
    "seed": 0,
    "predict_chunk": PREDICT_CHUNK_DEFAULT,
    "predict_train": False,
    # Optimizer bound overrides (see ``_tightened_optimizer_bounds``).
    "logit_weight_t_min": None,
    "logit_weight_t_max": None,
    "logit_weight_w_min": None,
    "logit_weight_w_max": None,
    "log_metric_t_min": None,
    "log_metric_w_min": None,
    "log_metric_t2_max": None,
    "log_metric_w2_min": None,
    "log_metric_w2_max": None,
    # Explicit σ_spec floor override; if None and ``sigma_spec_adaptive_frac`` is set,
    # floor = max(0.005, frac * median yerr on spec) before optimization.
    "sigma_spec_min": None,
    "sigma_spec_adaptive_frac": 0.35,
}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _build_initial_config(kwargs: Mapping[str, Any], bundle_arrays: Mapping[str, np.ndarray]) -> gu.KernelConfig:
    args = kwargs
    if args.get("additive_wls"):
        lw = float(args["lw"]) if args["lw"] is not None else float(args["lw_short"])
        lw2 = float(args["lw2"])
    else:
        lw = float(args["lw"]) if args["lw"] is not None else float(bundle_arrays["kernel_wls_scale"])
        lw2 = lw * 16.0
    if args.get("additive_time"):
        lt = float(args["lt"]) if args["lt"] is not None else float(args["lt_short"])
        lt2 = float(args["lt2"])
    else:
        lt = float(args["lt"]) if args["lt"] is not None else float(bundle_arrays["kernel_time_scale"])
        lt2 = lt * 16.0

    va = bundle_arrays["y_var_scale"]
    log_amp_seed = args.get("log_amp")
    log_amp = float(log_amp_seed) if log_amp_seed is not None else float(np.log(float(va)))

    return gu.KernelConfig(
        name_t=str(args["kernel_time"]),
        name_w=str(args["kernel_wls"]),
        additive_t=bool(args["additive_time"]),
        additive_w=bool(args["additive_wls"]),
        log_amp=log_amp,
        log_metric_t=float(np.log(lt)),
        log_metric_w=float(np.log(lw)),
        log_metric_t2=float(np.log(lt2)),
        log_metric_w2=float(np.log(lw2)),
        logit_weight_t=_logit(float(args["w_short_t"])),
        logit_weight_w=_logit(float(args["w_short_w"])),
        log_sigma_phot=float(np.log(max(float(args["sigma_phot"]), 1e-6))),
        log_sigma_spec=float(np.log(max(float(args["sigma_spec"]), 1e-6))),
    )


def _apply_warm_start_config(cfg0: gu.KernelConfig, path: str | None) -> int:
    if not path or not os.path.isfile(str(path)):
        return 0
    import json

    with open(path, encoding="utf-8") as wf:
        wj = json.load(wf)
    inner = wj.get("config") if isinstance(wj.get("config"), dict) else wj
    if not isinstance(inner, dict):
        return 0
    applied = cfg0.apply_saved_inner_config(inner)
    return len(applied)


def _enforce_monotone_early(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    cutoff: float,
    floor_fraction: float = 0.5,
    slope_window: int = 5,
    min_slope: float = 0.005,
    smoothing_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, int]:
    mu_out = mu.copy()
    wls_unique = np.unique(X_fill[:, 0])
    n_modified = 0
    scale = max(float(smoothing_scale), 1e-6)
    for wls in wls_unique:
        mask = X_fill[:, 0] == wls
        idx = np.where(mask)[0]
        ph = X_fill[idx, 1]
        order = np.argsort(ph)
        idx_sorted = idx[order]
        ph_sorted = ph[order]
        early_mask = ph_sorted < cutoff
        if not early_mask.any() or early_mask.all():
            continue
        first_in = int(np.where(~early_mask)[0][0])
        t_cutoff = float(ph_sorted[first_in])
        mu_cutoff = float(mu_out[idx_sorted[first_in]])
        n_inside = int((~early_mask).sum())
        win = min(slope_window, n_inside - 1) if n_inside >= 2 else 0
        if win >= 1:
            t_w = ph_sorted[first_in : first_in + win + 1]
            mu_w = mu_out[idx_sorted[first_in : first_in + win + 1]]
            slope = float(np.polyfit(t_w - t_cutoff, mu_w - mu_cutoff, 1)[0])
        else:
            slope = 0.0
        slope = max(slope, float(min_slope))
        mu_extrap_full = mu_cutoff + slope * (ph_sorted - t_cutoff)
        mu_floor = float(floor_fraction) * mu_cutoff
        mu_extrap_full = np.maximum(mu_extrap_full, mu_floor)
        w_blend = 0.5 * (1.0 + np.tanh((ph_sorted - t_cutoff) / scale))
        mu_blend = (1.0 - w_blend) * mu_extrap_full + w_blend * mu_out[idx_sorted]
        diff = np.abs(mu_blend - mu_out[idx_sorted])
        n_modified += int(np.sum(diff > 1e-9))
        mu_out[idx_sorted] = mu_blend
    return mu_out, std, n_modified


def _enforce_blue_early(X_fill: np.ndarray, mu: np.ndarray, cutoff: float) -> tuple[np.ndarray, int]:
    mu_out = mu.copy()
    phases_unique = np.unique(X_fill[:, 1])
    n_modified = 0
    for ph in phases_unique:
        if ph >= cutoff:
            continue
        mask = X_fill[:, 1] == ph
        idx = np.where(mask)[0]
        wls = X_fill[idx, 0]
        order = np.argsort(wls)
        idx_sorted = idx[order]
        mu_in = mu_out[idx_sorted]
        mu_corr = np.minimum.accumulate(mu_in)
        n_modified += int(np.sum(mu_corr != mu_in))
        mu_out[idx_sorted] = mu_corr
    return mu_out, n_modified


def _predict_chunked(
    gp: george.GP, y: np.ndarray, X_query: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    n = X_query.shape[0]
    mu = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        m, v = gp.predict(y, X_query[s0:s1], return_var=True)
        mu[s0:s1] = m
        var[s0:s1] = v
    return mu, var


def _tightened_optimizer_bounds(
    cfg: gu.KernelConfig,
    *,
    sigma_spec_min: Optional[float] = None,
    log_metric_t_min: Optional[float] = None,
    log_metric_w_min: Optional[float] = None,
    log_metric_t2_max: Optional[float] = None,
    log_metric_w2_min: Optional[float] = None,
    log_metric_w2_max: Optional[float] = None,
    logit_weight_t_min: Optional[float] = None,
    logit_weight_t_max: Optional[float] = None,
    logit_weight_w_min: Optional[float] = None,
    logit_weight_w_max: Optional[float] = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    bounds = [list(b) for b in cfg.default_bounds(sigma_spec_min=sigma_spec_min)]
    names = cfg.free_param_names()
    idx = {n: i for i, n in enumerate(names)}
    msgs: list[str] = []

    def tighten(n: str, lo: Optional[float] = None, hi: Optional[float] = None) -> None:
        if n not in idx:
            return
        i = idx[n]
        a, b = bounds[i]
        if lo is not None:
            na = max(a, float(lo))
            if na > b:
                raise ValueError(f"bound override for {n}: new lower {na} exceeds upper {b}")
            a = na
        if hi is not None:
            nb = min(b, float(hi))
            if a > nb:
                raise ValueError(f"bound override for {n}: lower {a} exceeds new upper {nb}")
            b = nb
        if (a, b) != tuple(bounds[i]):
            msgs.append(f"{n}: ({bounds[i][0]:.6g},{bounds[i][1]:.6g}) -> ({a:.6g},{b:.6g})")
        bounds[i] = [a, b]

    if log_metric_t_min is not None:
        tighten("log_metric_t", lo=log_metric_t_min)
    if log_metric_w_min is not None:
        tighten("log_metric_w", lo=log_metric_w_min)
    if log_metric_t2_max is not None:
        tighten("log_metric_t2", hi=log_metric_t2_max)
    if log_metric_w2_min is not None:
        tighten("log_metric_w2", lo=log_metric_w2_min)
    if log_metric_w2_max is not None:
        tighten("log_metric_w2", hi=log_metric_w2_max)
    if logit_weight_t_min is not None:
        tighten("logit_weight_t", lo=logit_weight_t_min)
    if logit_weight_t_max is not None:
        tighten("logit_weight_t", hi=logit_weight_t_max)
    if logit_weight_w_min is not None:
        tighten("logit_weight_w", lo=logit_weight_w_min)
    if logit_weight_w_max is not None:
        tighten("logit_weight_w", hi=logit_weight_w_max)

    return [(float(a), float(b)) for a, b in bounds], msgs


def _resolve_sigma_spec_min(
    cfg: Mapping[str, Any],
    yerr: np.ndarray,
    point_class: np.ndarray,
) -> Optional[float]:
    explicit = cfg.get("sigma_spec_min")
    if explicit is not None:
        return max(float(explicit), 1e-6)
    frac = cfg.get("sigma_spec_adaptive_frac")
    if frac is None:
        return None
    spec_mask = point_class == gu.SPEC
    if not np.any(spec_mask):
        return None
    median_yerr_spec = float(np.median(yerr[spec_mask]))
    return max(0.005, float(frac) * median_yerr_spec)


def _make_neg_ll(
    base_cfg: gu.KernelConfig,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    mean_model,
):
    eval_count: dict[str, Any] = {"n": 0, "best": np.inf, "best_theta": None}

    def neg_ll(theta: np.ndarray) -> float:
        cfg = gu.KernelConfig(**asdict(base_cfg))
        try:
            cfg.update_from_vector(theta)
        except Exception:
            return 1e15
        try:
            kernel = gu.build_kernel(cfg)
        except (ValueError, FloatingPointError):
            return 1e15
        gp = george.GP(kernel, mean=mean_model) if mean_model is not None else george.GP(kernel)
        sigma_phot = float(np.exp(cfg.log_sigma_phot))
        sigma_spec = float(np.exp(cfg.log_sigma_spec))
        diag = gu.compute_diagonal(yerr, point_class, sigma_phot, sigma_spec)
        try:
            gp.compute(X, diag)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return 1e15
        ll = gp.log_likelihood(y)
        eval_count["n"] += 1
        if not np.isfinite(ll):
            return 1e15
        val = -ll
        if val < eval_count["best"]:
            eval_count["best"] = val
            eval_count["best_theta"] = np.asarray(theta, dtype=float).copy()
        return float(val)

    return neg_ll, eval_count


def _has_key(data: Any, key: str) -> bool:
    if isinstance(data, Mapping):
        return key in data
    if hasattr(data, "files"):
        return key in data.files
    return hasattr(data, "__contains__") and key in data


def _getitem_arr(data: Any, key: str) -> np.ndarray:
    z = np.asarray(data[key], dtype=np.float64)
    return z


def _open_bundle_arrays(bundle: BundleInput) -> dict[str, np.ndarray]:
    if isinstance(bundle, (str, os.PathLike)):
        data = np.load(str(bundle), allow_pickle=False)
    elif hasattr(bundle, "files") and hasattr(bundle, "__getitem__"):
        data = bundle  # NpzFile
    elif isinstance(bundle, Mapping):
        data = bundle
    else:
        raise TypeError("bundle must be path-like, npz file, or mapping of arrays")

    X = _getitem_arr(data, "X")
    y = _getitem_arr(data, "y").ravel()
    yerr = _getitem_arr(data, "yerr").ravel()
    X_fill = _getitem_arr(data, "X_fill")
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2)")
    if X_fill.ndim != 2 or X_fill.shape[1] != 2:
        raise ValueError("X_fill must have shape (N, 2)")
    k_w = (
        float(np.asarray(data["kernel_wls_scale"], dtype=float).ravel()[0])
        if _has_key(data, "kernel_wls_scale")
        else float(np.exp(-4.0))
    )
    k_t = (
        float(np.asarray(data["kernel_time_scale"], dtype=float).ravel()[0])
        if _has_key(data, "kernel_time_scale")
        else float(np.exp(-4.0))
    )
    y_vs = (
        float(np.asarray(data["y_var_scale"], dtype=float).ravel()[0])
        if _has_key(data, "y_var_scale")
        else float(np.var(y))
    )

    pp = np.zeros((0, 2), dtype=np.float64)
    pv = np.zeros((0,), dtype=np.float64)
    if _has_key(data, "prior_points") and _has_key(data, "prior_values"):
        ppt = np.asarray(data["prior_points"], dtype=np.float64)
        pvt = np.asarray(data["prior_values"], dtype=np.float64).ravel()
        if ppt.size:
            pp = np.reshape(ppt, (-1, 2))
        if pvt.size:
            pv = pvt.ravel()

    out: dict[str, np.ndarray] = {
        "X": X,
        "y": y,
        "yerr": yerr,
        "X_fill": X_fill,
        "kernel_wls_scale": np.array(k_w),
        "kernel_time_scale": np.array(k_t),
        "y_var_scale": np.array(y_vs),
        "prior_points": pp,
        "prior_values": pv,
    }
    return out


def run_gp_from_bundle(
    bundle: BundleInput,
    *,
    cache_workdir: str | None = None,
    write_run_json: str | os.PathLike | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Train collaborator GP + predict on bundle ``X_fill``.

    Parameters
    ----------
    bundle :
        Minimal bundle ``.npz`` path or in-memory arrays with keys matching ``gp2dim_export``.
    cache_workdir :
        Directory holding ``prior_linear_interp.pkl`` for ``mean='linear'`` (recommended under
        run diagnostics, not repo root).

    Extra keyword arguments override ``DEFAULT_KWARGS`` (kernel names, optimization, mono/blue cuts, …).

    Returns
    -------
    dict with ``mu``, ``mu_raw``, ``std``, ``X_fill``, ``X``, ``y``, ``yerr``,
    ``config_final`` (KernelConfig-like dict), ``log_likelihood``, timing, diagnostics.
    """
    cfg = dict(DEFAULT_KWARGS)
    cfg.update(kwargs)

    bd = _open_bundle_arrays(bundle)
    X = bd["X"]
    y = bd["y"]
    yerr = bd["yerr"]
    X_fill = bd["X_fill"]

    point_class = gu.classify_points(X, threshold=int(cfg["phot_spec_threshold"]))
    prior_pts = bd["prior_points"]
    prior_val = bd["prior_values"]

    cache_dir = cache_workdir
    if cfg["mean"] == "linear" and cache_dir is None:
        cache_dir = os.getcwd()

    mean_model = gu.build_mean(
        str(cfg["mean"]),
        prior_pts=prior_pts if prior_pts.size else None,
        prior_val=prior_val if prior_val.size else None,
        cache_workdir=cache_dir if prior_pts.size and prior_val.size else cache_dir,
    )

    cfg0 = _build_initial_config(cfg, bd)
    warm_path = cfg.pop("warm_start_config_json", None)
    if warm_path:
        _apply_warm_start_config(cfg0, str(warm_path))
    t_total = time.time()

    ll_initial = ll_final_opt = None
    counter = {"n": 0}
    elapsed_opt = 0.0
    bound_msgs: list[str] = []
    sigma_spec_min = _resolve_sigma_spec_min(cfg, yerr, point_class)

    if cfg["optimize"]:
        sub_n = int(cfg["optimize_subsample"])
        if sub_n and sub_n > 0 and sub_n < X.shape[0]:
            rng = np.random.default_rng(int(cfg["seed"]))
            phot_idx = np.where(point_class == gu.PHOT)[0]
            spec_idx = np.where(point_class == gu.SPEC)[0]
            n_spec_keep = max(sub_n - phot_idx.size, 100)
            spec_keep = rng.choice(spec_idx, size=min(n_spec_keep, spec_idx.size), replace=False)
            sub_idx = np.sort(np.concatenate([phot_idx, spec_keep]))
            X_opt, y_opt, yerr_opt, cls_opt = X[sub_idx], y[sub_idx], yerr[sub_idx], point_class[sub_idx]
        else:
            X_opt, y_opt, yerr_opt, cls_opt = X, y, yerr, point_class

        neg_ll, counter = _make_neg_ll(cfg0, X_opt, y_opt, yerr_opt, cls_opt, mean_model)
        theta0 = cfg0.to_vector()
        bounds, bound_msgs = _tightened_optimizer_bounds(
            cfg0,
            sigma_spec_min=sigma_spec_min,
            log_metric_t_min=cfg.get("log_metric_t_min"),
            log_metric_w_min=cfg.get("log_metric_w_min"),
            log_metric_t2_max=cfg.get("log_metric_t2_max"),
            log_metric_w2_min=cfg.get("log_metric_w2_min"),
            log_metric_w2_max=cfg.get("log_metric_w2_max"),
            logit_weight_t_min=cfg.get("logit_weight_t_min"),
            logit_weight_t_max=cfg.get("logit_weight_t_max"),
            logit_weight_w_min=cfg.get("logit_weight_w_min"),
            logit_weight_w_max=cfg.get("logit_weight_w_max"),
        )
        ll_initial = float(-neg_ll(theta0))
        t0 = time.time()
        res = minimize(
            neg_ll,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(cfg["max_iter"]), "disp": False, "ftol": 1e-7},
        )
        elapsed_opt = time.time() - t0
        if counter["best_theta"] is not None and -counter["best"] >= -res.fun:
            cfg0.update_from_vector(counter["best_theta"])
            ll_final_opt = float(-counter["best"])
        else:
            cfg0.update_from_vector(res.x)
            ll_final_opt = float(-res.fun)
        cfg_final = cfg0
    else:
        cfg_final = cfg0
        ll_final_opt = None

    kernel = gu.build_kernel(cfg_final)
    gp = george.GP(kernel, mean=mean_model) if mean_model is not None else george.GP(kernel)
    sigma_phot = float(np.exp(cfg_final.log_sigma_phot))
    sigma_spec = float(np.exp(cfg_final.log_sigma_spec))
    diag = gu.compute_diagonal(yerr, point_class, sigma_phot, sigma_spec)

    t0 = time.time()
    gp.compute(X, diag)
    elapsed_compute = time.time() - t0
    log_lik = float(gp.log_likelihood(y))

    chunk = int(cfg["predict_chunk"])
    mu, var = _predict_chunked(gp, y, X_fill, chunk)
    n_neg = int((var < 0).sum())
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    mu_raw = mu.copy()

    n_modified_mono = n_modified_blue = 0
    if cfg["enforce_mono_early"]:
        mu, std, n_modified_mono = _enforce_monotone_early(
            X_fill,
            mu,
            std,
            float(cfg["early_time_cutoff"]),
            floor_fraction=float(cfg["mono_floor_fraction"]),
            min_slope=float(cfg["mono_min_slope"]),
            smoothing_scale=float(cfg["mono_smoothing_scale"]),
        )
    if cfg["enforce_blue_early"]:
        mu, n_modified_blue = _enforce_blue_early(X_fill, mu, float(cfg["early_time_cutoff"]))

    payload: dict[str, Any] = {
        "mu": mu,
        "mu_raw": mu_raw,
        "std": std,
        "var": var,
        "X_fill": X_fill,
        "X": X,
        "y": y,
        "yerr": yerr,
        "point_class_train": point_class,
        "sigma_eff_train": diag,
        "config_final": cfg_final.as_dict(),
        "log_likelihood": log_lik,
        "log_likelihood_initial_subsample": ll_initial,
        "log_likelihood_after_opt_subsample": ll_final_opt,
        "neg_ll_evals": int(counter.get("n", 0)),
        "optimize_seconds": float(elapsed_opt),
        "compute_seconds": float(elapsed_compute),
        "n_neg_var_clipped": n_neg,
        "n_modified_mono": n_modified_mono,
        "n_modified_blue": n_modified_blue,
        "total_runtime_seconds": time.time() - t_total,
        "effective_kwargs": dict(cfg),
        "prior_cache_dir": cache_workdir,
        "sigma_spec_min_applied": sigma_spec_min,
        "optimizer_bound_overrides": bound_msgs if cfg["optimize"] else [],
    }

    if cfg.get("predict_train"):
        mu_tr, var_tr = _predict_chunked(gp, y, X, chunk)
        var_tr = np.maximum(var_tr, 0.0)
        payload["mu_train"] = mu_tr
        payload["std_train"] = np.sqrt(var_tr)

    if write_run_json:
        rp = {}
        rp.update({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in payload.items() if k in (
            "log_likelihood", "optimize_seconds", "compute_seconds",
            "n_modified_mono", "n_modified_blue", "total_runtime_seconds",
        )})
        rp["config_final"] = payload["config_final"]
        rp["effective_kwargs"] = cfg
        with open(write_run_json, "w", encoding="utf-8") as fh:
            json.dump(rp, fh, indent=2, default=lambda o: "<ndarray>" if isinstance(o, np.ndarray) else str(o))

    return payload


def main_cli(argv: Optional[list[str]] = None) -> int:
    """Minimal CLI: ``python -m twodim_gp.run_inference --input bundle.npz``."""
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    default_bundle = os.path.normpath(os.path.join(here, "..", "..", "gp_rjf", "gp_minimal_bundle.npz"))

    p = argparse.ArgumentParser(description="Collaborator GP (vendored) minimal CLI")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default=os.path.join(here, "runs_cli"))
    p.add_argument("--tag", default="run0")
    p.add_argument("--no-optimize", action="store_true")
    ns = p.parse_args(argv)

    run_dir = os.path.join(ns.output_dir, ns.tag)
    os.makedirs(run_dir, exist_ok=True)
    merged = run_gp_from_bundle(
        ns.input,
        cache_workdir=os.path.join(run_dir, "prior_cache"),
        optimize=not ns.no_optimize,
    )

    pred_path = os.path.join(run_dir, "predictions.npz")
    np.savez_compressed(
        pred_path,
        X_fill=merged["X_fill"],
        mu=merged["mu"],
        std=merged["std"],
        mu_raw=merged["mu_raw"],
    )
    print("[twodim_gp] wrote", pred_path)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
