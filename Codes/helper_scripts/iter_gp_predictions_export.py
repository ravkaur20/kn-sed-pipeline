"""Save pipeline-compatible ``predictions.npz`` + ``config.json`` for iter GP runs."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np


def save_iter_predictions_npz(
    path: str,
    *,
    mu: np.ndarray,
    mu_raw: np.ndarray,
    std: np.ndarray,
    x1_fill: np.ndarray,
    x2_fill: np.ndarray,
    grid_norm_info: Mapping[str, Any],
    merged: Mapping[str, Any] | None = None,
    wls_log_grid: np.ndarray | None = None,
    phase_log10_columns: np.ndarray | None = None,
) -> None:
    """Write ``predictions.npz`` with fields required by ``twodim_gp/plot_results.py``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x1 = np.asarray(x1_fill, dtype=float).ravel()
    x2 = np.asarray(x2_fill, dtype=float).ravel()
    mu_a = np.asarray(mu, dtype=float).ravel()
    if merged is not None and "X_fill" in merged:
        x_fill = np.asarray(merged["X_fill"], dtype=float)
    else:
        x_fill = np.column_stack([x1, x2])

    payload: dict[str, Any] = {
        "mu": mu_a,
        "mu_raw": np.asarray(mu_raw, dtype=float).ravel(),
        "std": np.asarray(std, dtype=float).ravel(),
        "X_fill": x_fill,
        "x1_fill": x1,
        "x2_fill": x2,
        "grid_norm_info": json.dumps(dict(grid_norm_info)),
    }
    if wls_log_grid is not None:
        payload["wls_log_grid"] = np.asarray(wls_log_grid, dtype=float)
    if phase_log10_columns is not None:
        payload["phase_log10_columns"] = np.asarray(phase_log10_columns, dtype=float)

    if merged:
        for key in (
            "point_class_train",
            "sigma_eff_train",
            "mu_train",
            "std_train",
            "log_likelihood",
        ):
            if key in merged and merged[key] is not None:
                payload[key] = np.asarray(merged[key])

    np.savez_compressed(path, **payload)


def save_iter_gp_config_json(
    path: str,
    *,
    merged: Mapping[str, Any] | None,
    grid_norm_info: Mapping[str, Any],
) -> None:
    """Sibling ``config.json`` for ``plot_results`` suptitles."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg_final = dict(merged.get("config_final", {})) if merged else {}
    body: dict[str, Any] = {
        "config": cfg_final,
        "grid_norm_info": dict(grid_norm_info),
    }
    if merged:
        for key in ("log_likelihood", "total_runtime_seconds", "optimize_seconds"):
            if key in merged:
                body[key] = merged[key]
        if "log_likelihood" in merged:
            body["log_likelihood_at_compute"] = merged["log_likelihood"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2)
