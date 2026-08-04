"""Optional t≈0 anchor rows for ``fitted_phot_logspace_*.dat`` (not real photometry).

Use when you want the per-band GP in notebook 4 to pull toward zero flux at explosion.
Mangling / raw MJD-range logic must **not** use these rows; keep using
``photometry_filter_utils`` on ``1_LCs_flux_raw`` only.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

# Defaults shared with ``append_explosion_anchor_row`` and pre-``gp.compute`` LC training augmentation.
DEFAULT_T0_ANCHOR_LOG_PHASE: float = -8.0
DEFAULT_T0_ANCHOR_LOG_FLUX_CAP: float = -50.0
DEFAULT_T0_ANCHOR_LOG_FLUX_ERR: float = 2.0

__all__ = [
    "append_explosion_anchor_row",
    "augment_lc_gp_training_for_t0_anchor",
    "DEFAULT_T0_ANCHOR_LOG_PHASE",
    "DEFAULT_T0_ANCHOR_LOG_FLUX_CAP",
    "DEFAULT_T0_ANCHOR_LOG_FLUX_ERR",
]


def augment_lc_gp_training_for_t0_anchor(
    log_phase,
    orig_log_flux,
    orig_log_flux_err,
    sudo_mask,
    median_log_flux,
    flux_norm,
    err_flux_norm,
    *,
    log_phase_anchor: float = DEFAULT_T0_ANCHOR_LOG_PHASE,
    log_flux_cap: float = DEFAULT_T0_ANCHOR_LOG_FLUX_CAP,
    log_flux_err: float = DEFAULT_T0_ANCHOR_LOG_FLUX_ERR,
):
    """Append one synthetic training point before ``george.GP.compute`` (not real photometry).

    ``flux_norm`` / ``err_flux_norm`` are the arrays passed to ``gp`` after subtracting
    ``median_log_flux`` from log flux. The new point uses ``log_flux_cap`` in log-flux space and
    is marked ``True`` in the boolean ``sudo_mask`` output (extend the mask by one element).

    Returns updated arrays (numpy) in the same order as the first six parameters.
    """
    log_phase = np.asarray(log_phase, dtype=float)
    orig_log_flux = np.asarray(orig_log_flux, dtype=float)
    orig_log_flux_err = np.asarray(orig_log_flux_err, dtype=float)
    sudo_mask = np.asarray(sudo_mask, dtype=bool)
    flux_norm = np.asarray(flux_norm, dtype=float)
    err_flux_norm = np.asarray(err_flux_norm, dtype=float)
    cap = float(log_flux_cap)
    log_phase_ext = np.append(log_phase, float(log_phase_anchor))
    orig_ext = np.append(orig_log_flux, cap)
    orig_err_ext = np.append(orig_log_flux_err, float(log_flux_err))
    sudo_ext = np.append(sudo_mask, True)
    fn_ext = np.append(flux_norm, cap - float(median_log_flux))
    err_ext = np.append(err_flux_norm, float(log_flux_err))
    return log_phase_ext, orig_ext, orig_err_ext, sudo_ext, fn_ext, err_ext


def append_explosion_anchor_row(
    path_in: str,
    path_out: str | None = None,
    *,
    log_phase: float = DEFAULT_T0_ANCHOR_LOG_PHASE,
    log_flux_cap: float = DEFAULT_T0_ANCHOR_LOG_FLUX_CAP,
    log_flux_err: float = DEFAULT_T0_ANCHOR_LOG_FLUX_ERR,
    only_bands_with_data: bool = True,
    backup_suffix: str = ".before_explosion_anchor",
) -> pd.DataFrame:
    """Append one row at ``Log_Phase=log_phase`` with capped log flux for LC GP training.

    If ``only_bands_with_data``, only *_log_flux columns that have ≥1 finite value in the table
    are set; others remain NaN.
    """
    if path_out is None:
        path_out = path_in
    df = pd.read_csv(path_in, sep="\t")
    if "Log_Phase" not in df.columns:
        raise ValueError("Expected Log_Phase column in %s" % path_in)

    flux_cols = [c for c in df.columns if c.endswith("_log_flux") and not c.endswith("_log_flux_err")]
    row = dict.fromkeys(df.columns, np.nan)
    row["Log_Phase"] = float(log_phase)

    for c in flux_cols:
        if only_bands_with_data:
            if not np.any(np.isfinite(pd.to_numeric(df[c], errors="coerce").values)):
                continue
        row[c] = float(log_flux_cap)
        err_c = c.replace("_log_flux", "_log_flux_err")
        if err_c in df.columns:
            row[err_c] = float(log_flux_err)

    out_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    if os.path.abspath(path_out) == os.path.abspath(path_in):
        bak = path_in + backup_suffix
        shutil.copy2(path_in, bak)
    out_df.to_csv(path_out, sep="\t", index=False)
    return out_df
