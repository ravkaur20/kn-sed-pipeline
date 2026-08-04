"""Build photometry enrich npz for ``plot_bands_gp_overview --enrich``."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np


def _denorm_log_phase(x2_norm: np.ndarray, gn: Mapping[str, Any]) -> np.ndarray:
    x2 = np.asarray(x2_norm, dtype=float).ravel()
    if gn.get("coord_parametrization") == "zscore":
        return np.asarray(gn["x2_mean"], dtype=float) + np.asarray(gn["x2_std"], dtype=float) * x2
    return np.asarray(gn["offset2"], dtype=float) + np.asarray(gn["norm2"], dtype=float) * x2


def _denorm_log_wl(x1_norm: np.ndarray, gn: Mapping[str, Any]) -> np.ndarray:
    x1 = np.asarray(x1_norm, dtype=float).ravel()
    if gn.get("coord_parametrization") == "zscore":
        return np.asarray(gn["x1_mean"], dtype=float) + np.asarray(gn["x1_std"], dtype=float) * x1
    return np.asarray(x1, dtype=float) * float(gn["norm1"])


def build_phot_enrich_npz(
    bundle_path: str,
    out_path: str,
    *,
    phot4mangling_path: str,
    grid_norm_info: Mapping[str, Any],
    t0_fix: float,
    avail_filters: Sequence[str],
    filter_eff_log_wl: Mapping[str, float] | None = None,
    phot_spec_threshold: int = 50,
) -> str:
    """Align ``band_name`` / ``mjd`` per bundle training row for photometry panels."""
    from twodim_gp import gp_utils as gu

    bundle = np.load(bundle_path, allow_pickle=False)
    x = np.asarray(bundle["X"], dtype=float)
    n = x.shape[0]
    point_class = gu.classify_points(x, threshold=int(phot_spec_threshold))
    phot_mask = point_class == gu.PHOT

    import pandas as pd

    phot_df = pd.read_csv(phot4mangling_path, sep="\t")
    mjd_col = "MJD" if "MJD" in phot_df.columns else "spec_mjd"
    mjds = np.asarray(phot_df[mjd_col], dtype=float)

    eff_wl: dict[str, float] = dict(filter_eff_log_wl or {})
    if not eff_wl:
        for filt in avail_filters:
            col = "%s_fit_log_flux" % filt
            if col not in phot_df.columns:
                continue
            eff_wl[str(filt)] = float("nan")

    band_name = np.full(n, "", dtype=object)
    mjd_arr = np.full(n, np.nan, dtype=float)

    log_ph = _denorm_log_phase(x[:, 1], grid_norm_info)
    log_wl = _denorm_log_wl(x[:, 0], grid_norm_info)
    phase_days = np.power(10.0, np.clip(log_ph, -50, 50))
    spec_mjd = float(t0_fix) + phase_days

    phot_rows = np.nonzero(phot_mask)[0]
    for i in phot_rows:
        sm = float(spec_mjd[i])
        pos = int(np.argmin(np.abs(mjds - sm)))
        row = phot_df.iloc[pos]
        mjd_arr[i] = float(row[mjd_col])
        best_filt = None
        best_d = np.inf
        for filt in avail_filters:
            col = "%s_fit_log_flux" % filt
            if col not in phot_df.columns:
                continue
            val = row[col]
            if not np.isfinite(val):
                continue
            ew = eff_wl.get(filt)
            if ew is None or not np.isfinite(ew):
                best_filt = filt
                break
            d = abs(float(log_wl[i]) - float(ew))
            if d < best_d:
                best_d = d
                best_filt = filt
        if best_filt is None and avail_filters:
            for filt in avail_filters:
                col = "%s_fit_log_flux" % filt
                if col in phot_df.columns and np.isfinite(row[col]):
                    best_filt = filt
                    break
        band_name[i] = str(best_filt or "")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, band_name=band_name, mjd=mjd_arr)
    return out_path
