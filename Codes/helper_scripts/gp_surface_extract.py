"""Extract GP surface spectra at observed spectroscopy epochs (Phase 3)."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np
from scipy import interpolate

import GP2dim_utils as _g
from mangle_spectra_log import demangle_log_spectrum


def grid_norm_uses_zscore(grid_norm_info: Mapping[str, Any]) -> bool:
    return _g._coord_mode(grid_norm_info) == "zscore"


def interpolate_mangling_mask_to_wls(
    mask_log: np.ndarray,
    wls_src_linear: np.ndarray,
    wls_dst_linear: np.ndarray,
) -> np.ndarray:
    """Interpolate additive log-space mangling mask between linear Angstrom grids."""
    m = np.asarray(mask_log, dtype=float).ravel()
    src = np.asarray(wls_src_linear, dtype=float).ravel()
    dst = np.asarray(wls_dst_linear, dtype=float).ravel()
    if m.size != src.size:
        raise ValueError("mask_log and wls_src_linear must have the same length")
    if m.size < 1 or dst.size < 1:
        return np.zeros_like(dst, dtype=float)
    order = np.argsort(src)
    src_s = src[order]
    m_s = m[order]
    log_src = np.log10(np.clip(src_s, 1e-30, None))
    log_dst = np.log10(np.clip(dst, 1e-30, None))
    return np.interp(log_dst, log_src, m_s)


def gp_prediction_wls_linear(wls_log_grid: np.ndarray) -> np.ndarray:
    """Linear Angstrom grid from ``predictions.npz`` ``wls_log_grid`` (log10 lambda)."""
    return np.power(10.0, np.asarray(wls_log_grid, dtype=float))


def _x2_mask_for_phase(
    x2_fill: np.ndarray, phase_log10_days: float, grid_norm_info: Mapping[str, Any]
) -> np.ndarray:
    return _g.x2_mask_for_phase(x2_fill, phase_log10_days, grid_norm_info)


def scaled_mu_to_log10_flux(
    scaled_mu: np.ndarray, grid_norm_info: Mapping[str, Any]
) -> np.ndarray:
    """Map GP scaled ln(flux) predictions to log10(Fλ)."""
    offset = float(grid_norm_info["offset"])
    scale_factor = float(grid_norm_info["scale_factor"])
    lin = _g.scaled_ln_to_linear(scaled_mu, offset, scale_factor)
    return np.log10(np.clip(lin, 1e-30, None))


def _log10_wls_from_x1_norm(
    x1_norm: np.ndarray, grid_norm_info: Mapping[str, Any]
) -> np.ndarray:
    return _g.log10_wavelength_from_x1_norm(x1_norm, grid_norm_info)


def extract_gp_spectrum_at_epoch(
    mu_fill: np.ndarray,
    std_fill: np.ndarray,
    x1_fill: np.ndarray,
    x2_fill: np.ndarray,
    grid_norm_info: Mapping[str, Any],
    phase_log10_days: float,
    target_wls_linear_angstrom: np.ndarray,
    *,
    wls_log_grid: np.ndarray,
    phase_log10_columns: np.ndarray,
    mu_key: str = "mu",
) -> dict[str, np.ndarray]:
    """Slice GP μ at one log10(phase-day) column and interpolate onto target λ grid."""
    mu_arr = np.asarray(mu_fill, dtype=float).ravel()
    std_arr = np.asarray(std_fill, dtype=float).ravel()
    x1 = np.asarray(x1_fill, dtype=float).ravel()
    x2 = np.asarray(x2_fill, dtype=float).ravel()
    if mu_arr.size != x1.size or mu_arr.size != x2.size:
        raise ValueError("mu_fill / x1_fill / x2_fill length mismatch")

    col_mask = _x2_mask_for_phase(x2, float(phase_log10_days), grid_norm_info)
    if not np.any(col_mask):
        raise ValueError(
            "No X_fill rows match phase_log10_days=%.6f" % float(phase_log10_days)
        )

    x1_col = x1[col_mask]
    mu_col = mu_arr[col_mask]
    std_col = std_arr[col_mask]
    log10_wls = _log10_wls_from_x1_norm(x1_col, grid_norm_info)
    log10_flux = scaled_mu_to_log10_flux(mu_col, grid_norm_info)
    log10_std = np.abs(
        scaled_mu_to_log10_flux(mu_col + std_col, grid_norm_info) - log10_flux
    )

    order = np.argsort(log10_wls)
    log10_wls = log10_wls[order]
    log10_flux = log10_flux[order]
    log10_std = log10_std[order]

    target_log_wls = np.log10(np.clip(np.asarray(target_wls_linear_angstrom, dtype=float), 1e-30, None))
    f_flux = interpolate.interp1d(
        log10_wls,
        log10_flux,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    f_std = interpolate.interp1d(
        log10_wls,
        log10_std,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    out_flux = f_flux(target_log_wls)
    out_std = f_std(target_log_wls)
    return {
        "log10_wls": target_log_wls,
        "log10_flux": out_flux,
        "log10_fluxerr": out_std,
        "phase_log10_days": np.full(target_log_wls.shape, float(phase_log10_days)),
        "target_wls_linear": np.asarray(target_wls_linear_angstrom, dtype=float),
    }


def demangle_extracted_spectrum(
    log10_flux: np.ndarray, mangling_mask_log: np.ndarray
) -> np.ndarray:
    return demangle_log_spectrum(log10_flux, mangling_mask_log)


def extract_all_observed_epochs(
    predictions: Mapping[str, Any],
    *,
    spec_entries: list[Any],
    t0_fix: float,
    prescaled_wls_by_basename: Mapping[str, np.ndarray],
    observed_phase_log10: Optional[set[float]] = None,
    mu_key: str = "mu",
    wavelength_grid: Literal["prescaled", "gp"] = "prescaled",
) -> list[dict[str, Any]]:
    """Extract GP spectra for prescaled-list epochs only."""
    mu = np.asarray(predictions[mu_key if mu_key in predictions else "mu"], dtype=float)
    std = np.asarray(predictions["std"], dtype=float)
    x1 = np.asarray(predictions["x1_fill"], dtype=float)
    x2 = np.asarray(predictions["x2_fill"], dtype=float)
    gn = predictions["grid_norm_info"]
    if isinstance(gn, np.ndarray) and gn.dtype == object:
        gn = gn.item()
    wls_log_grid = np.asarray(predictions["wls_log_grid"], dtype=float)
    phase_cols = np.asarray(predictions["phase_log10_columns"], dtype=float)
    gp_wls_linear = gp_prediction_wls_linear(wls_log_grid)

    out: list[dict[str, Any]] = []
    for entry in spec_entries:
        bn = entry.basename
        if bn not in prescaled_wls_by_basename:
            continue
        phase = max(float(entry.mjd) - float(t0_fix), 1e-5)
        phase_log = float(np.log10(phase))
        if observed_phase_log10 is not None:
            if not any(np.isclose(phase_log, p, rtol=0.0, atol=1e-6) for p in observed_phase_log10):
                continue
        if wavelength_grid == "gp":
            target_wls = gp_wls_linear
        else:
            target_wls = prescaled_wls_by_basename[bn]
        extracted = extract_gp_spectrum_at_epoch(
            mu,
            std,
            x1,
            x2,
            gn,
            phase_log,
            target_wls,
            wls_log_grid=wls_log_grid,
            phase_log10_columns=phase_cols,
            mu_key=mu_key,
        )
        out.append(
            {
                "basename": bn,
                "mjd": float(entry.mjd),
                "phase_log10_days": phase_log,
                "extracted": extracted,
                "wavelength_grid": wavelength_grid,
            }
        )
    return out
