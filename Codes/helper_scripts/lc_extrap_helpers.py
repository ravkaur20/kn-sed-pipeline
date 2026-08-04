"""
Shared helpers for LC early-time Bazin extrapolation.
Imported by 2_LC_modelRising_KN_fullfit_log.ipynb (sys.path must include helper_scripts/).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "EXTRA_ERR_ABS_CAP_DEFAULT",
    "EXTRA_ERR_REL_DATAERR_MAX_DEFAULT",
    "EXTRA_COV_CONDITION_MAX_DEFAULT",
    "clip_extrap_uncertainties",
    "covariance_is_bad",
    "decode_band",
    "normalize_bands_keep",
    "pick_reference_band",
    "filters_within_explosion_window",
    "early_bands_stage_a",
    "validate_bands_keep",
    "phase_reference_mjd",
    "bazin_forced_zero_t0_bounds_guess",
]


EXTRA_ERR_ABS_CAP_DEFAULT = 0.5  # max σ in normalized-flux units on synthetic early points
EXTRA_ERR_REL_DATAERR_MAX_DEFAULT = None  # e.g. 8.0 → cap at 8 * median(fluxerr_)
EXTRA_COV_CONDITION_MAX_DEFAULT = 5e13


def decode_band(b) -> str:
    """Normalize band name from bytes or str."""
    return b.decode() if isinstance(b, bytes) else str(b)


def _unique_bands(phot) -> np.ndarray:
    """Unique band names from pandas DataFrame or numpy structured photometry array."""
    bands = phot["band"]
    if hasattr(bands, "unique"):
        return bands.unique()
    return np.unique(bands)


def pick_reference_band(
    avail_filters,
    exclude: set[str],
    candidates: Sequence[str],
) -> str | None:
    """Return first reference candidate present in data and not excluded."""
    avail = {decode_band(f) for f in avail_filters}
    for cand in candidates:
        if cand not in exclude and cand in avail:
            return cand
    return None


def filters_within_explosion_window(
    phot,
    explosion_mjd: float,
    window_days: float,
    exclude: set[str],
) -> tuple[list[str], list[str]]:
    """Split bands by whether any photometry falls within [explosion, explosion+window]."""
    within: list[str] = []
    outside: list[str] = []
    bands = _unique_bands(phot)
    for filt in bands:
        b = decode_band(filt)
        if b in exclude:
            continue
        mask = phot["band"] == filt
        mjds = np.asarray(phot["MJD"][mask], dtype=float)
        if np.any((mjds >= explosion_mjd) & (mjds <= explosion_mjd + window_days)):
            within.append(b)
        else:
            outside.append(b)
    return within, outside


def early_bands_stage_a(
    filters_within: Sequence[str],
    exclude: set[str],
    include: Sequence[str],
    reference_band: str | None,
    se_sne: Sequence[str],
    snname: str,
) -> list[str]:
    """Build Stage-A candidate band list (explosion window + include_dict, minus exclude)."""
    ref = decode_band(reference_band) if reference_band else None
    seen: set[str] = set()
    out: list[str] = []

    for filt in filters_within:
        b = decode_band(filt)
        if b in exclude or b in seen:
            continue
        if ref is not None and b == ref:
            continue
        if snname in se_sne and b in ("swift_UVW1", "swift_UVW2", "swift_UVM2"):
            continue
        if snname not in se_sne and b in (
            "swift_U",
            "Bessell_U",
            "swift_UVW1",
            "swift_UVW2",
            "swift_UVM2",
        ):
            continue
        out.append(b)
        seen.add(b)

    for b in include:
        bs = decode_band(b)
        if bs not in seen and bs not in exclude and (ref is None or bs != ref):
            out.append(bs)
            seen.add(bs)

    return out


def normalize_bands_keep(keep) -> list[str]:
    """Flatten nested band lists and dedupe while preserving order."""
    out: list[str] = []
    seen: set[str] = set()

    def _walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                _walk(sub)
            return
        b = decode_band(item)
        if b not in seen:
            out.append(b)
            seen.add(b)

    _walk(keep)
    return out


def validate_bands_keep(keep: Sequence[str], preview_cache_keys) -> None:
    """Raise if any kept band was not previewed."""
    keep_norm = normalize_bands_keep(keep)
    keys = {decode_band(k) for k in preview_cache_keys}
    missing = set(keep_norm) - keys
    if missing:
        raise ValueError(f"Bands not in preview cache: {sorted(missing)}")


def phase_reference_mjd(
    snname: str,
    reference_band_data: dict | None,
    explosion_mjd: float,
    pre_bump: dict,
) -> float:
    """Phase anchor MJD: reference-band peak if fit, else explosion MJD (with pre_bump offset)."""
    if reference_band_data is not None:
        return float(reference_band_data["mjd_xpeak"])
    if snname in pre_bump:
        return float(explosion_mjd + pre_bump[snname][0])
    return float(explosion_mjd)


def bazin_forced_zero_t0_bounds_guess(
    t_,
    flux_,
    *,
    margin_min: float = 5.0,
    margin_frac: float = 0.05,
) -> tuple[float, float, float]:
    """Return (t0_lo, t0_hi, p_t0) for forced-zero Bazin curve_fit bounds.

    When explosion MJD is fixed, Bazin t0 is bounded by the band's own time span
    (not mjd_phase_ref), so per-band peaks after explosion remain valid initial guesses.
    """
    t = np.asarray(t_, dtype=float)
    flux = np.asarray(flux_, dtype=float)
    if t.size == 0:
        return 0.0, 1.0, 0.5
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    margin = max(float(margin_min), float(margin_frac) * (t_max - t_min + 1.0))
    t0_lo = t_min - margin
    t0_hi = t_max + margin
    peak_t = float(t[int(np.argmax(flux))]) if flux.size else t_min
    p_t0 = float(np.clip(peak_t, t0_lo, t0_hi))
    return t0_lo, t0_hi, p_t0


def clip_extrap_uncertainties(
    newpts_err: np.ndarray,
    flux_: np.ndarray,
    fluxerr_: np.ndarray,
    abs_cap: float | None = EXTRA_ERR_ABS_CAP_DEFAULT,
    rel_med_max: float | None = EXTRA_ERR_REL_DATAERR_MAX_DEFAULT,
) -> np.ndarray:
    """Clip positive extrapolation sigmas to avoid huge plot noise from unstable MC."""
    ee = np.asarray(newpts_err, dtype=float).copy()
    ee = np.where(np.isfinite(ee), ee, np.nan)
    ee = np.nan_to_num(ee, nan=abs_cap if abs_cap is not None else 1.0, posinf=abs_cap or 1.0, neginf=0.0)
    ee = np.clip(ee, 0.0, np.inf)
    if abs_cap is not None:
        ee = np.minimum(ee, float(abs_cap))
    if rel_med_max is not None and fluxerr_ is not None and len(fluxerr_) > 0:
        med = float(np.nanmedian(np.asarray(fluxerr_, dtype=float)))
        if np.isfinite(med) and med > 0:
            ee = np.minimum(ee, float(rel_med_max) * med)
    return ee


def covariance_is_bad(
    cov: np.ndarray,
    cond_max: float = EXTRA_COV_CONDITION_MAX_DEFAULT,
) -> bool:
    """True if pcov is unusable for multivariate_normal sampling."""
    c = np.asarray(cov, dtype=float)
    if c.size == 0 or np.any(np.isinf(c)) or np.any(np.isnan(c)):
        return True
    d = np.diag(c)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return True
    try:
        cond = float(np.linalg.cond(c))
    except np.linalg.LinAlgError:
        return True
    return (not np.isfinite(cond)) or (cond > cond_max)
