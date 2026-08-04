"""Log-space spectrum mangling helpers extracted from ``5_Mangle_spectra_KN_log.ipynb``.

Phase 1 provides I/O, mask apply/demangle, and band-flux / mangling-mask fitting for reuse
in notebook 5 and the future iterative GP+mangle driver.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SPEC_DTYPE = np.dtype([("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")])
LOG_SPEC_DTYPE = SPEC_DTYPE  # mangled on disk: log10(wls), log10(flux)


def err_to_log10(flux: np.ndarray, err_flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    err_flux = np.asarray(err_flux, dtype=float)
    return err_flux / (flux * np.log(10.0))


def err_from_log10(logflux: np.ndarray, logerr_flux: np.ndarray) -> np.ndarray:
    return np.log(10.0) * np.power(10.0, logflux) * logerr_flux


def calc_lam_avg(wls: np.ndarray, transmission: np.ndarray) -> float:
    from scipy import integrate

    return float(
        integrate.trapezoid(transmission * wls, wls)
        / integrate.trapezoid(transmission, wls)
    )


def calc_lam_eff(wls: np.ndarray, transmission: np.ndarray, flux: np.ndarray) -> float:
    from scipy import integrate

    return float(
        integrate.trapezoid(transmission * flux * wls, wls)
        / integrate.trapezoid(transmission * flux, wls)
    )


def load_linear_spectrum(path: str) -> np.ndarray:
    """Load ``wls, flux, fluxerr`` (linear Å and linear Fλ); drop NaN/nonpositive flux."""
    raw = np.genfromtxt(
        path, dtype=None, encoding="utf-8", names=["wls", "flux", "fluxerr"]
    )
    mask = (
        np.isfinite(raw["wls"])
        & np.isfinite(raw["flux"])
        & np.isfinite(raw["fluxerr"])
        & (raw["flux"] > 0.0)
    )
    return raw[mask]


def load_mangled_spectrum(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load mangled file (3 or 4 columns). Returns ``(log_spec, mangling_mask_or_None)``."""
    try:
        raw = np.genfromtxt(
            path,
            dtype=None,
            encoding="utf-8",
            names=["wls", "flux", "fluxerr", "mangling_mask"],
        )
        mask = raw["mangling_mask"]
    except (ValueError, OSError):
        raw = np.genfromtxt(
            path, dtype=None, encoding="utf-8", names=["wls", "flux", "fluxerr"]
        )
        mask = None
    log_spec = np.array(
        list(zip(raw["wls"], raw["flux"], raw["fluxerr"])), dtype=LOG_SPEC_DTYPE
    )
    if mask is not None and np.all(np.isfinite(mask)):
        return log_spec, np.asarray(mask, dtype=float)
    return log_spec, None


def save_mangled_spectrum(
    path: str,
    wls_linear: np.ndarray,
    log_flux: np.ndarray,
    log_flux_err: np.ndarray,
    mangling_mask: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("#wls\tflux\tfluxerr\tmangling_mask\n")
        for w, f, fe, mm in zip(wls_linear, log_flux, log_flux_err, mangling_mask):
            fout.write("%E\t%E\t%E\t%E\n" % (w, f, fe, mm))


def apply_mangling_mask_linear(
    wls: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    mangling_mask_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply additive log-space mask; return linear wls, log10 flux, log10 err."""
    log_flux = np.log10(np.clip(flux, 1e-30, None)) + np.asarray(mangling_mask_log, dtype=float)
    log_err = err_to_log10(flux, fluxerr)  # approximate unchanged shape in log
    return np.asarray(wls, dtype=float), log_flux, log_err


def demangle_log_spectrum(
    log_flux: np.ndarray, mangling_mask_log: np.ndarray
) -> np.ndarray:
    return np.asarray(log_flux, dtype=float) - np.asarray(mangling_mask_log, dtype=float)


def _resolve_filter_path(filter_path: str, filter_name: str, snname: str, csp_sne: Sequence[str]) -> str:
    """Resolve filter throughput file under ``Inputs/Filters/`` (parent dir)."""
    base = os.path.normpath(filter_path.rstrip(os.sep))
    if os.path.basename(base) == "GeneralFilters":
        base = os.path.dirname(base)
    band = str(filter_name).strip()
    if "swift" in band.lower():
        for sub in ("Swift", "SWIFT"):
            p = os.path.join(base, sub, "%s.dat" % band)
            if os.path.isfile(p):
                return p
        return os.path.join(base, "Swift", "%s.dat" % band)
    if snname in csp_sne:
        p = os.path.join(base, "Site3_CSP", "%s.txt" % band)
        if os.path.isfile(p):
            return p
        return p
    cands = [
        os.path.join(base, "GeneralFilters", "%s.dat" % band),
        os.path.join(base, "%s.dat" % band),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return cands[0]


def band_flux_trapz(
    spec_wls: np.ndarray,
    spec_flux: np.ndarray,
    spec_fluxerr: np.ndarray,
    filter_name: str,
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
) -> tuple[float, float, float, float, float, float]:
    """Trapezoid synthetic photometry (linear wls / linear flux)."""
    from scipy import integrate, interpolate

    filt_path = _resolve_filter_path(filter_path, filter_name, snname, csp_sne)
    filt = np.genfromtxt(
        filt_path, dtype=None, encoding="utf-8", names=["wls", "flux"]
    )
    min_wls = float(np.min(filt["wls"]))
    max_wls = float(np.max(filt["wls"]))
    lam_avg = calc_lam_avg(filt["wls"], filt["flux"])

    cut = (spec_wls > min_wls) & (spec_wls < max_wls)
    wls_c = np.asarray(spec_wls, dtype=float)[cut]
    flux_c = np.asarray(spec_flux, dtype=float)[cut]
    ferr_c = np.asarray(spec_fluxerr, dtype=float)[cut]

    filt_interp = interpolate.interp1d(filt["wls"], filt["flux"], kind="linear")(wls_c)
    filt_xlam = filt_interp * wls_c
    lam_eff = calc_lam_eff(wls_c, filt_interp, flux_c)
    denom = integrate.trapezoid(filt_xlam, wls_c)
    raw_phot = integrate.trapezoid(filt_xlam * flux_c, wls_c) / denom
    raw_phot_err = (
        integrate.trapezoid((filt_xlam * ferr_c) ** 2, wls_c) ** 0.5 / denom
    )
    return lam_avg, lam_eff, float(raw_phot), float(raw_phot_err), min_wls, max_wls


def _mangle_gp_kernel_length(norm_wls: float, kernel_divide: int) -> float:
    """Matern32 length scale in normalized log10-wavelength coordinates."""
    import pipeline_config as pconf

    mode = str(getattr(pconf, "MANGLE_GP_KERNEL_MODE", "fixed_5")).strip().lower()
    if mode == "kernel_divide_scaled":
        return float(kernel_divide) / max(abs(float(norm_wls)), 1e-6)
    return float(getattr(pconf, "MANGLE_GP_KERNEL_FIXED", 5.0))


def compute_mangling_mask(
    raw_spec: np.ndarray,
    phot4mangling_row: Mapping[str, Any],
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
    kernel_divide: int = 800,
    min_gp_metric: float = 0.09,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    """Fit wavelength-dependent log-space mangling mask (NB5 logic).

    Returns ``(mangling_mask, log_mangled_flux, log_mangled_err, meta)`` or ``None``.
    """
    from scipy import optimize

    try:
        import george
        from george.kernels import Matern32Kernel
    except ImportError as exc:
        raise ImportError("george is required for compute_mangling_mask") from exc

    if george is None or Matern32Kernel is None:
        raise ImportError("george is required for compute_mangling_mask")

    spec_mjd = float(phot4mangling_row["spec_mjd"])
    log_diffs: list[float] = []
    log_diffs_err: list[float] = []
    log_wls_eff: list[float] = []
    used_filters: list[str] = []

    wls = np.asarray(raw_spec["wls"], dtype=float)
    flux = np.asarray(raw_spec["flux"], dtype=float)
    fluxerr = np.asarray(raw_spec["fluxerr"], dtype=float)

    for filt in avail_filters:
        if filt not in filter_mjd_dict:
            continue
        band = filter_mjd_dict[filt]
        if band["min_mjd"] == band["max_mjd"]:
            continue
        if not (band["min_mjd"] <= spec_mjd <= band["max_mjd"]):
            continue

        if photometry_target == "gp_fit":
            fitted_logphot = float(phot4mangling_row["%s_fit_log_flux" % filt])
            fitted_logphot_err = float(phot4mangling_row["%s_fit_log_fluxerr" % filt])
            in_mjd = bool(phot4mangling_row["%s_inrange" % filt])
        else:
            raise NotImplementedError(
                "photometry_target=%r not implemented in Phase 1" % photometry_target
            )

        lam_avg, lam_eff, raw_phot, raw_phot_err, min_wls, max_wls = band_flux_trapz(
            wls, flux, fluxerr, filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
        )
        if raw_phot <= 0.0:
            continue
        condition = (max_wls > np.min(wls)) & (min_wls < np.max(wls)) & (raw_phot > 0.0)
        if not in_mjd or not condition:
            continue

        raw_logphot = np.log10(raw_phot)
        raw_logphot_err = raw_phot_err / (raw_phot * np.log(10.0))
        log_diffs.append(fitted_logphot - raw_logphot)
        log_diffs_err.append(float(np.sqrt(fitted_logphot_err ** 2 + raw_logphot_err ** 2)))
        log_wls_eff.append(float(np.log10(lam_eff)))
        used_filters.append(filt)

    if len(log_diffs) < 1:
        return None

    log_diffs_a = np.asarray(log_diffs, dtype=float)
    log_wls_eff_a = np.asarray(log_wls_eff, dtype=float)
    log_diffs_err_a = np.asarray(log_diffs_err, dtype=float)

    if len(wls) > 10 ** 4:
        full_wls = wls[:: max(1, int(len(wls) / 5000.0))]
    else:
        full_wls = wls
    full_log_wls = np.log10(full_wls)

    norm_wls = float(np.median(full_log_wls))
    full_log_wls_normed = full_log_wls - norm_wls
    log_wls_eff_normed = log_wls_eff_a - norm_wls
    norm_diff = float(np.median(log_diffs_a))
    log_diffs_normed = log_diffs_a - norm_diff

    kernel_len = _mangle_gp_kernel_length(norm_wls, kernel_divide)
    k = np.var(log_diffs_normed) * Matern32Kernel(kernel_len)
    gp = george.GP(k)
    gp.compute(np.atleast_2d(log_wls_eff_normed).T, log_diffs_err_a)
    p0 = gp.get_parameter_vector()

    def ll(p: np.ndarray) -> float:
        gp.set_parameter_vector(p)
        scale = float(np.exp(gp.get_parameter_dict()["kernel:k2:metric:log_M_0_0"]))
        if scale < min_gp_metric:
            return np.inf
        return -gp.lnlikelihood(log_diffs_normed, quiet=True)

    def grad_ll(p: np.ndarray) -> np.ndarray:
        gp.set_parameter_vector(p)
        return -gp.grad_lnlikelihood(log_diffs_normed, quiet=True)

    results = optimize.minimize(ll, p0, jac=grad_ll)
    gp.set_parameter_vector(results.x)
    mu_log, cov = gp.predict(log_diffs_normed, full_log_wls - norm_wls)
    std_log = np.sqrt(np.diag(cov))

    if len(wls) > 10 ** 4:
        mu_full_log = np.interp(np.log10(wls), full_log_wls, mu_log)
        std_full_log = np.interp(np.log10(wls), full_log_wls, std_log)
    else:
        mu_full_log = mu_log
        std_full_log = std_log

    mangling_mask = mu_full_log + norm_diff
    raw_log_flux = np.log10(np.clip(flux, 1e-30, None))
    mangled_log = raw_log_flux + mangling_mask
    mangled_log_err = np.sqrt(
        std_full_log ** 2 + (fluxerr / (flux * np.log(10.0))) ** 2
    )
    meta = {
        "used_filters": used_filters,
        "norm_diff": norm_diff,
        "n_filters": len(used_filters),
    }
    return mangling_mask, mangled_log, mangled_log_err, meta


def _collect_mangle_constraints(
    wls: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    phot4mangling_row: Mapping[str, Any],
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    """Return ``(log_diffs, log_diffs_err, log_wls_eff, used_filters)`` or ``None``."""
    spec_mjd = float(phot4mangling_row["spec_mjd"])
    log_diffs: list[float] = []
    log_diffs_err: list[float] = []
    log_wls_eff: list[float] = []
    used_filters: list[str] = []

    wls = np.asarray(wls, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fluxerr = np.asarray(fluxerr, dtype=float)

    for filt in avail_filters:
        if filt not in filter_mjd_dict:
            continue
        band = filter_mjd_dict[filt]
        if band["min_mjd"] == band["max_mjd"]:
            continue
        if not (band["min_mjd"] <= spec_mjd <= band["max_mjd"]):
            continue

        if photometry_target == "gp_fit":
            fitted_logphot = float(phot4mangling_row["%s_fit_log_flux" % filt])
            fitted_logphot_err = float(phot4mangling_row["%s_fit_log_fluxerr" % filt])
            in_mjd = bool(phot4mangling_row["%s_inrange" % filt])
        else:
            raise NotImplementedError(
                "photometry_target=%r not implemented" % photometry_target
            )

        _la, lam_eff, raw_phot, raw_phot_err, min_wls, max_wls = band_flux_trapz(
            wls, flux, fluxerr, filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
        )
        if raw_phot <= 0.0:
            continue
        condition = (max_wls > np.min(wls)) & (min_wls < np.max(wls)) & (raw_phot > 0.0)
        if not in_mjd or not condition:
            continue

        raw_logphot = np.log10(raw_phot)
        raw_logphot_err = raw_phot_err / (raw_phot * np.log(10.0))
        log_diffs.append(fitted_logphot - raw_logphot)
        log_diffs_err.append(float(np.sqrt(fitted_logphot_err ** 2 + raw_logphot_err ** 2)))
        log_wls_eff.append(float(np.log10(lam_eff)))
        used_filters.append(filt)

    if len(log_diffs) < 1:
        return None
    return (
        np.asarray(log_diffs, dtype=float),
        np.asarray(log_diffs_err, dtype=float),
        np.asarray(log_wls_eff, dtype=float),
        used_filters,
    )


def _fit_mangling_mask_on_grid(
    log_diffs: np.ndarray,
    log_diffs_err: np.ndarray,
    log_wls_eff: np.ndarray,
    target_wls_linear: np.ndarray,
    *,
    kernel_divide: int = 800,
    min_gp_metric: float = 0.09,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Matern32 GP mangling mask; evaluate on ``target_wls_linear`` (linear Å)."""
    from scipy import optimize

    import george
    from george.kernels import Matern32Kernel

    wls = np.asarray(target_wls_linear, dtype=float)
    if len(wls) > 10 ** 4:
        full_wls = wls[:: max(1, int(len(wls) / 5000.0))]
    else:
        full_wls = wls
    full_log_wls = np.log10(full_wls)

    norm_wls = float(np.median(full_log_wls))
    log_wls_eff_normed = np.asarray(log_wls_eff, dtype=float) - norm_wls
    norm_diff = float(np.median(log_diffs))
    log_diffs_normed = np.asarray(log_diffs, dtype=float) - norm_diff

    kernel_len = _mangle_gp_kernel_length(norm_wls, kernel_divide)
    k = np.var(log_diffs_normed) * Matern32Kernel(kernel_len)
    gp = george.GP(k)
    gp.compute(np.atleast_2d(log_wls_eff_normed).T, np.asarray(log_diffs_err, dtype=float))
    p0 = gp.get_parameter_vector()

    def ll(p: np.ndarray) -> float:
        gp.set_parameter_vector(p)
        scale = float(np.exp(gp.get_parameter_dict()["kernel:k2:metric:log_M_0_0"]))
        if scale < min_gp_metric:
            return np.inf
        return -gp.lnlikelihood(log_diffs_normed, quiet=True)

    def grad_ll(p: np.ndarray) -> float:
        gp.set_parameter_vector(p)
        return -gp.grad_lnlikelihood(log_diffs_normed, quiet=True)

    results = optimize.minimize(ll, p0, jac=grad_ll)
    gp.set_parameter_vector(results.x)
    mu_log, cov = gp.predict(log_diffs_normed, full_log_wls - norm_wls)
    std_log = np.sqrt(np.diag(cov))

    if len(wls) > 10 ** 4:
        mu_full_log = np.interp(np.log10(wls), full_log_wls, mu_log)
        std_full_log = np.interp(np.log10(wls), full_log_wls, std_log)
    else:
        mu_full_log = mu_log
        std_full_log = std_log

    mangling_mask = mu_full_log + norm_diff
    meta = {"norm_diff": norm_diff, "n_constraints": int(log_diffs.size)}
    return mangling_mask, std_full_log, meta


def _filter_overlap_width(min_wls: float, max_wls: float, arm_wls: np.ndarray) -> float:
    arm_min = float(np.min(arm_wls))
    arm_max = float(np.max(arm_wls))
    lo = max(min_wls, arm_min)
    hi = min(max_wls, arm_max)
    return max(0.0, hi - lo)


def _filter_linear_wavelength_range(
    filter_name: str,
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
) -> tuple[float, float]:
    filt_path = _resolve_filter_path(filter_path, filter_name, snname, csp_sne)
    filt = np.genfromtxt(
        filt_path, dtype=None, encoding="utf-8", names=["wls", "flux"]
    )
    return float(np.min(filt["wls"])), float(np.max(filt["wls"]))


def _combined_filter_overlap_width(
    fmin: float, fmax: float, arm_wls_list: Sequence[np.ndarray]
) -> float:
    intervals: list[tuple[float, float]] = []
    for wls in arm_wls_list:
        w = np.asarray(wls, dtype=float)
        if w.size < 1:
            continue
        lo = max(fmin, float(np.min(w)))
        hi = min(fmax, float(np.max(w)))
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda t: t[0])
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return float(sum(hi - lo for lo, hi in merged))


def _ordered_member_names(
    names: Sequence[str], merge_order: Optional[Sequence[str]]
) -> list[str]:
    from spectra_pre_scale import _arm_sort_key

    if merge_order:
        return sorted(names, key=lambda n: _arm_sort_key(n, list(merge_order)))
    return sorted(names)


def stitch_arm_spectrum_for_filter(
    member_specs: Mapping[str, np.ndarray],
    names_ordered: Sequence[str],
    fmin: float,
    fmax: float,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Concatenate prescaled/demangled bins from multiple arms within a filter band."""
    wls_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    for name in names_ordered:
        if name not in member_specs:
            continue
        spec = member_specs[name]
        w = np.asarray(spec["wls"], dtype=float)
        cut = (w > fmin) & (w < fmax)
        if not np.any(cut):
            continue
        wls_parts.append(w[cut])
        flux_parts.append(np.asarray(spec["flux"], dtype=float)[cut])
        err_parts.append(np.asarray(spec["fluxerr"], dtype=float)[cut])
    if not wls_parts:
        return None
    wls = np.concatenate(wls_parts)
    flux = np.concatenate(flux_parts)
    err = np.concatenate(err_parts)
    order = np.argsort(wls)
    return wls[order], flux[order], err[order]


def _single_filter_constraint(
    wls: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    phot4mangling_row: Mapping[str, Any],
    filt: str,
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
) -> Optional[tuple[float, float, float]]:
    c = _collect_mangle_constraints(
        wls,
        flux,
        fluxerr,
        phot4mangling_row,
        [filt],
        filter_mjd_dict,
        filter_path=filter_path,
        snname=snname,
        csp_sne=csp_sne,
        photometry_target=photometry_target,
    )
    if c is None:
        return None
    ld, le, lw, uf = c
    if len(ld) != 1 or len(uf) != 1:
        return None
    return float(ld[0]), float(le[0]), float(lw[0])


def _bundle_constraint_for_filter(
    filt: str,
    member_specs: Mapping[str, np.ndarray],
    names: Sequence[str],
    phot4mangling,
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
    member_mjd: Optional[Mapping[str, float]] = None,
    merge_order: Optional[Sequence[str]] = None,
    use_gp_wavelength_grid: bool = False,
    stitch_synphot: bool = True,
) -> Optional[tuple[float, float, float, str]]:
    """Return ``(log_diff, log_diff_err, log_wls_eff, mode_tag)`` for one filter."""
    if filt not in filter_mjd_dict:
        return None
    fmin, fmax = _filter_linear_wavelength_range(
        filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
    )
    names_ordered = _ordered_member_names(names, merge_order)

    if use_gp_wavelength_grid:
        ref_name = max(names, key=lambda n: len(member_specs[n]["wls"]))
        spec = member_specs[ref_name]
        mjd = float(member_mjd[ref_name]) if member_mjd and ref_name in member_mjd else float(spec["wls"][0])
        row = _phot_row_for_mjd(phot4mangling, mjd)
        out = _single_filter_constraint(
            np.asarray(spec["wls"], dtype=float),
            np.asarray(spec["flux"], dtype=float),
            np.asarray(spec["fluxerr"], dtype=float),
            row,
            filt,
            filter_mjd_dict,
            filter_path=filter_path,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=photometry_target,
        )
        if out is None:
            return None
        d, e, wl = out
        return d, e, wl, "gp_grid"

    overlapping: list[tuple[str, float]] = []
    arm_wls_list: list[np.ndarray] = []
    best_name: Optional[str] = None
    best_width = -1.0
    best_constraint: Optional[tuple[float, float, float]] = None
    for name in names:
        spec = member_specs[name]
        wls = np.asarray(spec["wls"], dtype=float)
        flux = np.asarray(spec["flux"], dtype=float)
        fluxerr = np.asarray(spec["fluxerr"], dtype=float)
        width = _filter_overlap_width(fmin, fmax, wls)
        if width <= 0.0:
            continue
        overlapping.append((name, width))
        arm_wls_list.append(wls)
        mjd = float(member_mjd[name]) if member_mjd and name in member_mjd else float(wls[0])
        row = _phot_row_for_mjd(phot4mangling, mjd)
        out = _single_filter_constraint(
            wls, flux, fluxerr, row, filt, filter_mjd_dict,
            filter_path=filter_path, snname=snname, csp_sne=csp_sne,
            photometry_target=photometry_target,
        )
        if out is None:
            continue
        if width > best_width:
            best_width = width
            best_name = name
            best_constraint = out

    if best_name is None or best_constraint is None:
        return None

    combined_width = _combined_filter_overlap_width(fmin, fmax, arm_wls_list)
    use_stitch = (
        bool(stitch_synphot)
        and len(overlapping) >= 2
        and combined_width > best_width + 1e-9
    )
    if use_stitch:
        stitched = stitch_arm_spectrum_for_filter(
            member_specs, names_ordered, fmin, fmax
        )
        if stitched is not None:
            sw, sf, se = stitched
            ref_name = overlapping[0][0]
            mjd = float(member_mjd[ref_name]) if member_mjd and ref_name in member_mjd else float(sw[0])
            row = _phot_row_for_mjd(phot4mangling, mjd)
            out = _single_filter_constraint(
                sw, sf, se, row, filt, filter_mjd_dict,
                filter_path=filter_path, snname=snname, csp_sne=csp_sne,
                photometry_target=photometry_target,
            )
            if out is not None:
                d, e, wl = out
                return d, e, wl, "stitched"

    d, e, wl = best_constraint
    return d, e, wl, "best_arm:%s" % best_name


def _phot_row_for_mjd(phot4mangling, spec_mjd: float):
    import pandas as pd

    df = phot4mangling
    if not isinstance(df, pd.DataFrame):
        raise TypeError("phot4mangling must be a pandas DataFrame")
    rows = df[np.isclose(df["spec_mjd"].astype(float), float(spec_mjd), rtol=0.0, atol=1e-6)]
    if len(rows) == 1:
        return rows.iloc[0]
    if len(rows) == 0:
        idx = int(np.argmin(np.abs(df["spec_mjd"].astype(float) - float(spec_mjd))))
        return df.iloc[idx]
    return rows.iloc[0]


def _avail_filters_from_phot4mangling(phot4mangling) -> list[str]:
    return [
        col.replace("_fit_log_flux", "")
        for col in phot4mangling.columns
        if col.endswith("_fit_log_flux")
    ]


def compute_mangling_mask_bundle(
    member_specs: Mapping[str, np.ndarray],
    phot4mangling,
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
    member_mjd: Optional[Mapping[str, float]] = None,
    merge_order: Optional[Sequence[str]] = None,
    use_gp_wavelength_grid: bool = False,
    stitch_synphot: Optional[bool] = None,
    kernel_divide: int = 800,
    min_gp_metric: float = 0.09,
) -> Optional[tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]]:
    """One shared mangling mask per exposure group; applied per arm.

    Returns ``(masks_by_name, mangled_by_name, meta)`` where ``mangled_by_name[name]`` is
    ``(log_mangled_flux, log_mangled_err, linear_wls)``.
    """
    if len(member_specs) < 1:
        return None

    try:
        import pipeline_config as _pc

        if stitch_synphot is None:
            stitch_synphot = bool(getattr(_pc, "MANGLE_BUNDLE_STITCH_SYNPHOT", True))
    except ImportError:
        if stitch_synphot is None:
            stitch_synphot = True

    names = sorted(member_specs.keys())
    pooled_diffs: list[float] = []
    pooled_errs: list[float] = []
    pooled_wls: list[float] = []
    used_filters: list[str] = []
    mode_for_filter: dict[str, str] = {}

    for filt in avail_filters:
        c = _bundle_constraint_for_filter(
            filt,
            member_specs,
            names,
            phot4mangling,
            filter_mjd_dict,
            filter_path=filter_path,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=photometry_target,
            member_mjd=member_mjd,
            merge_order=merge_order,
            use_gp_wavelength_grid=use_gp_wavelength_grid,
            stitch_synphot=bool(stitch_synphot),
        )
        if c is None:
            continue
        d, e, wl, mode_tag = c
        pooled_diffs.append(d)
        pooled_errs.append(e)
        pooled_wls.append(wl)
        used_filters.append(filt)
        mode_for_filter[filt] = mode_tag

    if len(pooled_diffs) < 1:
        return None

    if use_gp_wavelength_grid:
        ref_name = max(names, key=lambda n: len(member_specs[n]["wls"]))
        fit_wls = np.asarray(member_specs[ref_name]["wls"], dtype=float)
    else:
        all_wls = np.unique(
            np.concatenate([np.asarray(member_specs[n]["wls"], dtype=float) for n in names])
        )
        all_wls.sort()
        fit_wls = all_wls

    if fit_wls.size > 8000:
        step = max(1, fit_wls.size // 5000)
        fit_wls = fit_wls[::step]

    mask_fit, std_fit, fit_meta = _fit_mangling_mask_on_grid(
        np.asarray(pooled_diffs),
        np.asarray(pooled_errs),
        np.asarray(pooled_wls),
        fit_wls,
        kernel_divide=kernel_divide,
        min_gp_metric=min_gp_metric,
    )

    masks: dict[str, np.ndarray] = {}
    mangled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in names:
        spec = member_specs[name]
        wls = np.asarray(spec["wls"], dtype=float)
        flux = np.asarray(spec["flux"], dtype=float)
        fluxerr = np.asarray(spec["fluxerr"], dtype=float)
        m = np.interp(np.log10(wls), np.log10(fit_wls), mask_fit)
        s = np.interp(np.log10(wls), np.log10(fit_wls), std_fit)
        masks[name] = m
        raw_log = np.log10(np.clip(flux, 1e-30, None))
        mangled_log = raw_log + m
        mangled_err = np.sqrt(s ** 2 + (fluxerr / (flux * np.log(10.0))) ** 2)
        mangled[name] = (mangled_log, mangled_err, wls)

    meta = {
        "used_filters": used_filters,
        "mode_for_filter": mode_for_filter,
        "n_filters": len(used_filters),
        "bundle_mode": True,
        "use_gp_wavelength_grid": bool(use_gp_wavelength_grid),
        "stitch_synphot": bool(stitch_synphot),
        **fit_meta,
    }
    return masks, mangled, meta


def compute_seam_jumps(
    arm_specs: Mapping[str, np.ndarray],
    merge_order: Sequence[str],
    *,
    flux_in_log10: bool = False,
    half_width_a: float = 50.0,
) -> list[dict[str, Any]]:
    """Median |Δlog10 F| at arm-pair join edges (prescaled or mangled)."""
    from spectra_pre_scale import _arm_sort_key

    ordered = sorted(
        arm_specs.keys(),
        key=lambda n: _arm_sort_key(n, list(merge_order)),
    )
    out: list[dict[str, Any]] = []
    for i in range(len(ordered) - 1):
        blue_name, red_name = ordered[i], ordered[i + 1]
        blue = arm_specs[blue_name]
        red = arm_specs[red_name]
        bw = np.asarray(blue["wls"], dtype=float)
        rw = np.asarray(red["wls"], dtype=float)

        def _log_flux(spec, wsel):
            f = np.asarray(spec["flux"], dtype=float)[wsel]
            if flux_in_log10:
                return f[np.isfinite(f)]
            return np.log10(np.clip(f[np.isfinite(f) & (f > 0)], 1e-30, None))

        b_hi = bw >= float(np.max(bw)) - half_width_a
        r_lo = rw <= float(np.min(rw)) + half_width_a
        bf = _log_flux(blue, b_hi)
        rf = _log_flux(red, r_lo)
        jump = float("nan")
        if bf.size and rf.size:
            jump = float(abs(np.median(bf) - np.median(rf)))
        out.append(
            {
                "ref": blue_name,
                "arm": red_name,
                "seam_jump_log10": jump,
                "gap_a": float(max(0.0, float(np.min(rw)) - float(np.max(bw)))),
            }
        )
    return out


def build_mangle_group_map(
    entries: Sequence[Any],
    groups: Sequence[Any],
) -> tuple[dict[str, str], dict[str, list[Any]]]:
    """Map basename → group_id and group_id → list of entries."""
    basename_to_group: dict[str, str] = {}
    group_entries: dict[str, list[Any]] = {}
    entry_by_bn = {e.basename: e for e in entries}
    for g in groups:
        members_in_list = []
        for m in g.members:
            bn = os.path.basename(str(m))
            if bn in entry_by_bn:
                basename_to_group[bn] = g.id
                members_in_list.append(entry_by_bn[bn])
        if members_in_list:
            group_entries[g.id] = members_in_list
    return basename_to_group, group_entries


def _mangled_output_basename(spec_mjd: float) -> str:
    return "%.6f_mangled_spec.txt" % float(spec_mjd)


def run_mangle_pipeline(
    snname: str,
    *,
    coco_path: str,
    output_dir: str,
    bundle_aware: bool = False,
    groups_json: str | None = None,
    save_diagnostics: bool = True,
    save_epoch_plots: bool = False,
    run_both_for_diag: bool = False,
    photometry_target: str | None = None,
    filter_path: str | None = None,
    kernel_divide: int | None = None,
    csp_sne: Sequence[str] = (),
    verbose: bool = False,
) -> dict[str, Any]:
    """Mangle all spectra in the spec list; write ``Outputs/<SN>/mangled_spectra/``.

    Returns summary dict with counts, report path, diagnostics path.
    """
    import json

    import pandas as pd

    import pipeline_config as pconf
    from photometry_filter_utils import (
        load_band_mjd_ranges_json,
        write_band_mjd_ranges_json,
    )
    from spectra_pre_scale import ScaleGroup, load_scale_groups_json, load_spec_list

    out_root = output_dir.rstrip(os.sep) + os.sep
    list_path = pconf.spec_list_path_for_mangling(coco_path, snname)
    entries = load_spec_list(list_path)
    if not entries:
        raise FileNotFoundError("Empty or missing spec list: %s" % list_path)

    mangled_dir = os.path.join(out_root, snname, "mangled_spectra")
    os.makedirs(mangled_dir, exist_ok=True)

    phot_path = os.path.join(out_root, snname, "fitted_phot4mangling_%s.dat" % snname)
    phot4mangling = pd.read_csv(phot_path, sep="\t")
    mjd_json = pconf.band_mjd_ranges_json_path(out_root, snname)
    if not os.path.isfile(mjd_json):
        raw_lc = pconf.raw_photometry_path(coco_path, snname)
        write_band_mjd_ranges_json(raw_lc, mjd_json)
        if verbose:
            print(
                "Wrote band MJD ranges: %s (from %s)" % (mjd_json, raw_lc),
                flush=True,
            )
    filter_mjd_dict = load_band_mjd_ranges_json(mjd_json)
    avail_filters = _avail_filters_from_phot4mangling(phot4mangling)
    filt_root = filter_path or pconf.filters_root(coco_path)
    if not filt_root.endswith(os.sep):
        filt_root = filt_root.rstrip(os.sep) + os.sep
    pt = photometry_target or getattr(pconf, "MANGLE_PHOTOMETRY_TARGET", "gp_fit")
    kd = int(kernel_divide if kernel_divide is not None else pconf.MANGLE_KERNEL_DIVIDE)

    rt_style = pconf.bootstrap_runtime(photometry_stage="extrapolated", snname=snname)
    epoch_plot_dir = pconf.sn_figs_dir(out_root, snname, pconf.MANGLE_EPOCH_FIGS_SUBDIR)
    if save_epoch_plots:
        os.makedirs(epoch_plot_dir, exist_ok=True)

    groups: list[ScaleGroup] = []
    group_entries: dict[str, list[Any]] = {}
    basename_to_group: dict[str, str] = {}
    if bundle_aware or run_both_for_diag:
        gj = pconf.resolve_mangle_groups_json(out_root, snname, groups_json)
        if os.path.isfile(gj):
            _, groups = load_scale_groups_json(gj)
            basename_to_group, group_entries = build_mangle_group_map(entries, groups)
        elif bundle_aware:
            raise FileNotFoundError(
                "bundle_aware=True but scale groups JSON not found: %s" % gj
            )

    report: dict[str, Any] = {
        "snname": snname,
        "bundle_aware": bool(bundle_aware),
        "spec_list": list_path,
        "groups": [],
        "ungrouped": [],
        "n_mangled": 0,
        "n_failed": 0,
        "failed": [],
    }

    diag_payload_groups: list[dict[str, Any]] = []

    def _record_mangle_failure(
        entry: Any,
        reason: str,
        *,
        group_id: str | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "basename": entry.basename,
            "mjd": float(entry.mjd),
            "path": entry.path,
            "reason": reason,
        }
        if group_id is not None:
            rec["group_id"] = group_id
        report["failed"].append(rec)
        print(
            "[mangle] FAILED MJD=%.6f basename=%s reason=%s"
            % (float(entry.mjd), entry.basename, reason),
            flush=True,
        )

    def _mangle_single(entry, *, use_bundle: bool) -> bool:
        spec = load_linear_spectrum(entry.path)
        row = _phot_row_for_mjd(phot4mangling, entry.mjd)
        out = compute_mangling_mask(
            spec,
            row,
            avail_filters,
            filter_mjd_dict,
            filter_path=filt_root,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=pt,
            kernel_divide=kd,
        )
        if out is None:
            _record_mangle_failure(entry, "no_phot_constraints")
            return False
        mask, log_f, log_e, meta = out
        save_mangled_spectrum(
            os.path.join(mangled_dir, _mangled_output_basename(entry.mjd)),
            spec["wls"],
            log_f,
            log_e,
            mask,
        )
        if save_epoch_plots:
            try:
                from mangle_epoch_plots import save_mangle_epoch_plot

                save_mangle_epoch_plot(
                    raw_spec=spec,
                    phot_row=row,
                    avail_filters=avail_filters,
                    filter_mjd_dict=filter_mjd_dict,
                    filter_path=filt_root,
                    snname=snname,
                    color_dict=rt_style.color_dict,
                    mark_dict=rt_style.mark_dict,
                    out_path=os.path.join(epoch_plot_dir, "%.6f_mangle_epoch.pdf" % entry.mjd),
                    kernel_divide=kd,
                    csp_sne=csp_sne,
                )
            except Exception as exc:
                if verbose:
                    print("[mangle] epoch plot failed:", exc, flush=True)
        report["ungrouped"].append(
            {"basename": entry.basename, "mjd": entry.mjd, "meta": meta, "mode": "per_arm"}
        )
        return True

    def _mangle_group(gid: str, members: list[Any], *, write_files: bool) -> Optional[dict[str, Any]]:
        specs = {e.basename: load_linear_spectrum(e.path) for e in members}
        mjds = {e.basename: float(e.mjd) for e in members}
        out = compute_mangling_mask_bundle(
            specs,
            phot4mangling,
            avail_filters,
            filter_mjd_dict,
            filter_path=filt_root,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=pt,
            member_mjd=mjds,
            merge_order=next((g.merge_order for g in groups if g.id == gid), None),
            use_gp_wavelength_grid=False,
            kernel_divide=kd,
        )
        if out is None:
            return None
        masks, mangled, meta = out
        merge_order = next((g.merge_order for g in groups if g.id == gid), [])
        prescaled_linear = {n: {"wls": specs[n]["wls"], "flux": specs[n]["flux"]} for n in specs}
        mangled_linear = {
            n: {
                "wls": mangled[n][2],
                "flux": np.power(10.0, np.clip(mangled[n][0], -50, 50)),
            }
            for n in mangled
        }
        seam_prescale = compute_seam_jumps(prescaled_linear, merge_order, flux_in_log10=False)
        seam_mangled = compute_seam_jumps(mangled_linear, merge_order, flux_in_log10=False)
        seam_flags = []
        reg_factor = float(getattr(pconf, "MANGLE_SEAM_REGRESSION_FACTOR", 1.5))
        for sp, sm in zip(seam_prescale, seam_mangled):
            jp = float(sp.get("seam_jump_log10", float("nan")))
            jm = float(sm.get("seam_jump_log10", float("nan")))
            flagged = bool(
                np.isfinite(jp)
                and np.isfinite(jm)
                and jp > 0
                and jm > reg_factor * jp
            )
            seam_flags.append({**sm, "prescale_jump": jp, "regression_flag": flagged})

        if write_files:
            for e in members:
                log_f, log_e, wls = mangled[e.basename]
                save_mangled_spectrum(
                    os.path.join(mangled_dir, _mangled_output_basename(e.mjd)),
                    wls,
                    log_f,
                    log_e,
                    masks[e.basename],
                )
            if save_epoch_plots:
                try:
                    from mangle_epoch_plots import save_mangle_epoch_plot_bundle

                    row = _phot_row_for_mjd(phot4mangling, float(members[0].mjd))
                    save_mangle_epoch_plot_bundle(
                        group_id=gid,
                        members=members,
                        specs=specs,
                        masks=masks,
                        mangled=mangled,
                        meta=meta,
                        phot_row=row,
                        avail_filters=avail_filters,
                        filter_mjd_dict=filter_mjd_dict,
                        filter_path=filt_root,
                        snname=snname,
                        color_dict=rt_style.color_dict,
                        mark_dict=rt_style.mark_dict,
                        out_path=os.path.join(epoch_plot_dir, "group_%s_epoch.pdf" % gid),
                        merge_order=merge_order,
                        csp_sne=csp_sne,
                    )
                except Exception as exc:
                    if verbose:
                        print("[mangle] bundle epoch plot failed:", exc, flush=True)

        return {
            "id": gid,
            "members": [e.basename for e in members],
            "merge_order": merge_order,
            "meta": meta,
            "seam_qa": seam_flags,
            "prescaled": prescaled_linear,
            "mangled": mangled_linear,
            "masks": masks,
        }

    grouped_basenames = set(basename_to_group.keys())

    if bundle_aware and not run_both_for_diag:
        for gid, members in group_entries.items():
            greport = _mangle_group(gid, members, write_files=True)
            if greport is None:
                report["n_failed"] += len(members)
                for entry in members:
                    _record_mangle_failure(entry, "bundle_mask_failed", group_id=gid)
            else:
                report["n_mangled"] += len(members)
                report["groups"].append({k: greport[k] for k in greport if k not in ("prescaled", "mangled", "masks")})
                diag_payload_groups.append(greport)
        for entry in entries:
            if entry.basename in grouped_basenames:
                continue
            if _mangle_single(entry, use_bundle=False):
                report["n_mangled"] += 1
            else:
                report["n_failed"] += 1
    elif run_both_for_diag:
        per_arm_by_group: dict[str, dict[str, np.ndarray]] = {}
        per_arm_flux_by_group: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for entry in entries:
            spec = load_linear_spectrum(entry.path)
            row = _phot_row_for_mjd(phot4mangling, entry.mjd)
            out = compute_mangling_mask(
                spec,
                row,
                avail_filters,
                filter_mjd_dict,
                filter_path=filt_root,
                snname=snname,
                csp_sne=csp_sne,
                photometry_target=pt,
                kernel_divide=kd,
            )
            if out is None:
                report["n_failed"] += 1
                _record_mangle_failure(entry, "no_phot_constraints")
                continue
            mask, log_f, log_e, meta = out
            gid = basename_to_group.get(entry.basename)
            if gid:
                per_arm_by_group.setdefault(gid, {})[entry.basename] = mask
                per_arm_flux_by_group.setdefault(gid, {})[entry.basename] = {
                    "wls": np.asarray(spec["wls"], dtype=float),
                    "flux": np.power(10.0, np.clip(log_f, -50, 50)),
                }
            save_mangled_spectrum(
                os.path.join(mangled_dir, _mangled_output_basename(entry.mjd)),
                spec["wls"],
                log_f,
                log_e,
                mask,
            )
            report["n_mangled"] += 1
        for gid, members in group_entries.items():
            greport = _mangle_group(gid, members, write_files=False)
            if greport is None:
                continue
            greport["per_arm_masks"] = per_arm_by_group.get(gid, {})
            greport["mangled_per_arm"] = per_arm_flux_by_group.get(gid, {})
            diag_payload_groups.append(greport)
            report["groups"].append(
                {
                    k: greport[k]
                    for k in greport
                    if k
                    not in (
                        "prescaled",
                        "mangled",
                        "masks",
                        "per_arm_masks",
                        "mangled_per_arm",
                    )
                }
            )
    else:
        for entry in entries:
            if _mangle_single(entry, use_bundle=False):
                report["n_mangled"] += 1
            else:
                report["n_failed"] += 1

    report_path = os.path.join(out_root, snname, "%s_mangle_report.json" % snname)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    diag_dir = None
    if save_diagnostics and getattr(pconf, "MANGLE_SAVE_DIAGNOSTICS", True):
        try:
            import mangle_diagnostics as md

            diag_dir = pconf.mangle_diagnostics_dir(out_root, snname)
            md.run_mangle_diagnostics(
                diag_dir,
                diag_payload_groups,
                report,
                per_arm_compare=run_both_for_diag,
            )
        except ImportError:
            if verbose:
                print("[mangle] mangle_diagnostics not available", flush=True)

    if verbose or report["n_failed"] > 0:
        msg = "[mangle] done n_mangled=%i n_failed=%i report=%s" % (
            report["n_mangled"],
            report["n_failed"],
            report_path,
        )
        if report["n_failed"] > 0:
            msg += " (see failed[] in report)"
        print(msg, flush=True)
    return {
        "report_path": report_path,
        "diagnostics_dir": diag_dir,
        "mangled_dir": mangled_dir,
        "report": report,
    }
