"""Per-epoch mangling QA plots with photometry overlay (notebook-style)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from mangle_spectra_log import band_flux_trapz, compute_mangling_mask


def save_mangle_epoch_plot(
    *,
    raw_spec: np.ndarray,
    phot_row: Mapping[str, Any],
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    filter_path: str,
    snname: str,
    color_dict: Mapping[str, str],
    mark_dict: Mapping[str, str],
    out_path: str,
    kernel_divide: int = 800,
    csp_sne: Sequence[str] = (),
) -> bool:
    """Fit mangling mask and save two-panel PDF (phot constraints + spectrum)."""
    out = compute_mangling_mask(
        raw_spec,
        phot_row,
        avail_filters,
        filter_mjd_dict,
        filter_path=filter_path,
        snname=snname,
        csp_sne=csp_sne,
        kernel_divide=kernel_divide,
    )
    if out is None:
        return False

    mask, mangled_log, mangled_log_err, meta = out
    wls = np.asarray(raw_spec["wls"], dtype=float)
    flux = np.asarray(raw_spec["flux"], dtype=float)
    fluxerr = np.asarray(raw_spec["fluxerr"], dtype=float)
    used_filters = list(meta.get("used_filters", []))

    log_diffs: list[float] = []
    log_diffs_err: list[float] = []
    log_wls_eff: list[float] = []
    fitted_log: list[float] = []
    fitted_log_err: list[float] = []

    for filt in used_filters:
        fitted_logphot = float(phot_row["%s_fit_log_flux" % filt])
        fitted_logphot_err = float(phot_row["%s_fit_log_fluxerr" % filt])
        _, lam_eff, raw_phot, raw_phot_err, min_wls, max_wls = band_flux_trapz(
            wls, flux, fluxerr, filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
        )
        if raw_phot <= 0:
            continue
        condition = (max_wls > np.min(wls)) & (min_wls < np.max(wls))
        if not condition:
            continue
        raw_logphot = np.log10(raw_phot)
        raw_logphot_err = raw_phot_err / (raw_phot * np.log(10.0))
        log_diffs.append(fitted_logphot - raw_logphot)
        log_diffs_err.append(float(np.sqrt(fitted_logphot_err ** 2 + raw_logphot_err ** 2)))
        log_wls_eff.append(float(np.log10(lam_eff)))
        fitted_log.append(fitted_logphot)
        fitted_log_err.append(fitted_logphot_err)

    if not log_diffs:
        return False

    log_wls_plot = np.log10(wls if len(wls) <= 10 ** 4 else wls[:: max(1, len(wls) // 5000)])
    norm_wls = float(np.median(log_wls_plot))
    norm_diff = float(meta.get("norm_diff", 0.0))

    fig = plt.figure(figsize=(14, 6))
    plt.rc("font", family="serif")
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    for f, w, r, rerr, fl, flerr in zip(
        used_filters[: len(log_wls_eff)],
        log_wls_eff,
        log_diffs,
        log_diffs_err,
        fitted_log,
        fitted_log_err,
    ):
        flabel = f.split("_")[1] + "(Swift)" if "swift" in f else f
        ax1.errorbar(
            w,
            r,
            yerr=rerr,
            marker=mark_dict.get(f, "o"),
            ms=8,
            mfc=color_dict.get(f, "grey"),
            mec=color_dict.get(f, "grey"),
            linestyle="None",
            ecolor=color_dict.get(f, "grey"),
            label=flabel,
        )
    ax1.plot(
        log_wls_plot,
        mask[: len(log_wls_plot)] if len(mask) == len(log_wls_plot) else mask,
        color="orange",
        label="Mangling\nfunction",
    )
    ax1.set_ylabel("Log Flux Difference\n(Fitted - Synthetic)", fontsize=13)
    ax1.set_title("Mangling epoch (Log-Space)", fontsize=15)
    ax1.legend(ncol=2, fontsize=10, loc="best")
    plt.tick_params(axis="x", labelbottom=False)

    raw_log_flux = np.log10(np.clip(flux, 1e-30, None))
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)
    ax2.plot(np.log10(wls), mangled_log, lw=0.9, color="k", label="Mangled Spectrum")
    ax2.plot(np.log10(wls), raw_log_flux, lw=0.6, color="r", alpha=1, label="Uncalibrated Spectrum")
    ax2.errorbar(
        log_wls_eff,
        fitted_log,
        yerr=fitted_log_err,
        marker="o",
        mfc="grey",
        mec="grey",
        ms=7,
        ecolor="grey",
        linestyle="None",
        label="Photometric flux\nfrom LC fitting",
    )
    ax2.set_ylabel(r"Log Flux ($\log_{10}(\mathrm{erg \ s^{-1} cm^{-2} \AA^{-1}})$)", fontsize=13)
    ax2.set_xlabel(r"Log Wavelength ($\log_{10}(\mathrm{\AA})$)", fontsize=13)
    ax2.legend(ncol=1, fontsize=10, loc="upper right")
    plt.subplots_adjust(hspace=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_mangle_epoch_plot_bundle(
    *,
    group_id: str,
    members: Sequence[Any],
    specs: Mapping[str, Mapping[str, np.ndarray]],
    masks: Mapping[str, np.ndarray],
    mangled: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    meta: Mapping[str, Any],
    phot_row: Mapping[str, Any],
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    filter_path: str,
    snname: str,
    color_dict: Mapping[str, str],
    mark_dict: Mapping[str, str],
    out_path: str,
    merge_order: Sequence[str] | None = None,
    csp_sne: Sequence[str] = (),
) -> bool:
    """Bundle-aware epoch plot using pre-fit shared mask (no re-fit per arm)."""
    used_filters = list(meta.get("used_filters", []))
    if not used_filters:
        return False

    ref_bn = members[0].basename if members else next(iter(specs.keys()), None)
    if ref_bn is None or ref_bn not in specs:
        return False
    raw_spec = specs[ref_bn]
    wls = np.asarray(raw_spec["wls"], dtype=float)
    flux = np.asarray(raw_spec["flux"], dtype=float)
    fluxerr = np.asarray(raw_spec["fluxerr"], dtype=float)
    mask_arr = np.asarray(masks.get(ref_bn, mangled[ref_bn][0]), dtype=float)
    mangled_log = np.asarray(mangled[ref_bn][0], dtype=float)

    log_diffs: list[float] = []
    log_diffs_err: list[float] = []
    log_wls_eff: list[float] = []
    fitted_log: list[float] = []
    fitted_log_err: list[float] = []

    for filt in used_filters:
        col = "%s_fit_log_flux" % filt
        if col not in phot_row:
            continue
        fitted_logphot = float(phot_row[col])
        fitted_logphot_err = float(phot_row["%s_fit_log_fluxerr" % filt])
        _, lam_eff, raw_phot, raw_phot_err, min_wls, max_wls = band_flux_trapz(
            wls, flux, fluxerr, filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
        )
        if raw_phot <= 0:
            continue
        if not ((max_wls > np.min(wls)) & (min_wls < np.max(wls))):
            continue
        raw_logphot = np.log10(raw_phot)
        raw_logphot_err = raw_phot_err / (raw_phot * np.log(10.0))
        log_diffs.append(fitted_logphot - raw_logphot)
        log_diffs_err.append(float(np.sqrt(fitted_logphot_err ** 2 + raw_logphot_err ** 2)))
        log_wls_eff.append(float(np.log10(lam_eff)))
        fitted_log.append(fitted_logphot)
        fitted_log_err.append(fitted_logphot_err)

    if not log_diffs:
        return False

    log_wls_plot = np.log10(wls if len(wls) <= 10 ** 4 else wls[:: max(1, len(wls) // 5000)])
    raw_log_flux = np.log10(np.clip(flux, 1e-30, None))
    member_names = ", ".join(getattr(m, "basename", str(m)) for m in members[:3])
    if len(members) > 3:
        member_names += ", …"

    fig = plt.figure(figsize=(14, 6))
    plt.rc("font", family="serif")
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    for f, w, r, rerr, fl, flerr in zip(
        used_filters[: len(log_wls_eff)],
        log_wls_eff,
        log_diffs,
        log_diffs_err,
        fitted_log,
        fitted_log_err,
    ):
        flabel = f.split("_")[1] + "(Swift)" if "swift" in f else f
        ax1.errorbar(
            w,
            r,
            yerr=rerr,
            marker=mark_dict.get(f, "o"),
            ms=8,
            mfc=color_dict.get(f, "grey"),
            mec=color_dict.get(f, "grey"),
            linestyle="None",
            ecolor=color_dict.get(f, "grey"),
            label=flabel,
        )
    n_mask = min(len(mask_arr), len(log_wls_plot))
    ax1.plot(log_wls_plot[:n_mask], mask_arr[:n_mask], color="orange", label="Bundle mask")
    ax1.set_ylabel("Log Flux Difference\n(Fitted - Synthetic)", fontsize=13)
    ax1.set_title(
        "Bundle mangling %s (%s)" % (group_id, member_names),
        fontsize=14,
    )
    ax1.legend(ncol=2, fontsize=9, loc="best")
    plt.tick_params(axis="x", labelbottom=False)

    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)
    ax2.plot(np.log10(wls), mangled_log, lw=0.9, color="k", label="Mangled (bundle)")
    ax2.plot(np.log10(wls), raw_log_flux, lw=0.6, color="r", alpha=1, label="Prescaled")
    ax2.errorbar(
        log_wls_eff,
        fitted_log,
        yerr=fitted_log_err,
        marker="o",
        mfc="grey",
        mec="grey",
        ms=7,
        ecolor="grey",
        linestyle="None",
        label="Photometric flux\nfrom LC fitting",
    )
    ax2.set_ylabel(r"Log Flux ($\log_{10}(\mathrm{erg \ s^{-1} cm^{-2} \AA^{-1}})$)", fontsize=13)
    ax2.set_xlabel(r"Log Wavelength ($\log_{10}(\mathrm{\AA})$)", fontsize=13)
    ax2.legend(ncol=1, fontsize=10, loc="upper right")
    plt.subplots_adjust(hspace=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True
