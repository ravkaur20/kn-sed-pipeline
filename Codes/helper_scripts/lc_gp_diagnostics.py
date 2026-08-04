"""Diagnostic plots for LC GP fit (step 3)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _masked_gp_curve(new_log_phase, mu, std):
    """Keep only finite GP samples (NaN outside per-band export window)."""
    valid = np.isfinite(mu) & np.isfinite(std)
    return new_log_phase[valid], mu[valid], std[valid]


def save_plot_gpfit(
    sn,
    *,
    color_dict: dict[str, str],
    mark_dict: dict[str, str],
    save_fig_output: str | None = None,
) -> str:
    if not hasattr(sn, "results_mainpath"):
        sn.create_results_folder()

    max_flux = [-np.inf]
    fig = plt.figure(figsize=(14, 6))
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    for f in sn.avail_filters:
        c = color_dict[f]
        log_phase, log_flux, err_log_flux, sudo_mask = sn.fitted_phot[f][
            "clipped_extended_data"
        ]
        new_log_phase, mu, std = sn.fitted_phot[f]["fit_highcadence"]
        plot_phase, plot_mu, plot_std = _masked_gp_curve(new_log_phase, mu, std)
        if "swift" in f:
            flabel = f.split("_")[1] + "(Swift)"
        else:
            flabel = f.split("_")[1]
        a = plt.errorbar(
            log_phase[~sudo_mask],
            log_flux[~sudo_mask],
            yerr=err_log_flux[~sudo_mask],
            fmt=mark_dict[f],
            mfc=c,
            ms=5,
            color=c,
            linestyle="None",
            label=flabel,
        )
        plt.errorbar(
            log_phase[sudo_mask],
            log_flux[sudo_mask],
            yerr=err_log_flux[sudo_mask],
            fmt=mark_dict[f],
            mfc="white",
            mec=c,
            ms=5,
            mew=0.5,
            color=c,
            linestyle="None",
        )
        if plot_phase.size:
            plt.plot(plot_phase, plot_mu, color=a[0].get_color())
            plt.fill_between(
                plot_phase,
                plot_mu + plot_std,
                plot_mu - plot_std,
                color=a[0].get_color(),
                alpha=0.1,
            )
        max_flux.append(max(log_flux))

    spec_log_phases = sn.get_spec_log_phase()
    if len(spec_log_phases) > 0:
        plt.vlines(
            spec_log_phases,
            min(sn.clipped_phot["Log_Flux"]) - 0.5,
            max(max_flux) + 0.5,
            lw=0.8,
            linestyle="-",
            color="k",
            label="Spectra",
        )
    try:
        plt.xlim(
            min(sn.clipped_phot["Log_Phase"]) - 0.2,
            max(sn.clipped_phot["Log_Phase"]) + 0.2,
        )
    except Exception:
        pass

    plt.xlabel(r"Log Time ($\log_{10}(\rm{Phase})$)", fontsize=13)
    plt.ylabel(
        r"Log Flux ($\log_{10}(\rm{erg \ s^{-1} cm^{-2} \AA^{-1}})$)", fontsize=13
    )
    plt.ylim(min(sn.clipped_phot["Log_Flux"]) - 1.0, max(max_flux) + 1.0)
    plt.legend(fontsize=13, ncol=2, loc="center right", fancybox=True, framealpha=0.5)
    plt.title(
        sn.snname + " light curve fitting using Gaussian Processes (Log-Space)",
        fontsize=15,
    )
    out = save_fig_output or (
        sn.results_mainpath + "fittedGP_%s_.pdf" % sn.snname
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_plot_gpfit_spec(
    sn,
    *,
    color_dict: dict[str, str],
    mark_dict: dict[str, str],
    save_fig_output: str | None = None,
) -> str:
    if not hasattr(sn, "results_mainpath"):
        sn.create_results_folder()

    fig = plt.figure(figsize=(9, 4))
    max_flux: list[float] = []
    for f in sn.avail_filters:
        c = color_dict[f]
        hc_log_phase, hc_mu, hc_std = sn.fitted_phot[f]["fit_highcadence"]
        plot_phase, plot_mu, _plot_std = _masked_gp_curve(hc_log_phase, hc_mu, hc_std)
        new_log_phase, mu, std = sn.fitted_phot[f]["fit_mjdspec"]
        min_lp = min(sn.get_singlefilter(f, extended_clipped=False)["Log_Phase"]) - 0.1
        max_lp = max(sn.get_singlefilter(f, extended_clipped=False)["Log_Phase"]) + 0.1
        inrange = (new_log_phase >= min_lp) & (new_log_phase <= max_lp)
        a = plt.errorbar(
            new_log_phase[inrange],
            mu[inrange],
            yerr=std[inrange],
            fmt=mark_dict[f],
            mfc=c,
            ms=3,
            elinewidth=0.8,
            color=c,
            linestyle="None",
        )
        if plot_phase.size:
            plt.plot(plot_phase, plot_mu, color=a[0].get_color(), label=f, alpha=0.3)
        hc_mu_valid = plot_mu if plot_phase.size else hc_mu[~np.isnan(hc_mu)]
        if hc_mu_valid.size > 0:
            max_flux.append(float(max(hc_mu_valid)))

    plt.ylim(min(sn.clipped_phot["Log_Flux"]) - 1.0, max(max_flux) + 1.0)
    plt.title(sn.snname + " fitted photometry for mangling (Log-Space)")
    out = save_fig_output or (
        sn.results_mainpath + "fittedGPspec_%s_.pdf" % sn.snname
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_per_filter_gp_plots(
    sn,
    *,
    color_dict: dict[str, str],
    mark_dict: dict[str, str],
    out_dir: str | None = None,
) -> list[str]:
    if not hasattr(sn, "results_mainpath"):
        sn.create_results_folder()
    base = out_dir or os.path.join(sn.results_mainpath, "lc_gp_per_filter")
    os.makedirs(base, exist_ok=True)
    paths: list[str] = []
    for f in sn.avail_filters:
        fig = plt.figure(figsize=(8, 4))
        log_phase, log_flux, err_log_flux, sudo_mask = sn.fitted_phot[f][
            "clipped_extended_data"
        ]
        new_log_phase, mu, std = sn.fitted_phot[f]["fit_highcadence"]
        plot_phase, plot_mu, plot_std = _masked_gp_curve(new_log_phase, mu, std)
        c = color_dict[f]
        plt.errorbar(
            log_phase[~sudo_mask],
            log_flux[~sudo_mask],
            yerr=err_log_flux[~sudo_mask],
            fmt=mark_dict[f],
            mfc=c,
            ms=5,
            color=c,
            linestyle="None",
            label=f,
        )
        if plot_phase.size:
            plt.plot(plot_phase, plot_mu, color=c)
            plt.fill_between(plot_phase, plot_mu - plot_std, plot_mu + plot_std, color=c, alpha=0.15)
        plt.title(f)
        plt.xlabel(r"$\log_{10}$(Phase)")
        plt.ylabel(r"$\log_{10}$(Flux)")
        safe = f.replace("/", "_").replace(" ", "_")
        path = os.path.join(base, "%s.pdf" % safe)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
