"""Early rising LC photometry loader / clipper for notebook 2 (Bazin extend)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt

from lc_bazin_fit_config import BazinFitConfig
from lc_bazin_models import bazin_forced_zero_factory

__all__ = ["EarlyLCPhotometry"]


class EarlyLCPhotometry:
    """Load, clip, and early-extend dust-corrected photometry for one SN."""

    def __init__(
        self,
        lc_path: str,
        snname: str,
        exclude_filt: tuple[str, ...] | list[str] = (),
        verbose: bool = False,
        dataspec_path: str | None = None,
    ):
        self.lc_data_path = lc_path + ("" if lc_path.endswith("/") else "/")
        self.snname = snname
        self.exclude_filt = set(exclude_filt)
        self.verbose = verbose
        self.dataspec_path = dataspec_path
        self.set_data_directory(verbose)

    def set_data_directory(self, verbose: bool) -> None:
        SNphotometry_PATH = os.path.join(self.lc_data_path, "%s.dat" % self.snname)
        try:
            if verbose:
                print("Looking for Photometry for %s in%s" % (self.snname, SNphotometry_PATH))
            if os.path.isfile(SNphotometry_PATH):
                if verbose:
                    print("Got it!")
                self.sn_rawphot_file = SNphotometry_PATH
            elif not os.path.isdir(self.lc_data_path):
                print("I cant find the directory with photometry. Check %s" % self.lc_data_path)
            else:
                print("I cant find the file with photometry. Check %s" % SNphotometry_PATH)
        except Exception as e:
            print(e)

    def _spec_root(self) -> str:
        if self.dataspec_path is not None:
            return self.dataspec_path
        import pipeline_config as pconf

        return pconf.bootstrap_runtime().dataspec_path

    def load(self, verbose: bool = False) -> None:
        if verbose:
            print("Loading %s" % self.sn_rawphot_file)
        try:
            lc_file = np.genfromtxt(
                self.sn_rawphot_file,
                dtype=None,
                encoding="utf-8",
                names=True,
                delimiter=",",
            )
            mask_filt = np.array([f not in self.exclude_filt for f in lc_file["band"]])
            lc_no_badfilters = lc_file[mask_filt]
            mask_filt = np.array([~np.isnan(f) for f in lc_no_badfilters["Flux"]])
            self.phot = lc_no_badfilters[mask_filt]
            self.avail_filters = np.unique(self.phot["band"])
            self.phot_extended = self.phot
            print("Photometry loaded")
        except Exception as e:
            print(e)
            print("Are you sure you gave me the right format? Check documentation in case.")

    def get_availfilter(self, verbose: bool = False) -> np.ndarray:
        if (not hasattr(self, "phot")) or (not hasattr(self, "avail_filters")):
            self.load()
        return self.avail_filters

    def extend_photometry(self) -> None:
        if hasattr(self, "clipped_phot"):
            self.phot_extended = self.clipped_phot
        else:
            self.clip_photometry()
            self.phot_extended = self.clipped_phot

    def get_singlefilter(
        self,
        single_filter: str,
        extended_clipped: bool = False,
        verbose: bool = False,
    ):
        if not hasattr(self, "phot"):
            self.load()
        if not isinstance(single_filter, str):
            print("Single filter string please")
            return None
        if single_filter not in self.avail_filters:
            print("Looks like the filter you are looking for is not available")
            return None
        if extended_clipped:
            if not hasattr(self, "phot_extended"):
                self.extend_photometry()
            filt_index = self.phot_extended["band"] == single_filter
            return self.phot_extended[filt_index]
        filt_index = self.phot["band"] == single_filter
        return self.phot[filt_index]

    def get_mjdpeak(self, verbose: bool = False) -> float:
        if not hasattr(self, "phot"):
            self.load()
        mjd_peaks_list = []
        for f in self.avail_filters:
            phot_perfilt = self.get_singlefilter(f)
            mjd_peak = phot_perfilt["MJD"][np.argmax(phot_perfilt["Flux"])]
            mjd_peaks_list.append(mjd_peak)
        return float(np.min(mjd_peaks_list))

    def clip_LC_filter(self, filter_name, clipping_mjd_delta=0.01, pre_bump=False):
        def clip_one_point(
            mjd_unclipped,
            flux_unclipped,
            fluxerr_unclipped,
            filtset_unclipped,
            instr_unclipped,
            clipping_index,
        ):
            mjd_tbc = np.array([mjd_unclipped[clipping_index], mjd_unclipped[clipping_index + 1]])
            flux_tbc = np.array([flux_unclipped[clipping_index], flux_unclipped[clipping_index + 1]])
            flux_err_tbc = np.array(
                [fluxerr_unclipped[clipping_index], fluxerr_unclipped[clipping_index + 1]]
            )
            mjd_avg = np.average(mjd_tbc)
            flux_avg, sum_w = np.average(flux_tbc, weights=1.0 / (flux_err_tbc) ** 2, returned=True)
            flux_err_avg = max([np.std(flux_tbc), np.sqrt(1.0 / sum_w)])
            clipped_mjd_sorted = np.delete(mjd_unclipped, clipping_index)
            clipped_flux_sorted = np.delete(flux_unclipped, clipping_index)
            clipped_flux_err_sorted = np.delete(fluxerr_unclipped, clipping_index)
            clipped_filtset_sorted = np.delete(filtset_unclipped, clipping_index)
            clipped_instr_sorted = np.delete(instr_unclipped, clipping_index)
            clipped_mjd_sorted[clipping_index] = mjd_avg
            clipped_flux_sorted[clipping_index] = flux_avg
            clipped_flux_err_sorted[clipping_index] = flux_err_avg
            return (
                clipped_mjd_sorted,
                clipped_flux_sorted,
                clipped_flux_err_sorted,
                clipped_filtset_sorted,
                clipped_instr_sorted,
            )

        LC_filt = self.get_singlefilter(filter_name)
        mjd_sorted = np.sort(LC_filt["MJD"])
        flux_sorted = LC_filt["Flux"][np.argsort(LC_filt["MJD"])]
        flux_err_sorted = LC_filt["Flux_err"][np.argsort(LC_filt["MJD"])]
        filtset_sorted = LC_filt["FilterSet"][np.argsort(LC_filt["MJD"])]
        instr_sorted = LC_filt["Instr"][np.argsort(LC_filt["MJD"])]

        if pre_bump:
            mask_bump = mjd_sorted > min(mjd_sorted) + 10.0
            new_mjd_sorted = np.array([round(m, 2) for m in mjd_sorted])[mask_bump]
            new_flux_sorted = np.copy(flux_sorted)[mask_bump]
            new_flux_err_sorted = np.copy(flux_err_sorted)[mask_bump]
            new_filtset_sorted = np.copy(filtset_sorted)[mask_bump]
            new_Instr_sorted = np.copy(instr_sorted)[mask_bump]
        else:
            new_mjd_sorted = np.array([round(m, 2) for m in mjd_sorted])
            new_flux_sorted = np.copy(flux_sorted)
            new_flux_err_sorted = np.copy(flux_err_sorted)
            new_filtset_sorted = np.copy(filtset_sorted)
            new_Instr_sorted = np.copy(instr_sorted)

        while len(np.where(np.abs(new_mjd_sorted[:-1] - new_mjd_sorted[1:]) < clipping_mjd_delta)[0]) >= 1:
            tbc_indexes = np.where(np.abs(new_mjd_sorted[:-1] - new_mjd_sorted[1:]) < clipping_mjd_delta)[0]
            ind = tbc_indexes[0]
            R = clip_one_point(
                new_mjd_sorted,
                new_flux_sorted,
                new_flux_err_sorted,
                new_filtset_sorted,
                new_Instr_sorted,
                ind,
            )
            new_mjd_sorted, new_flux_sorted, new_flux_err_sorted, new_filtset_sorted, new_Instr_sorted = R

        if pre_bump:
            new_mjd_sorted = np.concatenate([mjd_sorted[~mask_bump], new_mjd_sorted])
            new_flux_sorted = np.concatenate([flux_sorted[~mask_bump], new_flux_sorted])
            new_flux_err_sorted = np.concatenate([flux_err_sorted[~mask_bump], new_flux_err_sorted])
            new_filtset_sorted = np.concatenate([filtset_sorted[~mask_bump], new_filtset_sorted])
            new_Instr_sorted = np.concatenate([instr_sorted[~mask_bump], new_Instr_sorted])

        new_filter_sorted = np.full(len(new_mjd_sorted), filter_name, dtype="|S20")
        new_LC = []
        for i in zip(
            new_mjd_sorted,
            new_filter_sorted,
            new_flux_sorted,
            new_flux_err_sorted,
            new_filtset_sorted,
            new_Instr_sorted,
        ):
            new_LC.append(i)
        new_LC = np.array(new_LC, LC_filt.dtype)
        print(filter_name, "Before clipping %i, after %i" % (len(mjd_sorted), len(new_LC)))
        return new_LC

    def clip_photometry(self, pre_bump=False, verbose=False):
        if not hasattr(self, "phot"):
            self.load()
        filt_avail = self.avail_filters
        clipping_mjd_delta = 0.01
        LC_clipped = np.array([], self.phot.dtype)
        for ff in filt_avail:
            LC_xfilter = self.clip_LC_filter(ff, clipping_mjd_delta, pre_bump=pre_bump)
            LC_clipped = np.concatenate([LC_clipped, LC_xfilter])
        self.clipped_phot = LC_clipped
        self.phot_extended = LC_clipped
        return None

    def get_spec_mjd(self, verbose=False) -> np.ndarray:
        phase_list_file = self._spec_root() + "2_spec_lists_smoothed/" + self.snname + ".list"
        try:
            parse_phase = np.genfromtxt(phase_list_file, dtype=None, encoding="utf-8")
            return parse_phase["f0"]
        except Exception as exc:
            raise Exception(
                " WARNING \n I looked into %s and I found NO spectra? Ooops" % phase_list_file
            ) from exc

    def get_spec_list(self, verbose=False) -> np.ndarray:
        phase_list_file = self._spec_root() + "2_spec_lists_smoothed/" + self.snname + ".list"
        try:
            parse_phase = np.genfromtxt(phase_list_file, dtype=None, encoding="utf-8")
            return parse_phase["f2"]
        except Exception:
            print("I looked into %s and I found NO spectra? Ooops" % phase_list_file)
            return np.array([])

    def fit_early_extend_bazin(
        self,
        filt: str,
        config: BazinFitConfig,
        min_mjd: float | None = None,
        max_mjd: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Numeric rising Bazin extend; returns MJD grid, flux, err, and plot_data."""
        LC_filt_extended = self.get_singlefilter(filt, extended_clipped=True)
        mjd_unsorted = LC_filt_extended["MJD"]
        mjd = np.sort(mjd_unsorted)
        orig_flux = LC_filt_extended["Flux"][np.argsort(mjd_unsorted)]
        orig_flux_err = LC_filt_extended["Flux_err"][np.argsort(mjd_unsorted)]

        offset_time = mjd[np.argmax(orig_flux)]
        t = mjd - offset_time
        t_spectra = self.get_spec_mjd() - offset_time
        max_fl = max(orig_flux)
        fl = orig_flux / max_fl
        flerr = orig_flux_err / max_fl

        mask_rising = t <= t[np.argmax(fl)]
        t_rise = t[mask_rising]
        fl_rise = fl[mask_rising]
        flerr_rise = flerr[mask_rising]

        mjd_last_phot_point = min(t_rise)
        min_phase = -25.0 if min_mjd is None else float(min_mjd) - offset_time
        new_t = np.arange(min_phase, min(t) - 1.0, 1.0)

        t0_fix, _, _ = config.explosion_dates[self.snname]
        t_exp = t0_fix - offset_time
        bazin_forced_zero = bazin_forced_zero_factory(t_exp)

        p_a = max(fl_rise)
        p_t0 = t_rise[np.argmax(fl_rise)]
        p_t_fall = 30.0
        p_t_rise = 5.0
        p0_bazin = [p_a, p_t0, p_t_fall, p_t_rise]
        bounds_bazin = (
            [0, min(t_rise) - 20, 5.0, 0.1],
            [np.inf, max(t_rise) + 20, 100.0, 50.0],
        )

        res = np.array(p0_bazin, dtype=float)
        try:
            res, _cov = opt.curve_fit(
                bazin_forced_zero,
                t_rise,
                fl_rise,
                p0=p0_bazin,
                sigma=flerr_rise,
                absolute_sigma=True,
                bounds=bounds_bazin,
                maxfev=10000,
            )
            print("Bazin params [A, t0, t_fall, t_rise]: %s" % res)
            fit = max_fl * bazin_forced_zero(new_t, *res)
        except RuntimeError:
            print("Bazin fit failed to converge for " + filt)
            fit = np.zeros(len(new_t))

        fit_err = np.zeros(len(fit))
        plot_data = {
            "t": t,
            "fl": fl,
            "flerr": flerr,
            "t_rise": t_rise,
            "fl_rise": fl_rise,
            "flerr_rise": flerr_rise,
            "new_t": new_t,
            "fit_norm": fit / max_fl if max_fl else fit,
            "fit": fit,
            "fit_err": fit_err,
            "orig_flux": orig_flux,
            "orig_flux_err": orig_flux_err,
            "mask_rising": mask_rising,
            "t_exp": t_exp,
            "offset_time": offset_time,
            "max_fl": max_fl,
            "filt": filt,
            "snname": self.snname,
            "bazin_params": res,
            "t_spectra": t_spectra,
            "mjd_last_phot_point": mjd_last_phot_point,
        }
        new_mjd = new_t + offset_time
        if max_mjd is not None:
            new_mjd = new_mjd[new_mjd <= max_mjd]
            fit = fit[: len(new_mjd)]
            fit_err = fit_err[: len(new_mjd)]
        return new_mjd, fit, fit_err, plot_data

    def LC_early_extend_xfilter(
        self,
        filt: str,
        minMJD=None,
        maxMJD=None,
        config: BazinFitConfig | None = None,
    ):
        if config is None:
            raise ValueError("config (BazinFitConfig) is required for LC_early_extend_xfilter")
        new_mjd, fit, fit_err, _plot_data = self.fit_early_extend_bazin(
            filt, config, min_mjd=minMJD, max_mjd=maxMJD
        )
        return new_mjd, fit, fit_err

    def create_extended_LC(
        self,
        filters_to_fit=None,
        output_path: str | None = None,
        config: BazinFitConfig | None = None,
        name_file=None,
    ):
        if config is None:
            raise ValueError("config (BazinFitConfig) is required for create_extended_LC")
        if filters_to_fit is None:
            filters_to_fit = [
                "DECam_i",
                "DECam_z",
                "Swope_i",
                "FLAMINGOS-2_Ks",
                "FourStar_H",
                "FourStar_J",
                "FourStar_Ks",
                "GFC_i",
                "GFC_y",
                "GFC_z",
                "HSC_z",
                "SIRIUS_H",
                "SIRIUS_J",
                "SIRIUS_Ks",
                "Sinistro_g",
                "Sinistro_r",
                "Skymapper_r",
                "UVOT_M2",
                "UVOT_U",
                "UVOT_W1",
                "VISTA_J",
                "VISTA_Ks",
                "VISTA_Y",
            ]

        lc_file = pd.DataFrame(
            np.genfromtxt(
                self.sn_rawphot_file,
                names=["MJD", "band", "Flux", "Flux_err", "FilterSet", "Source"],
                usecols=[0, 1, 2, 3, 4, 5],
                dtype=None,
                encoding="utf-8",
            )
        )

        for ff in filters_to_fit:
            print("Extrapolating early-time points for: %s" % ff)
            extr_pts_pd = pd.DataFrame().reindex_like(lc_file)[:0]
            mjd_new, fit, fit_err = self.LC_early_extend_xfilter(ff, config=config)
            extr_pts_pd["MJD"] = mjd_new
            extr_pts_pd["band"] = np.full(len(mjd_new), fill_value=ff)
            extr_pts_pd["Flux"] = fit
            extr_pts_pd["Flux_err"] = fit_err
            extr_pts_pd["FilterSet"] = np.full(len(mjd_new), fill_value="SUDO_PTS")
            extr_pts_pd["Source"] = np.full(len(mjd_new), fill_value="SUDO_PTS")
            lc_file = pd.concat([lc_file, extr_pts_pd], ignore_index=True)

        t0_fix, _, _ = config.explosion_dates[self.snname]
        lc_file["Phase"] = lc_file["MJD"] - t0_fix
        lc_file = lc_file[lc_file["Phase"] >= 0].copy()
        lc_file.loc[lc_file["Phase"] == 0, "Phase"] = 1e-5
        lc_file = lc_file[lc_file["Flux"] >= 0].copy()
        lc_file.loc[lc_file["Flux"] == 0, "Flux"] = 1e-25
        lc_file["Log_Phase"] = np.log10(lc_file["Phase"])
        lc_file["Log_Flux"] = np.log10(lc_file["Flux"])
        lc_file["Log_Flux_err"] = lc_file["Flux_err"] / (lc_file["Flux"] * np.log(10))

        if output_path is not None:
            out = os.path.join(output_path, "%s.dat" % self.snname)
            lc_file.to_csv(out, na_rep="nan", index=False, sep="\t")
            print("Saved:", out)
        return lc_file
