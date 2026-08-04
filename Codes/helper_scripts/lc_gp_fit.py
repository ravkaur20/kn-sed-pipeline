"""Log-space LC GP fit (pipeline step 3)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import george
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from george.kernels import Matern32Kernel
from scipy import optimize as opt

import pipeline_config as pconf
from explosion_anchor_utils import append_explosion_anchor_row
from lc_gp_kernels import load_kernel_settings_file, resolve_kernel_for_band

class SNPhotometryClass():
    """Class with photometry for each object:
            - load the photometry from the DATA folder
            - get the phootmetry in each filter
            - plot the raw photometry 
            - fit the photometry using GP
    """
    
    def __init__(
        self,
        snname: str,
        *,
        datalc_path: str,
        dataspec_path: str,
        output_dir: str,
        exclude_filt: list[str] | None = None,
        kernel_settings: dict[str, dict[str, dict[str, Any]]] | None = None,
        anchor_t0_in_lc_gp: bool = True,
        verbose: bool = False,
    ):
        """
        """
        ## Initialise the class variables
        self.lc_data_path = datalc_path.rstrip(os.sep) + os.sep
        self.dataspec_path = dataspec_path.rstrip(os.sep) + os.sep
        self.output_dir = output_dir.rstrip(os.sep) + os.sep
        self.exclude_filt = list(exclude_filt or [])
        self.kernel_settings = kernel_settings or {}
        self.anchor_t0_in_lc_gp = bool(anchor_t0_in_lc_gp)
        self.snname = snname   
        self.set_data_directory(verbose)

    def set_data_directory(self, verbose):
        """
        Set a new data directory path.
        Enables the data directory to be changed by the user.
        """
        SNphotometry_PATH = os.path.join(self.lc_data_path, "%s.dat" % self.snname)
        
        try:
            if verbose: print('Looking for Photometry for %s in%s'%(self.snname, SNphotometry_PATH))
            if os.path.isfile(SNphotometry_PATH):
                if verbose: print ('Got it!')
                self.sn_rawphot_file = SNphotometry_PATH
                pass
            else:
                if not os.path.isdir(self.lc_data_path):
                    print ('I cant find the directory with photometry. Check %s'%self.lc_data_path)
                    pass
                else: 
                    print ('I cant find the file with photometry. Check %s'%SNphotometry_PATH)
                    pass
    
        except Exception as e:
            print (e)

    def load(self, verbose = False):
        """
        Loads a single photometry file.
        with ('MJD', 'Flux', 'Flux_err', 'band')
        
        Parameters
        - verbose
        ----------
        Returns
        - photometry in all filters
        -------
        """
        if verbose: print('Loading %s'%self.sn_rawphot_file)
        try:
            lc_file = np.genfromtxt(self.sn_rawphot_file, dtype=None, encoding="utf-8", 
                                    names=True)
            mask_filt = np.array([f not in self.exclude_filt for f in lc_file['band']])
            lc_no_badfilters = lc_file[mask_filt]
            mask_filt = np.array([~np.isnan(f) for f in lc_no_badfilters['Flux']])
            self.phot = lc_no_badfilters[mask_filt]
            
            self.avail_filters = np.unique(self.phot['band'])

            self.clipped_phot = self.phot
            print ('Photometry loaded')

            
        except Exception as e:
            print (e)
            print ('Are you sure you gave me the right format? Check documentation in case.')

    def get_availfilter(self, verbose = False):
        """
        get available filter for this SN
        """
        if (not hasattr(self, "phot"))|(not hasattr(self, "avail_filters")):
            self.load()
        return self.avail_filters
        
    def get_singlefilter(self, single_filter, extended_clipped = False, verbose = False):
        """
        Loads from photometry file just 1 filter photometry.
        with ('MJD', 'Flux', 'Flux_err', 'band')
        """
        if not hasattr(self, "phot"):
            self.load()

        if not (isinstance(single_filter, str)):
            print ('Single filter string please')
            return None
        
        if single_filter not in self.avail_filters:
            print ('Looks like the filter you are looking for is not available')
            return None
        
        if extended_clipped:
            if not hasattr(self, "clipped_phot"):
                self.clip_photometry()
            filt_index = self.clipped_phot['band']==single_filter
            return self.clipped_phot[filt_index] 
        else: 
            filt_index = self.phot['band']==single_filter
            return self.phot[filt_index]
        
    def get_mjdpeak(self, verbose = False):
        """
        Loads from photometry for each filter, measure peak for each filter
        get a rough estimate of the paek_mjd
        """
        if not hasattr(self, "phot"):
            self.load()
            
        mjd_peaks_list=[]
        for f in self.avail_filters:
            phot_perfilt = self.get_singlefilter(f)
            mjd_peak = phot_perfilt['MJD'][np.argmax(phot_perfilt['Flux'])]
            mjd_peaks_list.append(mjd_peak)
            
        return np.min(mjd_peaks_list)
    
    def plot_raw_phot(self, filt_list=None, plt_extended_clipped=False, save_fig=False, \
                      save_format='png', output_path_name=None):
        if not hasattr(self, "phot"):
            self.load()
        if (plt_extended_clipped)&(not hasattr(self, "clipped_phot")):
            self.clip_photometry()

        if filt_list:
            filt_toplot = filt_list
        else:
            filt_toplot = self.avail_filters
        fig = plt.figure(1)
        for f in filt_toplot:
            phot_f = self.get_singlefilter(f, extended_clipped=plt_extended_clipped)
            plt.errorbar(phot_f['MJD'], phot_f['Flux'], phot_f['Flux_err'],\
                         marker='.', linestyle='None', label=f)
        plt.legend(loc='best')
        plt.show()
        if save_fig:
            if output_path_name:
                fig.savefig(output_path_name)
            else:
                fig.savefig(self.sn_rawphot_file.replace('.dat', '_raw_photo.%s'%save_format))
        return None

    def plot_GP_fit_phot(self, filt_list=None, save_fig=False, save_format='png', output_path_name=None):
        if not hasattr(self, "phot"):
            self.load()
        
        if not hasattr(self, "gp"):
            self.load()

        if filt_list:
            filt_toplot = filt_list
        else:
            filt_toplot = self.avail_filters
        fig = plt.figure(1)
        for f in filt_toplot:
            phot_f = self.get_singlefilter(f)
            plt.errorbar(phot_f['MJD'], phot_f['Flux'], yerr=phot_f['Flux_err'],\
                         marker='.', linestyle='None', label=f)
        plt.legend(loc='best')
        plt.show()
        if save_fig:
            if output_path_name:
                fig.savefig(output_path_name)
            else:
                fig.savefig(self.sn_rawphot_file.replace('.dat', '_raw_photo.%s'%save_format))
        return None

    def clip_LC_filter(self, filter_name, clipping_mjd_delta = 0.5, pre_bump=False):
        
        def clip_one_point(mjd_unclipped, flux_unclipped, fluxerr_unclipped, filtset_unclipped, instr_unclipped, phase_unclipped, logphase_unclipped, logflux_unclipped, logfluxerr_unclipped, clipping_index):
            mjd_tbc = np.array([mjd_unclipped[clipping_index], mjd_unclipped[clipping_index+1]])
            flux_tbc = np.array([flux_unclipped[clipping_index], flux_unclipped[clipping_index+1]])
            flux_err_tbc = np.array([fluxerr_unclipped[clipping_index], fluxerr_unclipped[clipping_index+1]])
            mjd_avg = np.average(mjd_tbc)
            flux_avg, sum_w = np.average(flux_tbc, weights= 1./(flux_err_tbc)**2, returned=True)
            flux_err_avg = max([np.std(flux_tbc), np.sqrt(1./sum_w)])
            
            # Re-calculate log values directly from the new averaged linear values for accuracy
            t0_fix = self.phot['MJD'][0] - self.phot['Phase'][0]
            phase_avg = mjd_avg - t0_fix
            logphase_avg = np.log10(phase_avg) if phase_avg > 0 else -5.0
            logflux_avg = np.log10(flux_avg) if flux_avg > 0 else -25.0
            logfluxerr_avg = flux_err_avg / (flux_avg * np.log(10)) if flux_avg > 0 else 0.1
            
            clipped_mjd_sorted = np.delete(mjd_unclipped, clipping_index)
            clipped_flux_sorted = np.delete(flux_unclipped, clipping_index)
            clipped_flux_err_sorted = np.delete(fluxerr_unclipped, clipping_index)
            clipped_filtset_sorted = np.delete(filtset_unclipped, clipping_index)
            clipped_instr_sorted = np.delete(instr_unclipped, clipping_index)
            clipped_phase_sorted = np.delete(phase_unclipped, clipping_index)
            clipped_logphase_sorted = np.delete(logphase_unclipped, clipping_index)
            clipped_logflux_sorted = np.delete(logflux_unclipped, clipping_index)
            clipped_logfluxerr_sorted = np.delete(logfluxerr_unclipped, clipping_index)
            
            clipped_mjd_sorted[clipping_index] = mjd_avg
            clipped_flux_sorted[clipping_index] = flux_avg
            clipped_flux_err_sorted[clipping_index] = flux_err_avg
            clipped_phase_sorted[clipping_index] = phase_avg
            clipped_logphase_sorted[clipping_index] = logphase_avg
            clipped_logflux_sorted[clipping_index] = logflux_avg
            clipped_logfluxerr_sorted[clipping_index] = logfluxerr_avg
            
            return clipped_mjd_sorted, clipped_flux_sorted, clipped_flux_err_sorted, clipped_filtset_sorted, clipped_instr_sorted, clipped_phase_sorted, clipped_logphase_sorted, clipped_logflux_sorted, clipped_logfluxerr_sorted
    
        LC_filt = self.get_singlefilter(filter_name)
        mjd_sorted = np.sort(LC_filt['MJD'])
        sort_idx = np.argsort(LC_filt['MJD'])
        
        flux_sorted = LC_filt['Flux'][sort_idx]
        flux_err_sorted = LC_filt['Flux_err'][sort_idx]
        filtset_sorted = LC_filt['FilterSet'][sort_idx]
        instr_sorted = LC_filt['Instr'][sort_idx]
        phase_sorted = LC_filt['Phase'][sort_idx]
        logphase_sorted = LC_filt['Log_Phase'][sort_idx]
        logflux_sorted = LC_filt['Log_Flux'][sort_idx]
        logfluxerr_sorted = LC_filt['Log_Flux_err'][sort_idx]
        
        if pre_bump:
            mask_bump = mjd_sorted>min(mjd_sorted)+10.
            new_mjd_sorted = np.array([round(m,2) for m in mjd_sorted])[mask_bump]
            double = np.where(np.abs(new_mjd_sorted[:-1]-new_mjd_sorted[1:])<clipping_mjd_delta)
            new_flux_sorted = np.copy(flux_sorted)[mask_bump]
            new_flux_err_sorted = np.copy(flux_err_sorted)[mask_bump]
            new_filtset_sorted = np.copy(filtset_sorted)[mask_bump]
            new_Instr_sorted = np.copy(instr_sorted)[mask_bump]
            new_phase_sorted = np.copy(phase_sorted)[mask_bump]
            new_logphase_sorted = np.copy(logphase_sorted)[mask_bump]
            new_logflux_sorted = np.copy(logflux_sorted)[mask_bump]
            new_logfluxerr_sorted = np.copy(logfluxerr_sorted)[mask_bump]
        else:
            new_mjd_sorted = np.array([round(m,2) for m in mjd_sorted])
            double = np.where(np.abs(new_mjd_sorted[:-1]-new_mjd_sorted[1:])<clipping_mjd_delta)
            new_flux_sorted = np.copy(flux_sorted)
            new_flux_err_sorted = np.copy(flux_err_sorted)
            new_filtset_sorted = np.copy(filtset_sorted) 
            new_Instr_sorted = np.copy(instr_sorted) 
            new_phase_sorted = np.copy(phase_sorted)
            new_logphase_sorted = np.copy(logphase_sorted)
            new_logflux_sorted = np.copy(logflux_sorted)
            new_logfluxerr_sorted = np.copy(logfluxerr_sorted)
            
        while len(np.where(np.abs(new_mjd_sorted[:-1]-new_mjd_sorted[1:])<clipping_mjd_delta)[0])>=1:
            tbc_indexes = np.where(np.abs(new_mjd_sorted[:-1]-new_mjd_sorted[1:])<clipping_mjd_delta)[0]
            ind = tbc_indexes[0]
            R = clip_one_point(new_mjd_sorted, new_flux_sorted, new_flux_err_sorted, new_filtset_sorted, new_Instr_sorted, new_phase_sorted, new_logphase_sorted, new_logflux_sorted, new_logfluxerr_sorted, ind)
            new_mjd_sorted, new_flux_sorted, new_flux_err_sorted, new_filtset_sorted, new_Instr_sorted, new_phase_sorted, new_logphase_sorted, new_logflux_sorted, new_logfluxerr_sorted = R

        if pre_bump:
            new_mjd_sorted = np.concatenate([mjd_sorted[~mask_bump], new_mjd_sorted])
            new_flux_sorted = np.concatenate([flux_sorted[~mask_bump], new_flux_sorted])
            new_flux_err_sorted = np.concatenate([flux_err_sorted[~mask_bump], new_flux_err_sorted])
            new_filtset_sorted = np.concatenate([filtset_sorted[~mask_bump], new_filtset_sorted])      
            new_Instr_sorted = np.concatenate([instr_sorted[~mask_bump], new_Instr_sorted])      
            new_phase_sorted = np.concatenate([phase_sorted[~mask_bump], new_phase_sorted])
            new_logphase_sorted = np.concatenate([logphase_sorted[~mask_bump], new_logphase_sorted])
            new_logflux_sorted = np.concatenate([logflux_sorted[~mask_bump], new_logflux_sorted])
            new_logfluxerr_sorted = np.concatenate([logfluxerr_sorted[~mask_bump], new_logfluxerr_sorted])

        new_filter_sorted = np.full(len(new_mjd_sorted), filter_name, dtype='|S20')
        new_LC=[]
        for i in zip(new_mjd_sorted, new_filter_sorted, new_flux_sorted, new_flux_err_sorted, new_filtset_sorted, new_Instr_sorted, new_phase_sorted, new_logphase_sorted, new_logflux_sorted, new_logfluxerr_sorted):
            new_LC.append(i)
        new_LC = np.array(new_LC, LC_filt.dtype)
        
        print (filter_name, 'Before clipping %i, after %i'%(len(mjd_sorted), len(new_LC)))
        return new_LC
    
    def clip_photometry(self, pre_bump=False, verbose = False):
        if not hasattr(self, "phot"):
            self.load()
        
        filt_avail = self.avail_filters
        
        clipping_mjd_delta = 0.01
        LC_clipped = np.array([], self.phot.dtype) 
        for ff in filt_avail:
            LC_xfilter = self.clip_LC_filter(ff, clipping_mjd_delta, pre_bump=pre_bump)
            LC_clipped = np.concatenate([LC_clipped, LC_xfilter])
        self.clipped_phot = LC_clipped
        return None
    
    def get_spec_mjd(self, verbose=False):
        phase_list_file = self.dataspec_path + "2_spec_lists_prescaled/" + self.snname + ".list"
        try:
            parse_phase = np.genfromtxt(phase_list_file, dtype=None, encoding="utf-8")
            return parse_phase["f0"]
        except Exception:
            print("I looked into %s and I found NO spectra? Ooops" % phase_list_file)
            return np.array([])
            
    # NEW FUNCTION: Mirrors get_spec_mjd but converts the spectra dates to Log_Phase
    def get_spec_log_phase(self, verbose=False):
        spec_mjds = self.get_spec_mjd(verbose)
        if len(spec_mjds) == 0: 
            return np.array([])
        t0_fix = self.phot['MJD'][0] - self.phot['Phase'][0]
        spec_phases = spec_mjds - t0_fix
        spec_phases[spec_phases <= 0] = 1e-5
        return np.log10(spec_phases)
        
    def get_spec_list(self, verbose=False):
        #phase_list_file = self.dataspec_path + '/2_spec_lists_smoothed/' + self.snname+'.list'
        #Rav changed this for prescaling spectra:
        phase_list_file = self.dataspec_path + "2_spec_lists_prescaled/" + self.snname + ".list"
        try: 
            parse_phase = np.genfromtxt(phase_list_file, dtype=None,encoding="utf-8")
            
            return parse_phase['f2']
        except: 
            print ('I looked into %s and I found NO spectra? Ooops'%phase_list_file)
            return np.array([])

    def LCfit_withGP_xfilter(self, filt, minLogPhase=None, maxLogPhase=None):
        if not hasattr(self, "phot"):
            self.load()

        def ll(p):
            gp.set_parameter_vector(p)
            return -gp.lnlikelihood(flux_norm, quiet=False)#

        def grad_ll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_lnlikelihood(flux_norm, quiet=False)
                        
        mjd_peak = self.get_mjdpeak()
        
        # MAPPED TO LOG SPACE
        log_phase_spectra = self.get_spec_log_phase()
        if minLogPhase is None:
            minLogPhase= min([min(log_phase_spectra),np.min(self.clipped_phot['Log_Phase'])])  
        if maxLogPhase is None:
            maxLogPhase= np.max(self.phot['Log_Phase'])
        
        print ('Log_Phase range', maxLogPhase-minLogPhase)
        # Using 0.01 step size for log space
        new_log_phase = np.arange(minLogPhase, maxLogPhase, 0.01)

        LC_filt_extended = self.get_singlefilter(filt, extended_clipped=True)
        log_phase = LC_filt_extended['Log_Phase']
        
        orig_flux = (LC_filt_extended['Log_Flux'])
        orig_flux_err = LC_filt_extended['Log_Flux_err']

        # No TRY_LOG needed since data is already in log space
        flux_gp = (LC_filt_extended['Log_Flux'])
        flux_err_gp = LC_filt_extended['Log_Flux_err']

        sudo_pts = LC_filt_extended['FilterSet']=='SUDO_PTS'
        if type(sudo_pts) == bool:
            sudo_pts = LC_filt_extended['FilterSet'] == b'SUDO_PTS'
        sudo_pts = np.asarray(sudo_pts, dtype=bool)

        # LOG-SPACE SHIFT: Log fluxes are negative. Division flips them. Subtraction standardizes them safely.
        norm = np.median(flux_gp)
        flux_norm = flux_gp - norm
        err_flux_norm = flux_err_gp

        if self.anchor_t0_in_lc_gp:
            from explosion_anchor_utils import augment_lc_gp_training_for_t0_anchor
            log_phase, orig_flux, orig_flux_err, sudo_pts, flux_norm, err_flux_norm = (
                augment_lc_gp_training_for_t0_anchor(
                    log_phase,
                    orig_flux,
                    orig_flux_err,
                    sudo_pts,
                    norm,
                    flux_norm,
                    err_flux_norm,
                )
            )
        log_phase_T = np.atleast_2d(log_phase).T
                                
        set_scale, set_optimization, set_mean = resolve_kernel_for_band(
            self.snname, filt, self.kernel_settings
        )
        if set_mean: set_fit_mean = True
        else: set_fit_mean=False
            
        k= np.var(flux_norm)* Matern32Kernel(set_scale)
        gp = george.GP(k, mean=set_mean, fit_mean=set_fit_mean)#, 
                       #white_noise=10**-5, fit_white_noise=True)
        
        gp.compute(log_phase_T, err_flux_norm)
        if set_optimization:
            p0=gp.get_parameter_vector()
            results = opt.minimize(ll, p0, jac=grad_ll)
        print ('results ',filt, np.exp(gp.get_parameter_vector()))
        
        mu_gp, cov = gp.predict(flux_norm, new_log_phase)
        std_gp = np.sqrt(np.diag(cov))
        
        # LOG-SPACE UNSHIFT: Restore the median we subtracted earlier
        mu = mu_gp + norm
        std = std_gp
        
        mu_mjdspec, cov_mjdspec = gp.predict(flux_norm, log_phase_spectra)
        std_mjdspec = np.sqrt(np.diag(cov_mjdspec))
        mu_mjdspec = (mu_mjdspec + norm)
        std_mjdspec = std_mjdspec

        self.fitted_phot[filt] = {'clipped_extended_data': [log_phase, orig_flux, orig_flux_err, sudo_pts],
                                  'fit_highcadence':  [new_log_phase, mu, std],
                                 'fit_mjdspec':  [log_phase_spectra, mu_mjdspec, std_mjdspec]}
        return None

    def LCfit_withGP(self, minLogPhase=None, maxLogPhase=None):
        if not hasattr(self, "phot"):
            self.load()

        if not hasattr(self, "fitted_phot"):
            print ('Computing GP fit (for the first time)')
            self.fitted_phot = dict(zip(self.avail_filters, np.zeros(len(self.avail_filters))))
        else:
            print ('Forcing to do GP fit again')
            self.fitted_phot = dict(zip(self.avail_filters, np.zeros(len(self.avail_filters))))
        
        for f in self.avail_filters:
            self.LCfit_withGP_xfilter(f, minLogPhase=minLogPhase, maxLogPhase=maxLogPhase)
        return None

    def create_results_folder(self):
        results_directory = os.path.join(self.output_dir.rstrip(os.sep), self.snname) + os.sep
        os.makedirs(results_directory, exist_ok=True)
        self.results_mainpath = results_directory

    def mangling_GPfile(self, name_file=None):
        if not hasattr(self, "results_mainpath"):
            self.create_results_folder()

        if not hasattr(self, "fitted_phot"):
            self.LCfit_withGP()
        else:
            print(
                """LC fit already done. I will use the one that it.s available.
If you want to do again the LC fit call the function self.LCfit_withGP() again."""
            )

        _lp0, _, _ = self.fitted_phot[self.avail_filters[0]]["fit_mjdspec"]
        columns = {
            "spec_file": self.get_spec_list(),
            "spec_mjd": self.get_spec_mjd(),
            "spec_log_phase": self.get_spec_log_phase(),
            "ext_grid_phase": _lp0,
        }

        for f in self.avail_filters:
            min_lp = min(self.get_singlefilter(f, extended_clipped=False)["Log_Phase"])
            max_lp = max(self.get_singlefilter(f, extended_clipped=False)["Log_Phase"])
            log_phase, log_flux, err_log_flux = self.fitted_phot[f]["fit_mjdspec"]
            inrange = (log_phase >= min_lp) & (log_phase <= max_lp)

            columns[f + "_fit_log_flux"] = log_flux
            columns[f + "_fit_log_fluxerr"] = err_log_flux
            columns[f + "_inrange"] = inrange

        fit_result2mangling = pd.DataFrame(columns)

        fit_result2mangling.to_csv(
            self.results_mainpath + "fitted_phot4mangling_%s.dat" % self.snname,
            sep="\t",
            index=False,
        )
        return fit_result2mangling

    def full_fitted_LC_file(self, name_file=None):
        if not hasattr(self, "results_mainpath"):
            self.create_results_folder()

        if not hasattr(self, "fitted_phot"):
            self.LCfit_withGP()
        else:
            print(
                """LC fit already done. I will use the one that it.s available.
If you want to do again the LC fit call the function self.LCfit_withGP() again."""
            )

        log_phases = self.fitted_phot[self.avail_filters[0]]["fit_highcadence"][0]
        columns = {"Log_Phase": log_phases}

        for i in self.avail_filters:
            lc_filt = self.get_singlefilter(i)
            min_lp = min(lc_filt["Log_Phase"])
            max_lp = max(lc_filt["Log_Phase"])
            log_flux = self.fitted_phot[i]["fit_highcadence"][1]
            log_flux_err = self.fitted_phot[i]["fit_highcadence"][2]

            mask_out = (log_phases <= min_lp) | (log_phases >= max_lp)
            log_flux[mask_out] = np.nan
            log_flux_err[mask_out] = np.nan

            columns[i + "_log_flux"] = log_flux
            columns[i + "_log_flux_err"] = log_flux_err

        df = pd.DataFrame(columns)

        df.to_csv(
            self.results_mainpath + "fitted_phot_logspace_%s.dat" % self.snname,
            sep="\t",
            index=False,
            na_rep="nan",
        )
        return df


@dataclass
class LCGPFitResult:
    snname: str
    output_dir: str
    fitted_phot_logspace_path: str
    fitted_phot4mangling_path: str


def run_lc_gp_fit(
    snname: str,
    *,
    coco_path: str | None = None,
    kernel_settings_path: str | None = None,
    exclude_filt: list[str] | None = None,
    anchor_t0_in_lc_gp: bool | None = None,
    append_t0_row: bool = False,
    verbose: bool = False,
) -> tuple[SNPhotometryClass, LCGPFitResult]:
    """Load LC, clip, GP-fit all bands, write mangling + logspace tables."""
    coco = coco_path or pconf.COCO_PATH
    rt = pconf.bootstrap_runtime(photometry_stage="extrapolated", snname=snname)
    ex = exclude_filt if exclude_filt is not None else list(rt.exclude_filt)
    anchor = (
        bool(anchor_t0_in_lc_gp)
        if anchor_t0_in_lc_gp is not None
        else bool(pconf.ANCHOR_T0_IN_LC_GP)
    )
    ksettings: dict[str, dict[str, dict[str, Any]]] = {}
    if kernel_settings_path and os.path.isfile(kernel_settings_path):
        ksettings = load_kernel_settings_file(kernel_settings_path)

    sn = SNPhotometryClass(
        snname,
        datalc_path=rt.datalc_path,
        dataspec_path=rt.dataspec_path,
        output_dir=rt.output_dir,
        exclude_filt=ex,
        kernel_settings=ksettings,
        anchor_t0_in_lc_gp=anchor,
        verbose=verbose,
    )
    sn.load(verbose=verbose)
    sn.get_availfilter()
    sn.clip_photometry()
    sn.LCfit_withGP()
    df = sn.full_fitted_LC_file()
    sn.mangling_GPfile()

    log_path = os.path.join(
        sn.results_mainpath, "fitted_phot_logspace_%s.dat" % snname
    )
    if append_t0_row and os.path.isfile(log_path):
        append_explosion_anchor_row(log_path)

    mang_path = os.path.join(
        sn.results_mainpath, "fitted_phot4mangling_%s.dat" % snname
    )
    result = LCGPFitResult(
        snname=snname,
        output_dir=sn.results_mainpath,
        fitted_phot_logspace_path=log_path,
        fitted_phot4mangling_path=mang_path,
    )
    return sn, result
