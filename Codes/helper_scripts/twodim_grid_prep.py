"""2D GP training grid builder extracted from NB6 ``FullMangledSeries_Class``.

Builds wavelength × phase grids from mangled spectra + GP-fitted photometry for 2D GP
(``GP2dim_utils_iter.run_2DGP_GRID_iter``) and legacy NB6 notebooks.

Paths are resolved via ``output_dir`` (typically ``pipeline_config.outputs_root()``) and
``pipeline_config`` twodim layout helpers.
"""

from __future__ import annotations

import os
import re as _re

import numpy as np
import pandas as pd
from scipy import integrate, interpolate

import pipeline_config as _pconf
import GP2dim_utils as GP2dim
from mangle_spectra_log import _resolve_filter_path


def mangled_filename_to_mjd(filename: str) -> float:
    """Parse MJD from mangled spectrum filenames (NB5/NB6 convention).

    Supports ``<mjd>_mangled_spec.txt``, ``mangled_spec_<mjd>.txt``, and re-mangled variants.
    """
    stem = os.path.basename(str(filename))
    for suffix in (
        "_REmangled_spec.txt",
        "_REmangled_spec_FL.txt",
        "_mangled_spec.txt",
        "_mangled_spec_FL.txt",
    ):
        if stem.endswith(suffix):
            return float(stem[: -len(suffix)])
    m = _re.search(r"mangled_spec_([0-9]+(?:\.[0-9]+)?)", stem)
    if m:
        return float(m.group(1))
    raise ValueError("Cannot parse MJD from mangled spectrum filename %r" % filename)


def default_filters_dir(coco_path: str | None = None) -> str:
    base = (coco_path or _pconf.COCO_PATH).rstrip(os.sep)
    return os.path.join(base, "Inputs", "Filters")


class FullMangledSeries_Class():
    """Class to load and mangle a single spectrum:
    """

    @staticmethod
    def _wls_are_linear_angstrom(wls_arr):
        """Delegates to ``GP2dim.mangled_wls_max_is_linear_angstrom`` (paths cell imports newlog or linear_flux)."""
        return GP2dim.mangled_wls_max_is_linear_angstrom(wls_arr)

    @staticmethod
    def _spec_wls_linear(spec_rec):
        return GP2dim.mangled_wls_linear_angstrom(spec_rec)

    @staticmethod
    def _log10_flux_to_linear(flux_arr):
        return GP2dim.mangled_flux_linear_from_log10(flux_arr)
    
    def __init__(self, snname, t0_fix, type_=None, spec_file=None,
                 mode='extrapolate_spectra', verbose=False, DELTA=50.,
                 output_dir=None, filters_dir=None, twodim_layout='classic',
                 prepare_output_dir=True, mangled_spectra_dir=None,
                 csp_sne=None):
        """
        mode: ``extrapolate_spectra`` (production; extend mode removed)
        """
        self.snname = snname
        self.SNtype = type_
        self.mode = mode
        self.t0_fix = t0_fix
        self.output_dir = (output_dir or _pconf.outputs_root()).rstrip(os.sep) + os.sep
        self.filters_dir = (filters_dir or default_filters_dir()).rstrip(os.sep) + os.sep
        self.twodim_layout = str(twodim_layout)
        self.mangled_spectra_dir = mangled_spectra_dir
        if csp_sne is None:
            self.csp_sne = tuple(getattr(_pconf, "CSP_SNE", ()))
        else:
            self.csp_sne = tuple(csp_sne)

        self.get_mangledspec_list()
        self.DELTA = DELTA
        self.verbose = verbose
        self.plot_grid_rebin = bool(getattr(_pconf, "PLOT_GRID_REBIN", False))
        if prepare_output_dir:
            self.create_extended_spec_folder()
        else:
            self.save_plot_path = None
            self.save_dual_products = True
            self.pipeline_wl_min_a = _pconf.PIPELINE_WL_MIN_A
            self.pipeline_wl_max_a = _pconf.PIPELINE_WL_MAX_A
        
        # --- CHANGED: Point to the new logspace file ---
        self.path_fit_phot = self.output_dir+'/%s/fitted_phot_logspace_%s.dat'%(snname,snname)
        # -----------------------------------------------
        # Default: no dex-span chop. Setting this to 2 caps range to 2 dex in log10(phase) from the
        # *minimum* grid time (~0.1 d -> ~10 d max) — not 200 calendar days. KN needs None for ~25+ d.
        self.extrapolate_log_phase_span_dex = None
        # --- Log10(phase-day) greeding for LC on extrap times only (replaces m % 3 on mjds_grid index) ---
        # All below are in **dex** in log10(phase in days), not linear days.
        self.extrap_phot_cadence_dex_early = 0.04
        # Smaller = more LC at late phase (e.g. log10>1); larger = sparser; tune.
        self.extrap_phot_cadence_dex_late = 0.10
        self.extrap_phot_cadence_log_phase_split = 0.0
        self.extrap_phot_cadence_transition_halfwidth_dex = 0.15
        # Log-spaced early extrapolation: dex span from gmin (replaces 0.1 only).
        self.extrap_phot_early_span_dex = 1.5
        # fill_gaps: sub-threshold linear gap but large log span (args passed into GP2dim)
        self.extrapolate_tiny_linear_gap_min_log_span_dex = 0.15
        self.extrapolate_tiny_linear_gap_max_interior = 200

    def _extrap_phot_cadence_min_sep_dex(self, log_t):
        """Min separation in log10(phase days) for greedy LC placement; early vs late linear blend."""
        e = float(getattr(self, "extrap_phot_cadence_dex_early", 0.04))
        l_ = float(getattr(self, "extrap_phot_cadence_dex_late", 0.12))
        s = float(getattr(self, "extrap_phot_cadence_log_phase_split", 0.0))
        h = float(getattr(self, "extrap_phot_cadence_transition_halfwidth_dex", 0.15))
        t = float(log_t)
        lo_b, hi_b = s - h, s + h
        if t <= lo_b:
            return e
        if t >= hi_b:
            return l_
        span = hi_b - lo_b
        u = 0.0 if span <= 0.0 else (t - lo_b) / span
        u = min(1.0, max(0.0, u))
        return (1.0 - u) * e + u * l_

    def _select_extrap_phot_times(self, mjds_extention):
        """Subsample mjds_extention in log10(phase): dense early, sparser late; always keep first+last.

        This acts only on *extrapolation* times, not on spectrum-column indices in mjds_grid, so
        sparsity is tied to physical phase, not a raw column index m.
        """
        arr = np.sort(np.unique(np.round(np.asarray(mjds_extention, dtype=float), 10)))
        if arr.size == 0:
            return arr
        out = [float(arr[0])]
        last = float(arr[0])
        for t in arr[1:]:
            t = float(t)
            if t - last >= self._extrap_phot_cadence_min_sep_dex(t):
                out.append(t)
                last = t
        last_ex = float(arr[-1])
        if not any(np.isclose(out, last_ex, rtol=0.0, atol=1e-9)):
            out.append(last_ex)
        return np.sort(np.unique(np.round(out, 10)))

    @staticmethod
    def _extrap_time_in_allowed(t, allowed_arr):
        if allowed_arr is None or len(allowed_arr) == 0:
            return False
        return bool(np.any(np.isclose(allowed_arr, float(t), rtol=0.0, atol=1e-9)))

    def get_mangledspec_list(self, verbose=False):
        if self.mangled_spectra_dir:
            mypath = str(self.mangled_spectra_dir).rstrip(os.sep)
        else:
            mypath = self.output_dir + "/%s/mangled_spectra" % self.snname
        onlyfiles = [
            f
            for f in os.listdir(mypath)
            if os.path.isfile(os.path.join(mypath, f))
            and ("mangled_spec" in f)
            and (".txt" in f)
        ]
        self.mangledspec_list = onlyfiles
        self.mangled_file_path = mypath + os.sep
        return onlyfiles
    
    def load_mangledfile(self, file):
        mangled_spec = np.genfromtxt(self.mangled_file_path+file, dtype=None,\
                                     encoding="utf-8", names=['wls', 'flux', 'fluxerr', 'mang_mask'])
        return mangled_spec

          
    def load_manglingfile(self, mjd):
        #if not hasattr(self, "results_mainpath"):
        #    self.check_mangling_file()
        mangling_file = self.output_dir+'/%s/fitted_phot4mangling_%s.dat'%(self.snname,self.snname)
        if not os.path.isfile(mangling_file):
            raise Exception("I need the file with fitted photometry in order to mangle a spectrum")
        else:
            phot4mangling = pd.read_csv(mangling_file, sep='\t')
            #print (phot4mangling)#self.phot4mangling = 
            specmjd= mjd#float(self.spec_file.replace('mangled_spec_','').replace('.txt',''))
# --- CHANGED: Added rtol=0.0 to prevent numpy from scaling the tolerance ---
            self.phot4mangling = phot4mangling[phot4mangling['spec_mjd'] == mjd]
            #print("Searching for MJD:", specmjd)
            #print("Available MJDs:", phot4mangling['spec_mjd'].values)
            #print("Closest match:", phot4mangling['spec_mjd'].iloc[np.argmin(np.abs(phot4mangling['spec_mjd'] - specmjd))])
            if len(self.phot4mangling)<1:
                raise Exception(""" ### ERROR: 
I looked in the file with the PHOTOMETRY for MANGLING 
(i.e. fitted_phot4mangling_SNNAME.dat).
I was loading the photometry to mangle/extend the spectrum you are currently loading
in the GRID. I found NO photometry for it... Maybe you should re run GP fit or check your list of spec.""")

            elif len(self.phot4mangling)>1:
                raise Exception(""" ###  TRICKY ERROR: 
I looked in the file with the PHOTOMETRY for MANGLING 
(i.e. fitted_phot4mangling_SNNAME.dat).
I was loading the photometry to mangle/extend the spectrum you are currently loading
in the GRID. I found two spectra at the exact same MJD and this is problem when
I try to build the TIMExWLS grid with this.
Check the phot4mangling.txt file and check if you're MJDs are correct and 
have the right decimals.""")
# --- CHANGED: Look for the log flux columns ---
            self.avail_filters = [col.replace('_fit_log_flux','') for col in phot4mangling.columns\
                                  if col.endswith('_fit_log_flux')]       
    
    def create_extended_spec_folder(self):
        save_plot_path = _pconf.twodim_extended_base(self.output_dir, self.snname, self.mode)
        if not os.path.exists(save_plot_path):
            os.makedirs(save_plot_path)
        else:
            os.system('rm -rf %s'%save_plot_path)
            os.makedirs(save_plot_path)

        self.save_plot_path = save_plot_path
        self.save_dual_products = True
        self.pipeline_wl_min_a = _pconf.PIPELINE_WL_MIN_A
        self.pipeline_wl_max_a = _pconf.PIPELINE_WL_MAX_A
        self.gp_white_noise = float(_pconf.GP_WHITE_NOISE)
        self.gp_predict_dense_log_phase = False
        self.gp_predict_dense_log_phase_n = 64
        self.gp_2d_anchor_t0 = False
        self.gp_2d_t0_anchor_log_phase = -8.0
        self.gp_2d_t0_anchor_log10_flux_cap = -50.0
        self.gp_2d_t0_anchor_log10_flux_err = 2.0

    def running_mean_std(self, x, y, delta_fix=500.):
        #x = xnan[~np.isnan(ynan)]
        #y = ynan[~np.isnan(ynan)]
        
        total_bins = int((x.max()-x.min())/delta_fix)
        bins = np.linspace(x.min(),x.max(), total_bins)
        try:
            delta = bins[1]-bins[0]
            idx  = np.digitize(x,bins)
            running_median_x = np.array([np.mean(x[idx==k]) for k in np.arange(1,total_bins,1)])
            running_mean = np.array([np.mean(y[idx==k]) for k in np.arange(1,total_bins,1)])
            running_std = np.array([np.std(y[idx==k]) for k in np.arange(1,total_bins,1)])
        except IndexError:
            running_median_x = np.array([np.mean(x)])
            running_mean = np.array([np.mean(y)])
            running_std = np.array([np.std(y)])
            
        clean_running_median_x = np.copy(running_median_x)
        clean_running_median_x[running_mean<0.] = np.nan
        clean_running_mean = np.copy(running_mean)
        clean_running_mean[running_mean<0.] = np.nan
        clean_running_std = np.copy(running_std)
        clean_running_std[running_mean<0.] = np.nan
        
        return clean_running_median_x, clean_running_mean, clean_running_std

    def _load_filter_transmission(self, filter_name):
        filt_root = self.filters_dir.rstrip(os.sep)
        filt_path = _resolve_filter_path(
            filt_root, filter_name, self.snname, self.csp_sne
        )
        return np.genfromtxt(
            filt_path, dtype=None, encoding="utf-8", names=["wls", "flux"]
        )

    def get_filt_transmission(self, filter_name):
        filt_transm = self._load_filter_transmission(filter_name)
        return filt_transm["wls"], filt_transm["flux"]

    def lam_eff(self, filter_name): 
        wls, transmission = self.get_filt_transmission(filter_name)
        linear_eff = (integrate.trapezoid(transmission*wls, wls)/\
            integrate.trapezoid(transmission, wls))
        
        # --- CHANGED: Convert the final answer to log-space (rounded index key) ---
        return float(np.round(np.log10(linear_eff), 8))

    def load_phot_for_extention(self, file, anchor = False):
        data_spec_mangled = self.load_mangledfile(file)
        
        # 1. Get the linear MJD from the filename
        raw_mjd = mangled_filename_to_mjd(file)
        
        # 2. Pass the LINEAR MJD to the lookup function to avoid log-collisions
        self.load_manglingfile(raw_mjd)
        
        # 3. Now calculate the Log_Phase for the rest of the math
        phase = raw_mjd - self.t0_fix
        if phase <= 0: phase = 1e-5
        mjd = np.log10(phase)
        
        all_fitted_phot_list=[]
        # ... (rest of the function stays exactly the same)
        fitted_phot_list=[]
        fitted_photerr_list=[]
        wls_eff=[]
        filters4extention=[]
        for filt in self.avail_filters:
            lam_eff_value = self.lam_eff(filt)
# --- CHANGED: Read the log flux strings ---
            fitted_phot = self.phot4mangling['%s_fit_log_flux'%filt].values
            fitted_phot_err = self.phot4mangling['%s_fit_log_fluxerr'%filt].values
            all_fitted_phot_list.append(fitted_phot[0])
            inrange = self.phot4mangling['%s_inrange'%filt].values
            # lam_eff is log10(λ); mangled ``wls`` are linear Å from the log-mangle save format
            wls_lin_file = self._spec_wls_linear(data_spec_mangled)
            lam_lin = 10.0 ** float(lam_eff_value)
            if (lam_lin > np.max(wls_lin_file)) | (lam_lin < np.min(wls_lin_file)):
                if inrange:
                    if self.verbose: print (filt, lam_eff_value, fitted_phot, mjd)
                    fitted_phot_list.append(fitted_phot[0])
                    fitted_photerr_list.append(fitted_phot_err[0])
                    wls_eff.append(lam_eff_value)
                    filters4extention.append(filt)

        fitted_phot_list=np.array(fitted_phot_list)[np.argsort(wls_eff)]
        fitted_photerr_list = np.array(fitted_photerr_list)[np.argsort(wls_eff)]
        filters4extention=np.array(filters4extention)[np.argsort(wls_eff)]
        wls_eff = np.sort(wls_eff)

        self.phot4extention = {'mjd':mjd,'wls_eff':wls_eff,\
                                 'phot':fitted_phot_list,\
                                 'phot_err':fitted_photerr_list,
                              'names':filters4extention}
        return (self.phot4extention)
    
    def grid_all_spectraltimeseries(self):
        ungrid_data_wls= []
        ungrid_data_flux= []
        ungrid_data_fluxerr= []
        mjd =[]
        
        DELTA= self.DELTA#70.
        for f in self.mangledspec_list:
            spec = self.load_mangledfile(f)
            if self.verbose: print (f, len(spec))
            #smoothed_wls, smoothed_flux, smoothed_flux_err = \
            #        self.running_mean_std(spec['wls'], spec['flux'], delta_fix=DELTA)
            
            smoothed_wls = spec['wls'][~np.isnan(spec['flux'])]
            smoothed_flux = spec['flux'][~np.isnan(spec['flux'])]
            smoothed_flux_err = spec['fluxerr'][~np.isnan(spec['flux'])]
            
            ungrid_data_wls.append(smoothed_wls)
            ungrid_data_flux.append(smoothed_flux)
            ungrid_data_fluxerr.append(smoothed_flux_err)
# --- Convert filename MJD to Log Phase ---
            raw_mjd = mangled_filename_to_mjd(f)
            phase = raw_mjd - self.t0_fix
            if phase <= 0: phase = 1e-5
            mjd.append(np.log10(phase))
        #grid_wls = np.arange(1590., 11050., DELTA)
        #rav changed this to try
        # GP / GP2dim_utils expect grid **log10(wavelength in Å)**; files use linear Å
        all_log_wls = np.concatenate(
            [np.log10(np.clip(w, 1.0, None)) for w in ungrid_data_wls]
        )
        min_log_wls = np.nanmin(all_log_wls)
        max_log_wls = np.nanmax(all_log_wls)
        mid_lin = 10.0 ** (0.5 * (min_log_wls + max_log_wls))
        step_log = np.log10(1.0 + DELTA / max(mid_lin, 100.0))
        grid_wls = np.round(
            np.arange(min_log_wls, max_log_wls + step_log, step_log), 8
        )
        #rav changing this to try
        grid_mjd = np.array(mjd)
        #time_step = 0.1
        #grid_mjd = np.arange(min(mjd), max(mjd) + time_step, time_step)
        col_flux = {}
        col_fluxerr = {}

        do_plot = bool(getattr(self, "plot_grid_rebin", False))
        fig = None
        plt = None
        if do_plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(11, 4))

        for ii in range(len(ungrid_data_flux)):
            grid_flux = np.ones(len(grid_wls))
            grid_fluxerr = np.ones(len(grid_wls))

            lin_w = np.asarray(ungrid_data_wls[ii], dtype=float)
            log_spec_wls = np.log10(np.clip(lin_w, 1.0, None))
            ok = ~np.isnan(ungrid_data_flux[ii])
            minwls = np.nanmin(log_spec_wls[ok])
            maxwls = np.nanmax(log_spec_wls[ok])
            data_mask = (grid_wls >= minwls) & (grid_wls <= maxwls)
            flux_interp = np.interp(
                grid_wls[data_mask], log_spec_wls, ungrid_data_flux[ii]
            )
            flux_interp_err = np.interp(
                grid_wls[data_mask], log_spec_wls, ungrid_data_fluxerr[ii]
            )
            if do_plot and plt is not None and fig is not None:
                plt.plot(
                    lin_w,
                    ungrid_data_flux[ii],
                    lw=0.5,
                    color="k",
                    label="origina spectrophotometry",
                )
                plt.plot(
                    10.0 ** grid_wls[data_mask],
                    flux_interp,
                    ".r",
                    ms=3,
                    label='"discretized" spectophotometry',
                )
                plt.fill_between(
                    10.0 ** grid_wls[data_mask],
                    flux_interp - flux_interp_err,
                    flux_interp + flux_interp_err,
                    alpha=0.2,
                    color="r",
                )
                plt.ylabel(
                    "log10 flux"
                    if np.nanmax(np.abs(ungrid_data_flux[ii][ok])) < 30
                    else "Flux"
                )
                plt.xlabel("Wavelength (Å)")
            grid_flux[data_mask] = flux_interp
            grid_flux[~data_mask] = np.nan
            grid_fluxerr[data_mask] = flux_interp_err
            grid_fluxerr[~data_mask] = np.nan
            col_flux[grid_mjd[ii]] = grid_flux
            col_fluxerr[grid_mjd[ii]] = grid_fluxerr

        if do_plot and plt is not None and fig is not None:
            plt.title(
                self.snname
                + " Red: rebbined spectra for grid, Black: original unsmoothed spec"
            )
            out_dir = getattr(self, "save_plot_path", None) or self.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_pdf = os.path.join(out_dir, "grid_rebin_%s.pdf" % self.snname)
            fig.savefig(out_pdf, bbox_inches="tight")
            plt.close(fig)
        grid_all = pd.DataFrame(col_flux, index=grid_wls)
        grid_all_err = pd.DataFrame(col_fluxerr, index=grid_wls)
        self.grids = [grid_wls, grid_mjd, grid_all, grid_all_err]
        
        return grid_wls, grid_mjd, grid_all, grid_all_err

    
    def band_flux_modified(self, filter_name, file):
        
        spec_flux_log = self.load_mangledfile(file)
        
        # Mangled save format: linear Å in ``wls``, log10 flux in ``flux`` (see 5_Mangle_spectra_KN_log)
        spec_wls_lin = self._spec_wls_linear(spec_flux_log)
        spec_flux_lin = self._log10_flux_to_linear(spec_flux_log['flux'])
        
        filt_transm = self._load_filter_transmission(filter_name)
        
        # Calculate the linear effective wavelength
        linear_lam_eff = (integrate.trapezoid(filt_transm['flux']*filt_transm['wls'], filt_transm['wls'])/\
            integrate.trapezoid(filt_transm['flux'], filt_transm['wls']))
            
        filt_transm_interp_func = interpolate.interp1d(filt_transm['wls'], filt_transm['flux'], kind='linear')
        
        # --- CHANGED: Mask using the linear wavelengths ---
        cut_spec = (spec_wls_lin > min(filt_transm['wls'])) & (spec_wls_lin < max(filt_transm['wls']))
        
        wls_cut = spec_wls_lin[cut_spec]
        flux_cut = spec_flux_lin[cut_spec]
        
        filt_transm_interp = filt_transm_interp_func(wls_cut)
        
        raw_phot = integrate.trapezoid(filt_transm_interp*flux_cut, wls_cut)/\
                 integrate.trapezoid(filt_transm['flux'], filt_transm['wls'])
                 
        # --- CHANGED: Return both the effective wavelength and photometry natively in log-space ---
        return np.log10(linear_lam_eff), np.log10(np.clip(raw_phot, 1e-30, None))
                
        
    def extend_grid_all_spectraltimeseries(self):
        if not hasattr(self, 'grids'):
            self.grid_all_spectraltimeseries()
        
        grid_notext = (self.grids[2]).copy()
        grid_notext_err = (self.grids[3]).copy()

        uv_new_cols: dict[float, np.ndarray] = {}
        uv_new_cols_err: dict[float, np.ndarray] = {}
        uv_assignments: list[tuple[float, float, float, float]] = []
        n_grid_rows = len(grid_notext.index)

        for f in self.get_mangledspec_list():
            self.load_phot_for_extention(f)
            phot4ext = self.phot4extention
            for ind in range(len(phot4ext['wls_eff'])):#np.where(phot4ext['wls_eff']<3500.)[0]:
                #if th raw associated with the wls doesnt exist create it
                phot_cut = self.band_flux_modified(phot4ext['names'][ind], f)[1]
                UVwls = phot4ext['wls_eff'][ind]
                f_obs_lin = float(self._log10_flux_to_linear(phot4ext['phot'][ind]))
                f_syn_lin = float(self._log10_flux_to_linear(phot_cut))
                if phot4ext['names'][ind] in ['swift_UVW1', 'swift_UVW2', 'swift_UVM2']:
                    phot_perc = (100.0 * f_syn_lin / f_obs_lin) if f_obs_lin > 0.0 else 0.0
                else:
                    phot_perc = 0.0
                if self.verbose: print ('UV synth versus obs phot %.2f'%phot_perc)
                if self.verbose: print ('UV synth - obs phot %.2E'%(f_obs_lin - f_syn_lin))
                mjd_col = float(np.round(phot4ext['mjd'], 10))
                if UVwls not in grid_notext.index:
                    grid_notext.loc[UVwls]= np.nan*np.ones(grid_notext.shape[1])
                    grid_notext_err.loc[UVwls]= np.nan*np.ones(grid_notext_err.shape[1])
                if (
                    mjd_col not in grid_notext.columns
                    and mjd_col not in uv_new_cols
                ):
                    uv_new_cols[mjd_col] = np.full(n_grid_rows, np.nan)
                    uv_new_cols_err[mjd_col] = np.full(n_grid_rows, np.nan)
                if (phot_perc>1)&(phot_perc<99):
                    flux_val = np.log10(np.maximum(f_obs_lin - f_syn_lin, 1e-300))
                else:
                    flux_val = phot4ext['phot'][ind]
                uv_assignments.append(
                    (UVwls, mjd_col, float(flux_val), float(phot4ext['phot_err'][ind]))
                )

        if uv_new_cols:
            n_rows_now = len(grid_notext.index)
            for key, arr in list(uv_new_cols.items()):
                if len(arr) < n_rows_now:
                    pad = np.full(n_rows_now - len(arr), np.nan)
                    uv_new_cols[key] = np.concatenate([arr, pad])
                    uv_new_cols_err[key] = np.concatenate(
                        [uv_new_cols_err[key], pad]
                    )
            grid_notext = pd.concat(
                [grid_notext, pd.DataFrame(uv_new_cols, index=grid_notext.index)],
                axis=1,
            )
            grid_notext_err = pd.concat(
                [grid_notext_err, pd.DataFrame(uv_new_cols_err, index=grid_notext_err.index)],
                axis=1,
            )
        for UVwls, mjd_col, flux_val, err_val in uv_assignments:
            grid_notext.loc[UVwls, mjd_col] = flux_val
            grid_notext_err.loc[UVwls, mjd_col] = err_val

        LC_fit = self.open_LCfit_file()
        # Cleanly extract filter names from the logspace file
        filters_LC = [i.replace('_log_flux', '') for i in LC_fit.columns if '_log_flux' in i and '_err' not in i]
        
        # --- CHANGED: Pull the Log_Phase column instead of MJD ---
        mjds_LC = LC_fit['Log_Phase'].values
        min_mjds_LC =[]
        max_mjds_LC =[]
        for band in filters_LC:
            # --- CHANGED: Append '_log_flux' to the band string to match the dataframe column ---
            mjd_filt = mjds_LC[~np.isnan(LC_fit[band + '_log_flux'].values)]
            #print(f"{band} LC values:\n", LC_fit[band + '_log_flux'].values)
            #print(f"{band} LC values:\n", LC_fit[band].values)
            #added in this line
            if len(mjd_filt) == 0:
                print(f" !!!! Skipping band {band} — no valid data.")
                continue  # skip this iteration
            min_mjds_LC.append(min(mjd_filt))
            max_mjds_LC.append(max(mjd_filt))
            if self.lam_eff(band) not in grid_notext.index:
                grid_notext.loc[self.lam_eff(band)] = np.full(len(grid_notext.columns), np.nan)
                grid_notext_err.loc[self.lam_eff(band)] = np.full(len(grid_notext.columns), np.nan)

        if self.mode=='extrapolate_spectra':
            _pre_bump = getattr(self, "pre_bump", None)
            if _pre_bump is None:
                _pre_bump = tuple(getattr(_pconf, "PRE_BUMP_SNAMES", ()))
            else:
                _pre_bump = tuple(_pre_bump)

            # Gap fill (linear phase days, default 0.1 d): insert *interior* time columns between spectrum
            # epochs where the gap exceeds gap_size, so the 2D GP sees photometric constraints in long
            # stretches with no spectrum — same idea as the original linear notebook (not new synthetic
            # physics; values still come from np.interp of the fitted LC onto those times).
            _gap_d = float(getattr(self, 'extrapolate_gap_fill_days', 0.1))
            _cad_d = float(getattr(self, 'extrapolate_gap_fill_cadence_days', 0.1))
            spec_cols = np.asarray(grid_notext.columns, dtype=float)
            gap_fill_logs = GP2dim.fill_gaps_phase_logspace(
                min(min_mjds_LC),
                max(max_mjds_LC),
                spec_cols,
                gap_size_days=_gap_d,
                cadence_days=_cad_d,
                tiny_linear_gap_min_log_span_dex=float(
                    getattr(self, "extrapolate_tiny_linear_gap_min_log_span_dex", 0.15)
                ),
                tiny_linear_gap_max_interior=int(
                    getattr(self, "extrapolate_tiny_linear_gap_max_interior", 200)
                ),
            )

            _span_early = float(getattr(self, "extrap_phot_early_span_dex", 1.5))
            if self.snname in _pre_bump:
                early_extrap = np.linspace(
                    min(min_mjds_LC), min(min_mjds_LC) + _span_early, 15
                )
            else:
                early_extrap = np.linspace(
                    min(min_mjds_LC), min(min_mjds_LC) + _span_early, 7
                )
            mjds_extention = np.sort(
                np.unique(np.round(np.concatenate([early_extrap, gap_fill_logs]), 10))
            )

            # Optional cap on span in *dex* of log10(phase days): e.g. 2 dex from *grid minimum* caps
            # linear time to ~10 d if min phase ~0.1 d. Default None = no chop (KN to ~25+ d).
            _span_dex = getattr(self, 'extrapolate_log_phase_span_dex', None)
            if _span_dex is not None and float(_span_dex) > 0.0:
                lo, hi = float(np.min(mjds_extention)), float(np.max(mjds_extention))
                if (hi - lo) > float(_span_dex):
                    mjds_extention = mjds_extention[(mjds_extention - lo) <= float(_span_dex)]
                    if self.verbose:
                        print('Chopping extrapolation to extrapolate_log_phase_span_dex =', _span_dex, 'dex in log10(phase).')
            # Latest phase: max over bands, and max Log_Phase row in the table (robust if columns differ)
            _lc_hi_table = float(np.nanmax(LC_fit['Log_Phase'].values))
            if len(max_mjds_LC) > 0:
                _lc_hi = float(max(max_mjds_LC))
                _lc_hi = float(max(_lc_hi, _lc_hi_table))
            else:
                _lc_hi = _lc_hi_table
            # Always extend grid to latest LC log-phase so photometry can be placed to last observation
            if np.isfinite(_lc_hi) and float(np.max(mjds_extention)) < _lc_hi - 1e-9:
                mjds_extention = np.sort(np.unique(np.concatenate([mjds_extention, np.array([_lc_hi])])))
            mjds_grid = np.sort(
                np.unique(
                    np.round(
                        np.concatenate([mjds_extention, grid_notext.columns]),
                        10,
                    )
                )
            )
            # LC placement schedule on *extrap* log-phases only (log-cadence; not raw mjds_grid index).
            extrap_phot_times = self._select_extrap_phot_times(mjds_extention)

            if getattr(self, 'debug_training_grid', False) or self.verbose:
                print(
                    '[extend_grid] linear phase days: mjds_extention min/max',
                    float(10 ** np.min(mjds_extention)),
                    float(10 ** np.max(mjds_extention)),
                    'n=',
                    len(mjds_extention),
                )
                print(
                    '[extend_grid] linear phase days: mjds_grid min/max',
                    float(10 ** np.min(mjds_grid)),
                    float(10 ** np.max(mjds_grid)),
                    'n=',
                    len(mjds_grid),
                )
                if len(max_mjds_LC) > 0:
                    print(
                        '[extend_grid] max LC phase (days):',
                        float(10 ** max(max_mjds_LC)),
                    )
                print(
                    '[extend_grid] extrapolate_log_phase_span_dex:',
                    getattr(self, 'extrapolate_log_phase_span_dex', None),
                )
                print(
                    '[extend_grid] extrap_phot: n_mjds_ext=',
                    len(mjds_extention),
                    'n_extrap_phot_placed=',
                    len(extrap_phot_times),
                )
                for label, arr in [
                    ("mjds_extention", mjds_extention),
                    ("mjds_grid", mjds_grid),
                    ("extrap_phot_times", extrap_phot_times),
                ]:
                    a = np.asarray(arr, dtype=float)
                    m = (a >= -3.0) & (a <= -1.0)
                    print(
                        "[extend_grid] count log10(phase) in [-3,-1] —",
                        label,
                        int(np.sum(m)),
                        "of",
                        len(a),
                    )

            def _col_exists(cols, val, atol=1e-9):
                v = float(val)
                for c in cols:
                    if np.isclose(float(c), v, rtol=0.0, atol=atol):
                        return True
                return False

            def _get_col_key(cols, val, atol=1e-9):
                v = float(val)
                for c in cols:
                    if np.isclose(float(c), v, rtol=0.0, atol=atol):
                        return c
                return float(np.round(v, 10))

            extrap_new_cols: dict[float, np.ndarray] = {}
            extrap_new_cols_err: dict[float, np.ndarray] = {}
            n_grid_rows = len(grid_notext.index)
            for m in range(len(mjds_extention)):
                ext_t = float(mjds_extention[m])
                if not _col_exists(grid_notext.columns, ext_t):
                    key = float(np.round(ext_t, 10))
                    if key not in extrap_new_cols:
                        extrap_new_cols[key] = np.full(n_grid_rows, np.nan)
                        extrap_new_cols_err[key] = np.full(n_grid_rows, np.nan)
            if extrap_new_cols:
                grid_notext = pd.concat(
                    [grid_notext, pd.DataFrame(extrap_new_cols, index=grid_notext.index)],
                    axis=1,
                )
                grid_notext_err = pd.concat(
                    [
                        grid_notext_err,
                        pd.DataFrame(extrap_new_cols_err, index=grid_notext_err.index),
                    ],
                    axis=1,
                )
            #grid_notext.columns = mjds_grid
            _eps = 1e-5
            for band in filters_LC:
                # --- CHANGED: Read the log flux strings ---
                no_nan = (~np.isnan(LC_fit[band+'_log_flux'].values))&(~np.isnan(LC_fit[band+'_log_flux_err'].values))
                mjd_filt = mjds_LC[no_nan]
                flux_filt = (LC_fit[band+'_log_flux'].values)[no_nan]
                fluxerr_filt = (LC_fit[band+'_log_flux_err'].values)[no_nan]
                if self.verbose:
                    print(f"  mjd_filt ({band}):", mjd_filt)
                if len(mjd_filt) == 0:
                    print(f"Skipping band {band} — no valid flux or flux error data.")
                    continue
                order = np.argsort(mjd_filt)
                mjd_filt = mjd_filt[order]
                flux_filt = flux_filt[order]
                fluxerr_filt = fluxerr_filt[order]
                lo_f, hi_f = float(np.min(mjd_filt)), float(np.max(mjd_filt))
                mask_grid = (mjds_grid >= lo_f - _eps) & (mjds_grid <= hi_f + _eps)
                LC_val = np.full(len(mjds_grid), np.nan)
                LC_val[mask_grid] = np.interp(mjds_grid[mask_grid], mjd_filt, flux_filt)
                LCerr_val = np.full(len(mjds_grid), np.nan)
                LCerr_val[mask_grid] = np.interp(mjds_grid[mask_grid], mjd_filt, fluxerr_filt)
                row = float(self.lam_eff(band))
                mjds_ext_arr = np.asarray(mjds_extention, dtype=float)
                for m in range(len(mjds_grid)):
                    if not mask_grid[m]:
                        continue
                    if not np.isfinite(LC_val[m]) or not np.isfinite(LCerr_val[m]):
                        continue
                    on_extrap = np.any(
                        np.isclose(
                            mjds_ext_arr, float(mjds_grid[m]), rtol=0.0, atol=1e-9
                        )
                    )
                    # Extrap times only: sub-sample with log10(phase)-aware min spacing (extrap_phot_times),
                    # not the raw column index m (which mixed spectrum + extrap and broke early coverage).
                    # Spectrum-only times still have no LC here except endpoint_ok (hi_f) — see plan v1.
                    sparse_ok = on_extrap and self._extrap_time_in_allowed(
                        mjds_grid[m], extrap_phot_times
                    )
                    endpoint_ok = np.isclose(
                        float(mjds_grid[m]), hi_f, rtol=0.0, atol=1e-6
                    )
                    if sparse_ok or endpoint_ok:
                        c = _get_col_key(grid_notext.columns, mjds_grid[m])
                        grid_notext.loc[row, c] = LC_val[m]
                        grid_notext_err.loc[row, c] = LCerr_val[m]
            
        grid_notext = grid_notext.copy()
        grid_notext_err = grid_notext_err.copy()

        grid_ext = grid_notext.sort_index(inplace=False)
        grid_ext_err = grid_notext_err.sort_index(inplace=False)

        grid_ext_full = grid_ext.sort_index(axis=1, inplace=False)
        grid_ext_err_full = grid_ext_err.sort_index(axis=1, inplace=False)
        self.extended_grid = grid_ext_full
        return grid_ext_full.index, grid_ext_full.columns, grid_ext_full, grid_ext_err_full

    def get_filter_LC(self):
        LC_fit = self.open_LCfit_file()
# --- CHANGED: Strip the suffix completely to get clean filter names ---
        filters_LC = [i.replace('_log_flux', '') for i in LC_fit.columns if '_log_flux' in i and '_err' not in i]
        self.avail_filters = filters_LC
        return filters_LC
        
    def open_LCfit_file(self):
        #if not hasattr(self, "results_mainpath"):
        #    self.check_mangling_file()
        #LC_file = self.output_dir+'/%s/fitted_phot_%s.dat'%(self.snname,self.snname)
        LC_file = self.output_dir+'/%s/fitted_phot_logspace_%s.dat'%(self.snname,self.snname)
        if not os.path.isfile(LC_file):
            print ("I need the file with fitted photometry in order to mangle a spectrum")
        else:
            fittphot = pd.read_csv(LC_file, sep='\t')
        return fittphot
    
    def get_spec_mjd(self):
        log_phases = []
        for f in self.get_mangledspec_list():
            raw_mjd = mangled_filename_to_mjd(f)
            phase = raw_mjd - self.t0_fix
            if phase <= 0: phase = 1e-5
            log_phases.append(np.log10(phase))
        return np.array(log_phases)
    
