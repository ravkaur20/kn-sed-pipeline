"""Early-time Bazin / power-law LC fitting (NB2 preview/commit)."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from lc_bazin_fit_config import BazinFitConfig
from lc_extrap_helpers import (
    bazin_forced_zero_t0_bounds_guess,
    clip_extrap_uncertainties,
    covariance_is_bad,
)

__all__ = ["perform_bazin_fit", "performe_fit"]

n_hydr_avg: list[tuple[float, str]] = []


def perform_bazin_fit(
    sn,
    filt,
    t_,
    flux_,
    fluxerr_,
    phase_,
    mjd_phase_ref,
    *,
    config: BazinFitConfig,
    plot: bool = False,
):
    """Fit early rising LC with Bazin / power-law models; return extrapolation arrays."""
    explosion_dates = config.explosion_dates
    nfree_ = config.nfree
    n_sesn = config.n_sesn
    n_hydr = config.n_hydr
    sn_n_fix = config.sn_n_fix
    pre_bump = config.pre_bump
    se_sne = config.se_sne
    hydr_sne = config.hydr_sne


    t0_fix, t0_lower, t0_upper = explosion_dates[sn]
    if t0_lower:
        p_t0 = t0_lower+1.
    else: 
        t0_lower = mjd_phase_ref-100.
        p_t0 = min(min(phase_),-15.)+mjd_phase_ref

    if t0_upper:
        p_t0 = t0_upper-1.
    else: 
        t0_upper = mjd_phase_ref

    if sn in nfree_:
        n_fix = None
        p_n = 1.55
        n_upper = 3.
        n_lower = 0.05
    else:
        n_fix = n_sesn if (sn in se_sne) else n_hydr
        if sn in sn_n_fix: n_fix = sn_n_fix[sn]
        p_n = 1.5 if (sn in se_sne) else 0.3
        n_upper = None
        n_lower = None

    if sn in pre_bump.keys():
        deltaT, t_bump_fix, b_fix, sigma = pre_bump[sn]

        if (isinstance(t_bump_fix, float))&(isinstance(b_fix, float))&(isinstance(sigma, float)):
            names=['a','n']
            def fit(x, a, n):
                t0 = t0_fix
                t_bump = t_bump_fix
                sig = sigma
                b = b_fix
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t
            p0 = [0.5, 1.5]
            bounds=([0., 0.05],[100., 3.])   

        elif (isinstance(t_bump_fix, float))&(isinstance(b_fix, float))&(~isinstance(sigma, float)):
            names=['a','n','sig']
            def fit(x, a, n, sig):
                t0 = t0_fix
                t_bump = t_bump_fix
                b = b_fix
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t
            p0 = [0.5, 1.5, 1.]
            bounds=([0., 0.05, 0.05],[100., 3., 10.])   

        elif (isinstance(t_bump_fix, float))&(~isinstance(b_fix, float))&(isinstance(sigma, float)):
            names=['a','n','b']
            def fit(x, a, n, b):
                t0 = t0_fix
                t_bump = t_bump_fix
                sig = sigma
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t
            p0 = [0.5, 1.5, 0.5]
            bounds=([0., 0.05, 0.],[100., 3., 10000.])   

        elif (isinstance(t_bump_fix, float))&(isinstance(n_fix, float)):
            names=['a','b','sig']
            def fit(x, a, b, sig):
                t0 = t0_fix
                t_bump = t_bump_fix
                n = n_fix
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t
            p0 = [0.5, 0.5, 1.]
            bounds=([0., 0., 0.05],[100., 10000., 10.])   
        
        elif (isinstance(t_bump_fix, float)):
            names=['a','n','b','sig']
            def fit(x, a, n, b, sig):
                t0 = t0_fix
                t_bump = t_bump_fix
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t
            p0 = [0.5, 1.5, 0.5, 1.]
            bounds=([0., 0.05, 0., 0.05],[100., 3., 10000., 10.])   

        else:
            names=['a','n','b','t_bump','sig']
            def fit(x, a, n, b, t_bump, sig):
                t0 = t0_fix
                f_t = np.zeros(len(x))
                f_t[x>t0] = a * (x[x>t0]-t0)**n + b*np.exp(-np.power(x[x>t0] - t_bump, 2.) / (2 * np.power(sig, 2.)))*(1.-np.exp(-(x[x>t0]-t0)))
                return f_t        
            p0 = [0.5, 1.5, 0.5, t0_fix+1., 1.]
            bounds=([0., 0.05, 0., t0_fix, 0.05],[100., 3., 10000., t0_fix+5., 10.])     
        
        print ('Fitting', names, 'Initial guess', p0)
        if plot:
            plt.vlines(t0_fix, 0, 1, linestyle='-', alpha=0.5)
        
    elif (isinstance(n_fix, float))&(isinstance(t0_fix, float)):
        names=['a']
        def fit(x, a):
            t0 = t0_fix ; n = n_fix
            f_t = np.zeros(len(x)) ; f_t[x>t0] = a * (x[x>t0]-t0)**n
            return f_t
        p0 = [0.5]
        bounds=([0.],[10.])   
        if plot:
            plt.vlines(t0_fix, 0, 1, linestyle='-', alpha=0.5)
        print ('Fitting', names, 'Initial guess', p0)

    # elif (~isinstance(n_fix, float))&isinstance(t0_fix, float):
    #     names=['a','n']
    #     def fit(x, a, n):
    #         t0 = t0_fix
    #         f_t = np.zeros(len(x)) ; f_t[x>t0] = a * (x[x>t0]-t0)**n
    #         return f_t
    #     p0 = [0.5, p_n]#p_n
    #     bounds=([0., n_lower],[10.,n_upper])
    #     plt.vlines(t0_fix, 0,1, linestyle='-', alpha=0.5)
    #     print ('Fitting', names, 'Initial guess', p0)

    elif isinstance(t0_fix, float):
        # --- NEW FORCED-ZERO BAZIN FIT ---
        names=['a','t0','t_fall','t_rise']
        def fit(x, a, t0, t_fall, t_rise):
            # Calculate standard Bazin
            arg_fall = np.clip(-(x - t0) / t_fall, -700, 700)
            arg_rise = np.clip(-(x - t0) / t_rise, -700, 700)
            val = a * (np.exp(arg_fall) / (1 + np.exp(arg_rise)))
            
            # Calculate Bazin exactly at the explosion date
            arg_fall_exp = np.clip(-(t0_fix - t0) / t_fall, -700, 700)
            arg_rise_exp = np.clip(-(t0_fix - t0) / t_rise, -700, 700)
            val_exp = a * (np.exp(arg_fall_exp) / (1 + np.exp(arg_rise_exp)))
            
            # Subtracting the explosion value forces the curve to be 0 at t0_fix
            return val - val_exp
        
        bazin_t0_lo, bazin_t0_hi, p_t0 = bazin_forced_zero_t0_bounds_guess(t_, flux_)
        p0 = [max(flux_) if len(flux_)>0 else 0.5, p_t0, 30.0, 5.0]
        bounds=(
            [0., bazin_t0_lo, 5.0, 0.1],
            [np.inf, bazin_t0_hi, 100.0, 50.0]
        )
        if plot:
            plt.vlines(t0_fix, 0, 1, linestyle='-', alpha=0.5)
        print ('Fitting Forced-Zero Bazin', names, 'Initial guess', p0)
    
    elif (isinstance(n_fix, float))&(~isinstance(t0_fix, float)):
        names=['a','t0']
        def fit(x, a, t0):
            n = n_fix
            f_t = np.zeros(len(x)) ; f_t[x>t0] = a * (x[x>t0]-t0)**n
            return f_t
        _dt_lo, _dt_hi, p_t0 = bazin_forced_zero_t0_bounds_guess(t_, flux_)
        p_t0 = float(np.clip(p_t0, min(t0_lower, _dt_lo), max(t0_upper, _dt_hi)))
        p0 = [0.5, p_t0]
        bounds=([0., min(t0_lower, _dt_lo)],[10., max(t0_upper, _dt_hi)])
        if plot:
            plt.vlines([t0_lower, t0_upper], 0, 1, linestyle='--', alpha=0.5)
        print ('Fitting', names, 'Initial guess', p0)

    else:
        names=['a','t0','t_fall','t_rise','c']
        def fit(x, a, t0, t_fall, t_rise, c):
            arg_fall = np.clip(-(x - t0) / t_fall, -700, 700)
            arg_rise = np.clip(-(x - t0) / t_rise, -700, 700)
            return a * (np.exp(arg_fall) / (1 + np.exp(arg_rise))) + c
        
        # Bazin guesses: [A, t0, t_fall, t_rise, c]
        _dt_lo, _dt_hi, _p_t0 = bazin_forced_zero_t0_bounds_guess(t_, flux_)
        bazin_t0_lo = min(t0_lower, _dt_lo)
        bazin_t0_hi = max(t0_upper, _dt_hi)
        p_t0 = float(np.clip(_p_t0, bazin_t0_lo, bazin_t0_hi))
        p0 = [max(flux_) if len(flux_)>0 else 0.5, p_t0, 30.0, 5.0, 0.0]
        bounds=(
            [0., bazin_t0_lo, 5.0, 0.1, -np.inf],
            [np.inf, bazin_t0_hi, 100.0, 50.0, np.inf]
        )
        if plot:
            plt.vlines([t0_lower, t0_upper], 0, 1, linestyle='--', alpha=0.5)
        print ('Fitting Bazin', names, 'Initial guess', p0)

    R,cov = opt.curve_fit(fit, t_, flux_, p0=p0, 
                        sigma=fluxerr_, absolute_sigma=True, maxfev=10000, bounds=bounds)
        
    if 't0' in names:
        explos_day = R[np.array(names)=='t0'][0]
    else:
        explos_day = t0_fix
    # CHANGED THIS
    # dT = 0.25 if sn not in pre_bump.keys() else 0.1    
    # test_times = np.arange(explos_day, 15.+explos_day,10**-4)
    
    # FF_pts = np.arange(0.3, min(flux_), 0.05)
    # if sn not in hydr_sne: FF_pts = np.arange(0.01, min(flux_), 0.05)
    # TT_pts =[]
    # for f in FF_pts:
    #     TT_pts.append(test_times[np.argmin(np.abs(fit(test_times, *R)-f))] )
    # t_newpts = np.array(TT_pts)
    # NEW! ADDED THIS
    
    if 't0' in names and 't_rise' not in names: # fallback for older models
        explos_day = R[np.array(names)=='t0'][0]
    else:
        explos_day = t0_fix
        
    # Get the time of the first real data point
    first_time = locals().get('original_first_time', np.min(t_))
    
    if first_time > explos_day:
        # Evaluate the fitted model at the first real observation to get our upper flux limit
        first_real_flux = fit(first_time, *R)

        # FIX: Create a dense time grid strictly stopping AT first_time (no + 0.1)
        test_times = np.arange(explos_day, first_time, 10**-4)

        # Map the flux points back to times using the dense grid
        FF_pts = np.arange(0.3, first_real_flux, 0.05)
        if sn not in hydr_sne: 
            FF_pts = np.arange(0.01, first_real_flux, 0.05)

        TT_pts = []
        for f in FF_pts:
            if len(test_times) > 0:
                idx = np.argmin(np.abs(fit(test_times, *R) - f))
                mapped_time = test_times[idx]
                
                # FIX: Extra safety check to guarantee it doesn't map past the first point
                if mapped_time < first_time:
                    TT_pts.append(mapped_time)

        t_newpts = np.array(TT_pts)
    else:
        t_newpts = np.array([])
    
    # Define the full range for plotting the smooth fitted line
    t_extrap = np.arange(explos_day - 2.0, np.max(t_) + 10.0, 0.1)

    # if np.any(np.isinf(cov)): 
    #     print ('#######################'+'COVARIANCE MATRIX inf')
    #     fit_ = fit(t_extrap, *R)
    #     fit_err = 0.1*fit_#np.zeros(len(fit_))

    #     newpts_ = fit(t_newpts, *R)
    #     newpts_err = 0.1*newpts_ #np.zeros(len(newpts_))
    #     success=True
    if np.any(np.isinf(cov)) or covariance_is_bad(cov): 
        if np.any(np.isinf(cov)):
            print ('#######################'+'COVARIANCE MATRIX inf')
        else:
            print ('#######################'+'COVARIANCE MATRIX unstable (high condition number or invalid diag)')
        fit_ = fit(t_extrap, *R)
        fit_err = 0.1 * np.abs(fit_)
        newpts_ = fit(t_newpts, *R)
        newpts_err = 0.1 * np.abs(newpts_)
        fit_err = clip_extrap_uncertainties(fit_err, flux_, fluxerr_)
        newpts_err = clip_extrap_uncertainties(newpts_err, flux_, fluxerr_)
        success=True
    else:
        rand = np.random.multivariate_normal(R, cov, size=10000)
        
        fit_ = fit(t_extrap, *R)
        # Calculate standard deviation, ignoring infinite/nan overflows
        sim_fit = np.array([fit(t_extrap, *par) for par in rand])
        sim_fit[~np.isfinite(sim_fit)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit_err = np.nanstd(sim_fit, axis=0)
        # Fallback to 10% error if all samples overflowed
        fit_err = np.where(np.isnan(fit_err), 0.1 * np.abs(fit_), fit_err)

        newpts_ = fit(t_newpts, *R)
        # Calculate standard deviation for new points, ignoring overflows
        sim_newpts = np.array([fit(t_newpts, *par) for par in rand])
        sim_newpts[~np.isfinite(sim_newpts)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            newpts_err = np.nanstd(sim_newpts, axis=0)
        # Fallback to 10% error if all samples overflowed
        newpts_err = np.where(np.isnan(newpts_err), 0.1 * np.abs(newpts_), newpts_err)
        fit_err = clip_extrap_uncertainties(fit_err, flux_, fluxerr_)
        newpts_err = clip_extrap_uncertainties(newpts_err, flux_, fluxerr_)

        success=True

    if (sn in hydr_sne)&('n' in names):
        n_hydr_avg.append((R[np.array(names)=='n'][0],filt))

    return R, cov, t_extrap, fit_, fit_err, t_newpts, newpts_, newpts_err, dict(zip(names, np.round(R,1))).items() , success


performe_fit = perform_bazin_fit
