"""2D ln-flux GP grid prep: log10(wavelength) x log10(phase).

Supports min-max or z-score training coordinates via ``pipeline_config.USE_TWO_D_GP_ZSCORE_COORDS``.
Denorm uses ``grid_norm_info`` keys so saved runs remain readable.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy import integrate
import matplotlib.cm as cm
import scipy
import scipy.optimize as opt
from scipy.optimize import minimize
from itertools import cycle
from scipy.interpolate import griddata


from gp2dim_phase_merge import merge_extrap_mjds_dense_log_phase
from gp2dim_export import maybe_save_gp_minimal_export

import george
from george.kernels import Matern32Kernel

import sys
import time
#sys.path.insert(0, '/Users/mariavincenzi/PhD/pycoco_2/')
#import pycoco_general_info as PyCoCo_info

mycmap = plt.cm.viridis
mycmap.set_under('r')


from band_plot_style import COLOR_DICT as color_dict

# float64 exp overflows for arguments ~709; clip scaled ln-flux mapping for stability
_EXP_ARG_CAP = 700.0
# Avoid division by ~zero when ln-flux spread is tiny
_SCALE_FACTOR_ABS_FLOOR = 1e-8
# Floor on coordinate std (population ddof=0) so z-score divides stay stable
_COORD_STD_FLOOR = 1e-12

def _coord_mode(gn=None) -> str:
    """Return ``zscore`` or ``minmax`` from saved metadata or pipeline config."""
    if gn is not None:
        if gn.get("coord_parametrization") == "zscore":
            return "zscore"
        if "norm1" in gn:
            return "minmax"
    try:
        import pipeline_config as _pc
        return "zscore" if bool(getattr(_pc, "USE_TWO_D_GP_ZSCORE_COORDS", True)) else "minmax"
    except ImportError:
        return "zscore"


def _use_zscore_coords(gn=None) -> bool:
    return _coord_mode(gn) == "zscore"


def _x1_norm_to_log10_wavelength(x1_norm, gn):
    x1_norm = np.asarray(x1_norm, dtype=float)
    if _coord_mode(gn) == "zscore":
        return np.asarray(gn["x1_mean"], dtype=float) + np.asarray(gn["x1_std"], dtype=float) * x1_norm
    return np.asarray(gn["norm1"], dtype=float) * x1_norm


def _legacy_axis_params(gn):
	"""Return (x1m, x1s, x2m, x2s) so ``x*m + x*s * x*_norm`` denorms both coord modes."""
	if _coord_mode(gn) == "zscore":
		return (
			float(gn["x1_mean"]),
			float(gn["x1_std"]),
			float(gn["x2_mean"]),
			float(gn["x2_std"]),
		)
	return 0.0, float(gn["norm1"]), float(gn["offset2"]), float(gn["norm2"])


def _x2_norm_to_log10_phase(x2_norm, gn):
	x2_norm = np.asarray(x2_norm, dtype=float)
	if _coord_mode(gn) == "zscore":
		return np.asarray(gn["x2_mean"], dtype=float) + np.asarray(gn["x2_std"], dtype=float) * x2_norm
	return np.asarray(gn["offset2"], dtype=float) + np.asarray(gn["norm2"], dtype=float) * x2_norm


_UNSET = object()


def _resolve_gp_yerr_floors(GP2DIM_Class):
    """``gp_yerr_*`` on the class override ``pipeline_config`` when set."""
    frac = getattr(GP2DIM_Class, "gp_yerr_floor_frac", _UNSET)
    abs_f = getattr(GP2DIM_Class, "gp_yerr_abs_floor", _UNSET)
    if frac is _UNSET or abs_f is _UNSET:
        try:
            import pipeline_config as _pc

            if frac is _UNSET:
                frac = getattr(_pc, "GP_YERR_FLOOR_FRAC", None)
            if abs_f is _UNSET:
                abs_f = float(getattr(_pc, "GP_YERR_ABS_FLOOR", 0.0) or 0.0)
        except ImportError:
            if frac is _UNSET:
                frac = None
            if abs_f is _UNSET:
                abs_f = 0.0
    if abs_f is None:
        abs_f = 0.0
    else:
        abs_f = float(abs_f)
    return frac, abs_f


def _apply_training_yerr_floors(GP2DIM_Class, y, yerr, *, stage):
    """Raise training ``yerr`` optionally: spread-based floor differs for reshape vs pre-``gp.compute`` (legacy).

    ``stage`` is ``\"transform\"`` (reshape) or ``\"compute\"``.
    """
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float).copy()
    frac, abs_f = _resolve_gp_yerr_floors(GP2DIM_Class)
    if frac is not None and float(frac) > 0:
        f = float(frac)
        sy = float(np.nanstd(y))
        if stage == "transform":
            terr = max(f * (sy + 1e-12), f * 1e-6)
        elif stage == "compute":
            terr = max(f * (sy + 1e-12), 1e-12)
        else:
            raise ValueError("_apply_training_yerr_floors: stage must be transform or compute")
        yerr = np.maximum(yerr, terr)
    if abs_f > 0:
        yerr = np.maximum(yerr, abs_f)
    if not np.all(np.isfinite(yerr)) or np.any(yerr <= 0):
        raise ValueError(
            "2D GP training: need strictly positive finite yerr after optional floors. "
            "Check input flux uncertainties, set pipeline_config.GP_YERR_ABS_FLOOR > 0, "
            "or GP_YERR_FLOOR_FRAC (e.g. 1e-4 for legacy spread floor)."
        )
    return yerr


def _resolve_ln_flux_err_from_relative(GP2DIM_Class):
	"""If True, use ``sqrt(ln(1 + (sigma_F/F)^2))`` in ``transform2LOG_reshape``; else legacy ``sigma_log10*ln(10)``."""
	v = getattr(GP2DIM_Class, "gp_ln_flux_err_from_relative", _UNSET)
	if v is _UNSET:
		try:
			import pipeline_config as _pc

			return bool(getattr(_pc, "GP_LN_FLUX_ERR_FROM_RELATIVE", True))
		except ImportError:
			return True
	return bool(v)


def _resolve_ln_flux_offset_floor_enabled(GP2DIM_Class):
	"""If True, cap ln-flux offset at ln(``GP_LN_FLUX_OFFSET_FLOOR_LINEAR``)."""
	v = getattr(GP2DIM_Class, "gp_ln_flux_offset_floor", _UNSET)
	if v is _UNSET:
		try:
			import pipeline_config as _pc

			return bool(getattr(_pc, "GP_LN_FLUX_OFFSET_FLOOR", True))
		except ImportError:
			return True
	return bool(v)


def _resolve_ln_flux_offset_floor_linear(GP2DIM_Class):
	v = getattr(GP2DIM_Class, "gp_ln_flux_offset_floor_linear", _UNSET)
	if v is _UNSET:
		try:
			import pipeline_config as _pc

			return float(getattr(_pc, "GP_LN_FLUX_OFFSET_FLOOR_LINEAR", 1e-30))
		except ImportError:
			return 1e-30
	return float(v)


def _ln_flux_offset_from_data(data, GP2DIM_Class):
	"""Physical minimum ln F, optionally capped at ln(``GP_LN_FLUX_OFFSET_FLOOR_LINEAR``)."""
	physical_min = float(np.nanmin(data[~np.isnan(data)]))
	if _resolve_ln_flux_offset_floor_enabled(GP2DIM_Class):
		ln_floor = float(np.log(_resolve_ln_flux_offset_floor_linear(GP2DIM_Class)))
		return float(min(physical_min, ln_floor))
	return physical_min


def scaled_ln_to_linear(scaled_mu, offset, scale_factor):
	"""Map GP prediction from scaled ln-flux space back to linear flux (same as np.exp(mu*scale+offset), capped)."""
	arg = np.asarray(scaled_mu, dtype=float) * float(scale_factor) + float(offset)
	np.clip(arg, None, _EXP_ARG_CAP, out=arg)
	return np.exp(arg)


def x2_mask_for_phase(x2_fill, phase_log, gn):
	"""Rows of the prediction grid matching ``phase_log`` (log10 days)."""
	if _coord_mode(gn) == "zscore":
		target = (float(phase_log) - float(gn["x2_mean"])) / float(gn["x2_std"])
	else:
		target = (float(phase_log) - float(gn["offset2"])) / float(gn["norm2"])
	return np.isclose(x2_fill, target, rtol=0.0, atol=1e-9)


def phases_close(mj, phase_array, atol=1e-9):
	"""True if ``mj`` matches any spectroscopic epoch in ``phase_array`` (log10 days)."""
	return np.any(np.isclose(np.asarray(phase_array, dtype=float), float(mj), rtol=0.0, atol=atol))


def log_prediction_phase_coverage(phase_columns, *, lo=-3.0, hi=-1.0, label="prediction columns"):
	"""Print how many grid / prediction columns fall in early log10(phase days) range (per plan item 5)."""
	logp = np.asarray(phase_columns, dtype=float).ravel()
	if logp.size == 0:
		print("[GP2dim] %s: empty" % label, flush=True)
		return
	m = np.isfinite(logp) & (logp >= float(lo)) & (logp <= float(hi))
	print(
		"[GP2dim] %s: %i of %i with log10(phase days) in [%.1f, %.1f]"
		% (label, int(np.count_nonzero(m)), int(logp.size), lo, hi),
		flush=True,
	)


def phase_days_from_norm_x2(x2_norm, gn):
	"""Linear phase (days) from normalized x2 using ``grid_norm_info``."""
	lp = _x2_norm_to_log10_phase(x2_norm, gn)
	return np.power(10.0, np.clip(lp, -50.0, 50.0))


def log10_wavelength_from_x1_norm(x1_norm, gn):
	"""log10(wavelength) from normalized x1 using ``grid_norm_info``."""
	return _x1_norm_to_log10_wavelength(x1_norm, gn)


def mangled_wls_max_is_linear_angstrom(wls_arr):
	"""True if ``wls`` look like linear Å (KN log-mangle saves ``raw_spec['wls']`` this way)."""
	w = np.asarray(wls_arr, dtype=float)
	mx = np.nanmax(w)
	return bool(np.isfinite(mx) and mx > 200.0)


def mangled_wls_linear_angstrom(spec_rec):
	"""Return wavelength column in **linear** Å (linear files or log10-Å columns)."""
	w = np.asarray(spec_rec['wls'], dtype=float)
	if mangled_wls_max_is_linear_angstrom(w):
		return w
	return np.power(10.0, np.clip(w, -50.0, 8.0))


def mangled_flux_linear_from_log10(flux_arr):
	"""Convert log10 flux to linear F (mangled ``flux`` / ``fluxerr`` are dex in the log pipeline)."""
	f = np.asarray(flux_arr, dtype=float)
	return np.power(10.0, np.clip(f, -350.0, 300.0))


def fill_gaps_phase_logspace(
	min_log,
	max_log,
	spec_log_phases,
	gap_size_days=0.1,
	cadence_days=0.1,
	tiny_linear_gap_min_log_span_dex=0.15,
	tiny_linear_gap_max_interior=200,
):
	"""Interior log10(phase days) points for gaps between spectra, using **linear** phase-day thresholds.

	The original linear notebook used ``gap_size=0.1`` days in **linear** phase; filling in log10(phase)
	with 0.01 dex misses many small linear gaps. This matches the linear logic: convert to days,
	fill, convert back to log10.

	**Borderline linear gaps (``gap`` <= ``gap_size_days``):** if the first segment from ``min_lin``
	to the first spectrum is just under 0.1d in *linear* days, the strict ``gap > 0.1`` test adds no
	interior points, while log10(phase) can still span ~2 dex. In that case we add log-spaced
	``phase in days`` nodes so the grid is populated in log-phase (see KN early-time gap).
	"""
	min_lin = float(np.power(10.0, float(min_log)))
	max_lin = float(np.power(10.0, float(max_log)))
	spec_lin = np.sort(
		np.unique(np.power(10.0, np.asarray(spec_log_phases, dtype=float).ravel()))
	)
	f = np.concatenate(([min_lin], spec_lin, [max_lin]))
	gaps = f - np.concatenate(([min_lin], f[:-1]))
	extention_lin = []
	_ok_big = np.asarray(gaps) >= (float(gap_size_days) - 1.0e-12)
	for gap, offset in zip(gaps[_ok_big], f[_ok_big]):
		n_steps = int(round(gap / cadence_days, 0))
		if n_steps <= 0:
			continue
		step = gap / n_steps
		for i in range(n_steps - 1):
			extention_lin.append(offset - (i + 1) * step)
	for j in range(1, len(f)):
		lo_lin = max(float(f[j - 1]), 1e-8)
		hi_lin = max(float(f[j]), 1e-8)
		gap = hi_lin - lo_lin
		if not (gap > 0.0 and gap <= float(gap_size_days) + 1.0e-12):
			continue
		dlog = float(np.log10(hi_lin) - np.log10(lo_lin))
		if dlog < float(tiny_linear_gap_min_log_span_dex):
			continue
		n_int = int(np.ceil(dlog / 0.05))
		n_int = max(1, min(n_int, int(tiny_linear_gap_max_interior)))
		logs = np.linspace(np.log10(lo_lin), np.log10(hi_lin), n_int + 2, dtype=float)[1:-1]
		extention_lin.extend((np.power(10.0, np.clip(logs, -20.0, 20.0))).tolist())
	if not extention_lin:
		return np.array([], dtype=float)
	lin = np.clip(np.asarray(extention_lin, dtype=float), 1e-8, None)
	lin = np.unique(lin)
	return np.log10(lin)


def prepare_grid(snname, GP2DIM_Class):
	lcfit = GP2DIM_Class.open_LCfit_file()
	GP2DIM_Class.get_filter_LC()
	xa, ya, grid_nt, griderr_nt = GP2DIM_Class.grid_all_spectraltimeseries()
	xa_ext, ya_ext, grid_ext, griderr_ext = GP2DIM_Class.extend_grid_all_spectraltimeseries()
	
	raw_numbers = grid_ext.values
	raw_numbers_err = griderr_ext.values
	off_xa = xa_ext
	off_ya = ya_ext
	return (raw_numbers, raw_numbers_err, off_xa, off_ya, grid_ext.columns.values)


def transform2LOG_reshape(GP2DIM_Class, raw_numbers, raw_numbers_err,  off_xa, off_ya):
	"""Match GP2dim_utils_original: GP is fit on ln(flux) with scaled, dimensionless coordinates.

	Inputs from the log pipeline are log10(flux) and log10(flux) uncertainties (dex).

	Uncertainty on ln(F): default ``sigma_ln = sqrt(ln(1 + (sigma_F/F)^2))`` with
	``sigma_F = |F| ln(10) sigma_log10`` (``pipeline_config.GP_LN_FLUX_ERR_FROM_RELATIVE``).
	Legacy option: ``sigma_ln = sigma_log10 * ln(10)`` (exact log-base change for additive dex noise).

	Ln-flux offset: default ``min(min ln F, ln(GP_LN_FLUX_OFFSET_FLOOR_LINEAR))`` when
	``pipeline_config.GP_LN_FLUX_OFFSET_FLOOR`` is True (pipeline-compatible encoding).

	Linear flux remains positive after ``scaled_ln_to_linear`` (``exp``). For an unconstrained linear-flux
	GP, use ``GP2dim_utils_linear_flux`` (``USE_TWO_D_GP_LINEAR_FLUX``) only if negatives are acceptable
	or post-processed.

	Axes: x1 = log10(wavelength), x2 = log10(phase in days since explosion), same as the grid in NB 6.

	GP coordinates use **training-only z-scores** per axis (``coord_parametrization='zscore'`` in
	``grid_norm_info``); flux scaling is unchanged from ``GP2dim_utils``.
	"""
	LN10 = np.log(10.0)
	data_log10 = (raw_numbers.T.reshape(raw_numbers.shape[0]*raw_numbers.shape[1]))
	data_log10_err = (raw_numbers_err.T.reshape(raw_numbers_err.shape[0]*raw_numbers_err.shape[1]))

	data_log10 = np.copy(data_log10)
	data_log10_err = np.copy(data_log10_err)
	data_log10[~np.isfinite(data_log10)] = np.nan
	data_log10_err[~np.isfinite(data_log10_err)] = np.nan

	# ln F = ln(10) * log10 F (= ln(F) for F = 10**log10).
	if _resolve_ln_flux_err_from_relative(GP2DIM_Class):
		vl10 = np.asarray(data_log10, dtype=float)
		sig_log = np.asarray(data_log10_err, dtype=float)
		F = np.power(10.0, np.clip(vl10, -350.0, 300.0))
		ok = (
			np.isfinite(vl10)
			& np.isfinite(sig_log)
			& (sig_log >= 0.0)
			& np.isfinite(F)
			& (F > 0.0)
		)
		rel2 = (LN10 * np.maximum(sig_log, 0.0)) ** 2
		rel2 = np.clip(rel2, 0.0, np.finfo(np.float64).max / 4.0)
		data_err = np.full(vl10.shape, np.nan, dtype=float)
		data_err[ok] = np.sqrt(np.log1p(rel2[ok]))
		data = np.where(ok, vl10 * LN10, np.nan)
	else:
		data = data_log10 * LN10
		data_err = data_log10_err * LN10

	offset = _ln_flux_offset_from_data(data, GP2DIM_Class)
	spread = np.nanmedian(data[~np.isnan(data)] - offset)
	scale_factor = max(float(spread), _SCALE_FACTOR_ABS_FLOOR * max(abs(float(offset)), 1.0), _SCALE_FACTOR_ABS_FLOOR)

	data_scaled = (data - offset) / scale_factor
	data_error_scaled = data_err / scale_factor

	## Reshape the grid to feed the 2dGP
	resh_wls = []
	for i in range(raw_numbers.shape[1]):
		resh_wls = np.concatenate([resh_wls, off_xa])

	resh_mjd = []
	for i in off_ya:
		resh_mjd = np.concatenate([resh_mjd, np.ones(len(off_xa))*i])

	NOT_Isnan = (~np.isnan(data_scaled))&(~np.isnan(data_error_scaled))

	x1_data = resh_wls[NOT_Isnan]
	x2_data = resh_mjd[NOT_Isnan]

	y_data_nonan = np.copy(data_scaled[NOT_Isnan])
	y_data_nonan_err = np.copy(data_error_scaled[NOT_Isnan])
	y_data_nonan_err = _apply_training_yerr_floors(
		GP2DIM_Class, y_data_nonan, y_data_nonan_err, stage="transform"
	)

	if _use_zscore_coords():
		x1_mean = float(np.mean(x1_data))
		x1_std = float(np.std(x1_data, ddof=0))
		x2_mean = float(np.mean(x2_data))
		x2_std = float(np.std(x2_data, ddof=0))
		x2_train_min = float(np.min(x2_data))
		x1_std = max(x1_std, _COORD_STD_FLOOR)
		x2_std = max(x2_std, _COORD_STD_FLOOR)
		if not np.isfinite(x1_std) or not np.isfinite(x2_std):
			raise ValueError("transform2LOG_reshape: non-finite coordinate std; check grid.")
		x1_data_norm = (x1_data - x1_mean) / x1_std
		x2_data_norm = (x2_data - x2_mean) / x2_std
		GP2DIM_Class.grid_norm_info = {
			"offset": offset,
			"scale_factor": scale_factor,
			"coord_parametrization": "zscore",
			"x1_mean": x1_mean,
			"x1_std": x1_std,
			"x2_mean": x2_mean,
			"x2_std": x2_std,
			"x2_train_min": x2_train_min,
		}
	else:
		norm1 = float(np.max(x1_data))
		offset2 = float(np.min(x2_data))
		norm2 = float(np.max(x2_data - offset2))
		if norm2 <= 0.0 or not np.isfinite(norm2):
			raise ValueError("transform2LOG_reshape: invalid norm2 (log-phase range); check grid time columns.")
		x1_data_norm = x1_data / norm1
		x2_data_norm = (x2_data - offset2) / norm2
		GP2DIM_Class.grid_norm_info = {
			"offset": offset,
			"scale_factor": scale_factor,
			"norm1": norm1,
			"norm2": norm2,
			"offset2": offset2,
			"x2_train_min": float(np.min(x2_data)),
		}
	return (y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm)


def make_plots(GP2DIM_Class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm):	
	fig=plt.figure(1, figsize=(12,3))
	gn = GP2DIM_Class.grid_norm_info
	x_phase = _x2_norm_to_log10_phase(x2_data_norm, gn)
	y_logwl = _x1_norm_to_log10_wavelength(x1_data_norm, gn)

	plt.subplot(121)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	plt.title('Training Data')
	
	plt.grid(True)
	
	plt.scatter(x_phase, y_logwl, marker='s', s=9,c=y_data_nonan)
	plt.colorbar(label='Flux rescaled')
	#plt.savefig('gaussian_processes_2d_training_data.png', bbox_inches='tight')
	
	plt.subplot(122)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	#plt.xlim(x1_min,x1_max)
	#plt.ylim(x2_min,x2_max)
	plt.title('Training Data ERRORS')
	
	plt.grid(True)
		
	plt.scatter(x_phase, y_logwl,  marker='s', s=9, c=(y_data_nonan_err))
	plt.colorbar(label='Err Flux rescaled')
	plt.subplots_adjust(wspace=0.2)
	plt.show()
	fig.savefig(GP2DIM_Class.save_plot_path+'/data_for2d_interpolation.pdf', bbox_inches='tight')
	plt.close(fig)

	# Same training points in linear space (phase in days, wavelength in Angstroms) for visual checks
	phase_days = np.power(10.0, np.clip(x_phase, -50.0, 50.0))
	wls_angstrom = np.power(10.0, np.clip(y_logwl, -50.0, 50.0))
	fig2 = plt.figure(2, figsize=(12, 3))
	plt.subplot(121)
	plt.scatter(phase_days, wls_angstrom, marker='s', s=9, c=y_data_nonan)
	plt.xlabel('Phase (days since explosion)')
	plt.ylabel('Wavelength (Angstrom)')
	plt.title('Training Data (linear axes)')
	plt.grid(True)
	plt.colorbar(label='Flux rescaled')
	plt.subplot(122)
	plt.scatter(phase_days, wls_angstrom, marker='s', s=9, c=(y_data_nonan_err))
	plt.xlabel('Phase (days since explosion)')
	plt.ylabel('Wavelength (Angstrom)')
	plt.title('Training Data ERRORS (linear axes)')
	plt.grid(True)
	plt.colorbar(label='Err Flux rescaled')
	plt.subplots_adjust(wspace=0.2)
	plt.show()
	fig2.savefig(GP2DIM_Class.save_plot_path + '/data_for2d_interpolation_linear_axes.pdf', bbox_inches='tight')
	plt.close(fig2)
	
def setPRIOR(GP2DIM_Class, type_=None, PRIOR_file=None, PRIOR_folder=None):
	"""Same mapping as GP2dim_utils_original, but training uses log10(wavelength) and log10(phase days).

	Prior file: wavelength (A), phase relative to peak (days), color multiplier.
	Original: absolute_MJD = phase_prior + peak_MJD; normalize MJD.
	Here: time coordinate on the grid is log10(MJD - t0_fix); peak_MJD from the fitted LC file.
	"""
	gn = GP2DIM_Class.grid_norm_info
	offset = gn['offset']
	scale_factor = gn['scale_factor']
	t0 = GP2DIM_Class.t0_fix

	if not PRIOR_file:
		if type_ in ['II', 'IIn', 'IIP', 'IIL']:
			PRIOR_file = '/prior_Hrich.txt'
		elif type_ in ['Ib', 'Ic', 'Ibc', 'Ic-BL', 'IcBL', 'IIb']:
			PRIOR_file = '/prior_SE.txt'
		else: print ('Specify a PRIOR please')

	wls_prior, phase_prior, color_prior = np.genfromtxt(PRIOR_folder+PRIOR_file, delimiter=',', unpack=True)

	original_fit = pd.read_csv(GP2DIM_Class.path_fit_phot ,delimiter='\t')
	ref_cols = [c for c in original_fit.columns if c.endswith('_log_flux') and '_err' not in c]
	if not ref_cols:
		raise ValueError('setPRIOR: no *_log_flux columns in %s' % GP2DIM_Class.path_fit_phot)
	counts = {c: np.sum(np.isfinite(original_fit[c].values)) for c in ref_cols}
	best = max(counts, key=counts.get)
	if counts[best] < 2:
		raise ValueError('setPRIOR: insufficient valid points in reference band %s' % best)
	original_fit = original_fit[np.isfinite(original_fit[best].values)].copy()
	Vflux_log = original_fit[best].values
	Vflux = 10**(Vflux_log)
	others = [c for c in ref_cols if c != best]
	if len(others) > 0 and np.any(np.isfinite(original_fit[others[0]].values)):
		BVflux = 10**(original_fit[best].values) + 10**(original_fit[others[0]].values)
	else:
		BVflux = np.copy(Vflux)
	mjd_fit = t0 + 10**(original_fit['Log_Phase'].values)
	ok = ~np.isnan(BVflux)
	peak_MJD = mjd_fit[ok][np.argmax(BVflux[ok])]

	# Same as original: absolute MJD = prior phase (days relative to peak) + peak MJD
	absolute_MJD = phase_prior + peak_MJD
	phase_gp = np.log10(np.maximum(absolute_MJD - t0, 1e-12))
	if _coord_mode(gn) == "zscore":
		phase_prior_norm = (phase_gp - float(gn["x2_mean"])) / float(gn["x2_std"])
		wls_prior_norm = (np.log10(wls_prior) - float(gn["x1_mean"])) / float(gn["x1_std"])
	else:
		phase_prior_norm = (phase_gp - float(gn["offset2"])) / float(gn["norm2"])
		wls_prior_norm = np.log10(wls_prior) / float(gn["norm1"])

	reshaped_color_prior = color_prior.reshape(len(np.unique(wls_prior)), len(np.unique(phase_prior)))
	Vflux_phase = np.interp(np.unique(phase_prior), mjd_fit - peak_MJD, Vflux)

	flux_prior = reshaped_color_prior * Vflux_phase
	flux_prior_transform = (np.log(np.clip(flux_prior, 1e-300, None)) - offset) / scale_factor

	points = np.array([tup for tup in zip(wls_prior_norm, phase_prior_norm)])
	values = (flux_prior_transform).reshape(len(np.unique(phase_prior))*len(np.unique(wls_prior)))
	return points, values


def gp_dense_matrix_bytes_order_of_magnitude(n_train):
	"""Order-of-magnitude RAM if the GP used a dense N×N float64 covariance (George does not, but Cholesky cost ~ O(N³) scales similarly)."""
	n = int(n_train)
	if n < 0:
		raise ValueError("n_train must be non-negative")
	return 8 * n * n


def augment_2dgp_training_t0_anchor(
	x1_data_norm,
	x2_data_norm,
	y_data_nonan,
	y_data_nonan_err,
	grid_norm_info,
	*,
	log_phase_anchor: float,
	log10_flux_cap: float,
	log10_flux_err: float,
):
	"""Append one pseudo point per **unique** training ``x1`` at early log10(phase days).

	Flux is a capped log10 linear flux mapped into the same scaled ln-flux space as
	``transform2LOG_reshape`` (faint = large negative log10 F).
	"""
	LN10 = np.log(10.0)
	offset = float(grid_norm_info["offset"])
	sf = float(grid_norm_info["scale_factor"])
	ln_flux = float(log10_flux_cap) * LN10
	y_cap = (ln_flux - offset) / sf
	sigma = float(log10_flux_err) * LN10 / sf
	if _coord_mode(grid_norm_info) == "zscore":
		x2a = (float(log_phase_anchor) - float(grid_norm_info["x2_mean"])) / float(
			grid_norm_info["x2_std"]
		)
	else:
		x2a = (float(log_phase_anchor) - float(grid_norm_info["offset2"])) / float(
			grid_norm_info["norm2"]
		)
	u1 = np.unique(np.asarray(x1_data_norm, dtype=float))
	if u1.size == 0:
		return x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err
	x1_add = u1.astype(float)
	x2_add = np.full(u1.shape, x2a, dtype=float)
	y_add = np.full(u1.shape, y_cap, dtype=float)
	yerr_add = np.full(u1.shape, sigma, dtype=float)
	return (
		np.concatenate([np.asarray(x1_data_norm, dtype=float), x1_add]),
		np.concatenate([np.asarray(x2_data_norm, dtype=float), x2_add]),
		np.concatenate([np.asarray(y_data_nonan, dtype=float), y_add]),
		np.concatenate([np.asarray(y_data_nonan_err, dtype=float), yerr_add]),
	)


def run_2DGP_GRID(GP2DIM_Class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm,\
		kernel_wls_scale, kernel_time_scale, extrap_mjds, prior=False, points=np.nan, values=np.nan):
	
	""" ## for NUV extention:   extrap_mjds = grid_ext_columns
	## for spectra augmentation: 
	extrap_mjds = grid_ext.columns.values
	 if (len(extrap_mjds)>200):
		 extrap_mjds = grid_ext.columns.values[:200]
	 if (max(extrap_mjds-min(extrap_mjds))>200):
		 extrap_mjds = extrap_mjds[extrap_mjds-min(extrap_mjds)<200]
	 
	 tot_iteration = int(len(extrap_mjds)/slot_size+1)
	 print (tot_iteration)"""

	_log_progress = getattr(GP2DIM_Class, "gp_predict_progress", True)

	def _gp_log(msg):
		print(msg, flush=True)

	# TRAINING: X, y, terr
	gn = GP2DIM_Class.grid_norm_info

	if bool(getattr(GP2DIM_Class, "gp_2d_anchor_t0", False)):
		_nu_before = int(np.unique(np.asarray(x1_data_norm, dtype=float)).size)
		x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err = augment_2dgp_training_t0_anchor(
			x1_data_norm,
			x2_data_norm,
			y_data_nonan,
			y_data_nonan_err,
			GP2DIM_Class.grid_norm_info,
			log_phase_anchor=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log_phase", -8.0)),
			log10_flux_cap=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_cap", -50.0)),
			log10_flux_err=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_err", 2.0)),
		)
		if _log_progress:
			_gp_log(
				"[run_2DGP_GRID] 2D t0-anchor training: +%i pseudo points (%i unique x1 nodes)"
				% (_nu_before, _nu_before)
			)

	if prior:
		from george.modeling import Model

		class Model_2dim(Model):
			parameter_names = ()
			def get_value(self, t):
				verbose = getattr(GP2DIM_Class, 'verbose', False)
				if verbose:
					print("t shape:", t.shape)
					print("t contents:", t)
				points_eval = np.array([tup for tup in zip(t[:,0], t[:,1])])
				if points_eval.size == 0 and verbose:
					print("Warning: points_eval is empty!")
				grid_z1 = griddata(points, values, points_eval, method='nearest')
				if verbose:
					print("grid_z1 contains NaN:", np.any(np.isnan(grid_z1)))
				grid_z1[np.isnan(grid_z1)] = 0.
				#plt.plot(t[:,0]*norm1, grid_z1, '-b', label='PRIOR')
				return grid_z1
    	
		mean_model = Model_2dim()

	X = np.vstack((x1_data_norm, x2_data_norm)).T
	y = y_data_nonan
	yerr = _apply_training_yerr_floors(GP2DIM_Class, y, y_data_nonan_err, stage="compute")

	_n_train = len(y)
	_gp_log("[run_2DGP_GRID] starting (prior=%r) N_train=%i" % (bool(prior), _n_train))
	if getattr(GP2DIM_Class, "verbose", False) or getattr(GP2DIM_Class, "gp_print_training_size", True):
		_gp_log(
			"[run_2DGP_GRID] N_train = %i finite training points (scaled ln-space); X.shape=%s dtype=%s"
			% (_n_train, X.shape, X.dtype)
		)
		n2b = gp_dense_matrix_bytes_order_of_magnitude(_n_train)
		_gp_log(
			"[run_2DGP_GRID] rough dense N×N float64 footprint ~ %.2f GB (hint only; George uses factorization ~O(N³) time)"
			% (n2b / (1024.0 ** 3),)
		)
		if _n_train >= 12000:
			_gp_log(
				"[run_2DGP_GRID] WARNING: N_train is very large — expect long runtime and high RAM use. "
				"Reduce training density (e.g. larger DELTA in the grid builder) if the kernel dies."
			)

	#kernel_mix = Matern32Kernel([kernel_wls_scale, kernel_time_scale], ndim=2)
	k_wave = Matern32Kernel(metric=kernel_wls_scale, ndim=2, axes=1) # wavelength axis
	k_time = Matern32Kernel(metric=kernel_time_scale, ndim=2, axes=0) # time axis
	kernel_mix = k_wave * k_time
	kernel2dim = np.var(y) * kernel_mix
	_gp_wn = float(getattr(GP2DIM_Class, "gp_white_noise", 0.0))
	# George >=0.4: homogeneous jitter via GP(white_noise=ln(variance)); legacy code used
	# kernels.WhiteKernel(c) with c the *variance* added on the diagonal (not log).
	_gp_extra = {}
	if _gp_wn > 0.0:
		_gp_extra["white_noise"] = float(np.log(_gp_wn))

	if prior:
		gp = george.GP(kernel2dim, mean=mean_model, **_gp_extra)
	else:
		gp = george.GP(kernel2dim, **_gp_extra)

	_gp_log("[run_2DGP_GRID] calling gp.compute (no progress inside; may take minutes) …")
	_t0 = time.perf_counter()
	gp.compute(X, np.sqrt(yerr**2 + 1e-6**2))
	_gp_log("[run_2DGP_GRID] gp.compute finished in %.1f s" % (time.perf_counter() - _t0,))
		
	# wls_normed_range = np.sort(np.concatenate(( np.arange(1600.,3000., 40),
	# 										  np.arange(3000.,9000., 10),
	# 										  np.arange(9000.,10350., 40))))/GP2DIM_Class.grid_norm_info['norm1']
	#RAV added this
	# wls_min = np.min(GP2DIM_Class.grids[0])
	# wls_max = np.max(GP2DIM_Class.grids[0])
	# wls_normed_range = np.arange(wls_min, wls_max + 1, 40) / GP2DIM_Class.grid_norm_info['norm1']

	wls_min = float(np.min(GP2DIM_Class.grids[0]))
	wls_max = float(np.max(GP2DIM_Class.grids[0]))
	_wl_min_a = getattr(GP2DIM_Class, "pipeline_wl_min_a", None)
	_wl_max_a = getattr(GP2DIM_Class, "pipeline_wl_max_a", None)
	if _wl_min_a is not None:
		wls_min = min(wls_min, float(np.log10(float(_wl_min_a))))
	if _wl_max_a is not None:
		wls_max = max(wls_max, float(np.log10(float(_wl_max_a))))
	span_wl = float(wls_max - wls_min)
	if span_wl <= 0.0:
		raise ValueError("run_2DGP_GRID: invalid log10(wavelength) span (wls_max <= wls_min).")

	# Prediction grid in log10(lambda): cap count so gp.predict memory stays bounded.
	# Old code used np.arange(..., 0.005) which could create ~10^3 points per phase; combined with
	# return_cov=True that allocated (Ntest x Ntest) doubles per batch and often OOM-killed the kernel.
	# Defaults tuned for kilonova SED speed vs ~0.5–1% sampling in lambda (see notebook / LOGSPACE_PIPELINE_PLAN)
	_gp_n_wl = int(getattr(GP2DIM_Class, "gp_predict_n_wavelength", 300))
	_wl_step = float(getattr(GP2DIM_Class, "gp_predict_wl_step", 0.01))
	n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
	n_wl_use = max(2, min(_gp_n_wl, n_from_step))
	wls_log_grid = np.linspace(wls_min, wls_max, n_wl_use)
	if _coord_mode(gn) == "zscore":
		wls_normed_range = (wls_log_grid - float(gn["x1_mean"])) / float(gn["x1_std"])
	else:
		wls_normed_range = wls_log_grid / float(gn["norm1"])

	#mu_fill_resh = []
	mu_fill_resh = np.empty((0, 3))
	std_fill_resh = []
	
	slot_size = max(1, int(getattr(GP2DIM_Class, "gp_predict_slot_size", 3)))
	extrap_mjds = np.asarray(extrap_mjds, dtype=float)
	if extrap_mjds.size == 0:
		raise ValueError("run_2DGP_GRID: extrap_mjds is empty (no phase columns to predict).")
	_dense_on = bool(getattr(GP2DIM_Class, "gp_predict_dense_log_phase", False))
	_dense_n = int(getattr(GP2DIM_Class, "gp_predict_dense_log_phase_n", 64))
	if _dense_on:
		_phase_before = extrap_mjds.copy()
		extrap_mjds = merge_extrap_mjds_dense_log_phase(extrap_mjds, _dense_n)
		if _log_progress:
			_gp_log(
				"[run_2DGP_GRID] dense log-phase prediction: %i → %i columns (n_dense=%i)"
				% (len(_phase_before), len(extrap_mjds), _dense_n)
			)
			log_prediction_phase_coverage(_phase_before, label="phase columns (before dense merge)")
			log_prediction_phase_coverage(extrap_mjds, label="phase columns (after dense merge)")
	tot_iteration = max(1, int(len(extrap_mjds) / slot_size + 1))
	frac_tot_iteration = 0
	if _log_progress:
		_gp_log(
			"[run_2DGP_GRID] predict grid: n_wavelength=%i | extrap phase columns=%i | slot_size=%i | outer loops=%i"
			% (n_wl_use, len(extrap_mjds), slot_size, tot_iteration)
		)

	for j in range(tot_iteration):
		if _coord_mode(gn) == "zscore":
			mjd_normed_range = (
				(extrap_mjds[j * slot_size : (j + 1) * slot_size]) - float(gn["x2_mean"])
			) / float(gn["x2_std"])
		else:
			mjd_normed_range = (
				(extrap_mjds[j * slot_size : (j + 1) * slot_size]) - float(gn["offset2"])
			) / float(gn["norm2"])
		x1_fill = []#np.random.permutation(np.linspace(0,1., N))
		x2_fill = []#np.random.permutation(np.linspace(0,1., N))
		for i in wls_normed_range:
			for k in mjd_normed_range:
				x1_fill.append(i)
				x2_fill.append(k)
		
		x1_fill=np.array(x1_fill) 
		x2_fill=np.array(x2_fill)
		
		X_fill = np.vstack((x1_fill, x2_fill)).T	
		# return_var=True: diagonal only (avoids Ntest^2 covariance allocation; same mean as return_cov)
		# Chunk test points so each predict() stays small (avoids peak RAM / solver edge cases)
		_chunk = max(200, int(getattr(GP2DIM_Class, "gp_predict_chunk_size", 1500)))
		n_pred = len(X_fill)
		if _log_progress or getattr(GP2DIM_Class, "verbose", False):
			_n_chunk_outer = int(np.ceil(n_pred / float(_chunk)))
			_gp_log(
				"[run_2DGP_GRID] predict slot %i / %i | n_pred=%i | chunk_size=%i (~%i chunks)"
				% (j + 1, tot_iteration, n_pred, _chunk, _n_chunk_outer)
			)
		frac_tot_iteration = int(20.0 * (j + 1) / tot_iteration)
		#print('[','*'*frac_tot_iteration,' '*(20-frac_tot_iteration),']' + ' %i of %i'%(slot_size*(j+1),slot_size*tot_iteration)+' spec extrapolated', end='\r')
		mu_iter = np.empty(n_pred, dtype=float)
		var_iter = np.empty(n_pred, dtype=float)
		_log_chunks = bool(
			getattr(GP2DIM_Class, "gp_predict_log_chunks", False)
			or getattr(GP2DIM_Class, "verbose", False)
		)
		_chunk_idx = 0
		for s0 in range(0, n_pred, _chunk):
			s1 = min(s0 + _chunk, n_pred)
			_tc0 = time.perf_counter()
			m_sub, v_sub = gp.predict(y, X_fill[s0:s1], return_var=True)
			mu_iter[s0:s1] = m_sub
			var_iter[s0:s1] = v_sub
			_chunk_idx += 1
			if _log_chunks:
				_gp_log(
					"  [run_2DGP_GRID] chunk %i rows %i:%i done in %.2f s"
					% (_chunk_idx, s0, s1, time.perf_counter() - _tc0)
				)
		std_iter = np.sqrt(np.maximum(var_iter, 0.0))

		if getattr(GP2DIM_Class, "gp_diagnostic_slices", False) and j == 0:
			_diag_dir = GP2DIM_Class.save_plot_path
			os.makedirs(_diag_dir, exist_ok=True)
			n_phase_plot = min(3, len(mjd_normed_range))
			for pi in range(n_phase_plot):
				mj = float(mjd_normed_range[pi])
				mask = np.isclose(x2_fill, mj, rtol=0.0, atol=1e-12)
				if not np.any(mask):
					continue
				fig_d, ax_d = plt.subplots(figsize=(8, 2))
				ax_d.plot(log10_wavelength_from_x1_norm(x1_fill[mask], gn), mu_iter[mask], "-k", label="PREDICTION")
				if prior and isinstance(points, np.ndarray) and isinstance(values, np.ndarray) and points.size and values.size:
					pe = np.column_stack((x1_fill[mask], x2_fill[mask]))
					gz = griddata(points, values, pe, method="nearest")
					gz = np.where(np.isnan(gz), 0.0, gz)
					ax_d.plot(log10_wavelength_from_x1_norm(x1_fill[mask], gn), gz, "-b", label="PRIOR")
				ax_d.set_xlabel("log10(wavelength)")
				ax_d.set_ylabel("scaled ln-flux (GP space)")
				ax_d.legend(loc="best", fontsize=8)
				fig_d.savefig(
					os.path.join(_diag_dir, "gp_diag_slot0_phase%i.pdf" % pi),
					bbox_inches="tight",
				)
				plt.close(fig_d)

		mu_resh_iter = mu_iter.reshape(len(wls_normed_range), len(mjd_normed_range))
		std_resh_iter = std_iter.reshape(len(wls_normed_range), len(mjd_normed_range))

		#if mu_fill_resh==[]:
		if mu_fill_resh.size == 0:
			mu_fill_resh = np.copy(mu_resh_iter)
			std_fill_resh = np.copy(std_resh_iter)
		else:
			mu_fill_resh = np.concatenate([mu_fill_resh, mu_resh_iter], axis=1)
			std_fill_resh = np.concatenate([std_fill_resh, std_resh_iter], axis=1)

	_gp_log('[' + '*'*frac_tot_iteration + ' '*(20-frac_tot_iteration) + '] '
		+ '%i of %i' % (min(slot_size * (j + 1), len(extrap_mjds)), len(extrap_mjds)) + ' spec extrapolated')
	mu_fill = mu_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))
	std_fill = std_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))

	if _coord_mode(gn) == "zscore":
		mjd_normed_range = (extrap_mjds - float(gn["x2_mean"])) / float(gn["x2_std"])
	else:
		mjd_normed_range = (extrap_mjds - float(gn["offset2"])) / float(gn["norm2"])
	
	x1_fill = []#np.random.permutation(np.linspace(0,1., N))
	x2_fill = []#np.random.permutation(np.linspace(0,1., N))
	for i in wls_normed_range:
		for k in mjd_normed_range:
			x1_fill.append(i)
			x2_fill.append(k)
	
	x1_fill=np.array(x1_fill) 
	x2_fill=np.array(x2_fill)
	
	_gp_log('EXTENDING SPECTRA BETWEEN:')
	_gp_log(
		'log10(wavelength): %s %s'
		% (
			float(np.min(log10_wavelength_from_x1_norm(x1_fill, gn))),
			float(np.max(log10_wavelength_from_x1_norm(x1_fill, gn))),
		)
	)
	_log_ph = _x2_norm_to_log10_phase(x2_fill, gn)
	_gp_log(
		'log10(phase days): %s %s'
		% (float(np.min(_log_ph)), float(np.max(_log_ph)))
	)
	_gp_log("[run_2DGP_GRID] done.")
	_gp_log("[run_2DGP_GRID] gp parameter vector names: %s" % (gp.get_parameter_names(),))
	_gp_log("[run_2DGP_GRID] gp parameter vector: %s" % (gp.get_parameter_vector(),))

	maybe_save_gp_minimal_export(
		GP2DIM_Class,
		X=X,
		y=y,
		yerr=yerr,
		y_compute=np.sqrt(yerr ** 2 + 1.0e-6 ** 2),
		x1_fill=x1_fill,
		x2_fill=x2_fill,
		kernel_wls_scale=kernel_wls_scale,
		kernel_time_scale=kernel_time_scale,
		prior=prior,
		points=points,
		values=values,
		grid_norm_info=GP2DIM_Class.grid_norm_info,
		gp_module="GP2dim_utils",
		kernel_layout="per_axis_Matern32_product",
	)

	return (x1_fill, x2_fill, mu_fill, std_fill)


def make_results_plots(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill):
	gn = GP2DIM_Class.grid_norm_info
	offset = gn["offset"]
	scale_factor = gn["scale_factor"]
	x1m, x1s, x2m, x2s = _legacy_axis_params(gn)

	#plt.scatter(norm2*x2_fill, norm1*x1_fill, marker='.', c=mu_fill, alpha=1., 
	#		vmin=0., cmap = mycmap)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.xlabel('MJD')
	#plt.ylabel('wls')
	#plt.colorbar()
	
	# PLOT xWLS LC and check how smooth the time variation in each single wls is:
	fit_wls = (np.unique(x1_fill)[::10])
	len_wls = len(fit_wls)
	color=cycle(plt.cm.gnuplot(np.linspace(0.05,0.95,len_wls)))
	
	fig = plt.figure(figsize=(10,6))
	plt.subplot(221)
	plt.title('log10(wl): %.3f–%.3f'%(min(x1m + x1s * fit_wls[:int(len_wls/4)]),max(x1m + x1s * fit_wls[:int(len_wls/4)])))
	for i in fit_wls[:int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(222)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[int(len_wls/4):2*int(len_wls/4)]),max(x1m + x1s * fit_wls[int(len_wls/4):2*int(len_wls/4)])))
	for i in fit_wls[int(len_wls/4):2*int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(223)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[2*int(len_wls/4):3*int(len_wls/4)]),max(x1m + x1s * fit_wls[2*int(len_wls/4):3*int(len_wls/4)])))
	for i in fit_wls[2*int(len_wls/4):3*int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(224)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[3*int(len_wls/4):int(len_wls)]),max(x1m + x1s * fit_wls[3*int(len_wls/4):int(len_wls)])))
	for i in fit_wls[3*int(len_wls/4):int(len_wls)]:
	
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	fig.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, "gp_results_wavelength_slices.pdf"),
		bbox_inches="tight",
	)
	plt.show()
	plt.close(fig)

	# Linear phase (days) × linear flux (no log y); compare to log-y PDF above if dynamic range is large
	color2 = cycle(plt.cm.gnuplot(np.linspace(0.05, 0.95, len_wls)))
	fig_lin = plt.figure(figsize=(10, 6))
	fig_lin.suptitle(
		'Linear phase (days) and linear flux — y-range may look compressed vs gp_results_wavelength_slices.pdf',
		fontsize=9,
		y=1.02,
	)
	plt.subplot(221)
	plt.title('log10(wl): %.3f–%.3f' % (min(x1m + x1s * fit_wls[: int(len_wls / 4)]), max(x1m + x1s * fit_wls[: int(len_wls / 4)])))
	for i in fit_wls[: int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(222)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]),
			max(x1m + x1s * fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]),
		)
	)
	for i in fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(223)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]),
			max(x1m + x1s * fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]),
		)
	)
	for i in fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )

		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(224)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[3 * int(len_wls / 4) : int(len_wls)]),
			max(x1m + x1s * fit_wls[3 * int(len_wls / 4) : int(len_wls)]),
		)
	)
	for i in fit_wls[3 * int(len_wls / 4) : int(len_wls)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.tight_layout(rect=[0, 0, 1, 0.92])
	fig_lin.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, "gp_results_wavelength_slices_linear_phase_linear_flux.pdf"),
		bbox_inches="tight",
	)
	plt.show()
	plt.close(fig_lin)


def transform_back_andPlot(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill, y_data_nonan):

	gn = GP2DIM_Class.grid_norm_info
	offset = gn["offset"]
	scale_factor = gn["scale_factor"]
	x1m, x1s, x2m, x2s = _legacy_axis_params(gn)
	x2_train_min = float(gn.get("x2_train_min", x2m))

	mu_fill_conv = scaled_ln_to_linear(mu_fill, offset, scale_factor)
	std_fill_conv = np.abs(scale_factor * mu_fill_conv * std_fill)

	y_data_conv = scaled_ln_to_linear(y_data_nonan, offset, scale_factor)
	#else:
	#	mu_fill_conv = (mu_fill*scale_factor + offset)
	#	std_fill_conv = np.abs( scale_factor*std_fill )
	#
	#	y_data_conv =(y_data_nonan*scale_factor + offset)
		
	fig = plt.figure(1, figsize=(10,3))
	x2_fill_phase = x2m + x2s * x2_fill
	plt.subplot(121)
	plt.scatter(x2_fill_phase, x1m + x1s * x1_fill, marker='s', s=0.5,  c=mu_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	plt.colorbar()
	
	plt.subplot(122)
	plt.scatter(x2_fill_phase, x1m + x1s * x1_fill, marker='s', s=0.5,  c=std_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	plt.colorbar()
	plt.show()
	fig.savefig(GP2DIM_Class.save_plot_path+'/2d_surface.png', bbox_inches='tight')
	plt.close(fig)

	phase_lin_days = np.power(10.0, x2_fill_phase)
	wl_lin_angstrom = np.power(10.0, x1m + x1s * x1_fill)
	fig_lin = plt.figure(figsize=(10, 3))
	plt.subplot(121)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=0.5, c=mu_fill_conv, alpha=1.,
				vmin=0., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb = plt.colorbar()
	cb.set_label('Linear flux')
	plt.subplot(122)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=0.5, c=std_fill_conv, alpha=1.,
				vmin=0., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb2 = plt.colorbar()
	cb2.set_label('Std (linear)')
	_splin = GP2DIM_Class.save_plot_path
	fig_lin.savefig(os.path.join(_splin, 'gp_2d_surface_linear_axes.pdf'), bbox_inches='tight')
	fig_lin.savefig(os.path.join(_splin, 'gp_2d_surface_linear_axes.png'), bbox_inches='tight')
	plt.show()
	plt.close(fig_lin)

	fig_mu = plt.figure(figsize=(10, 3))
	plt.subplot(121)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=mu_fill, alpha=1., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb_m = plt.colorbar()
	cb_m.set_label('Scaled ln-flux mean (GP)')
	plt.subplot(122)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=std_fill, alpha=1., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb_m2 = plt.colorbar()
	cb_m2.set_label('Scaled ln-flux std (GP)')
	fig_mu.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, 'gp_2d_surface_linear_axes_scaled_mu.pdf'),
		bbox_inches='tight',
	)
	plt.show()
	plt.close(fig_mu)
	
	max_val = np.max(y_data_conv)
	med_val = np.median(y_data_conv)
	
	#fig = plt.figure(1, figsize=(8,4))
	#spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	#scale = (max_val-med_val)/5.
	#a=0
	#mangled_original_list = GP2DIM_Class.mangledspec_list
	#
	#for j in range(len(GP2DIM_Class.get_spec_mjd())):
	#	mj = spec_mjd_list[j]
	#	spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
	#	a +=1
	#	#mask = x2_fill==(mj-offset2)/norm2
	#	#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
	#	#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
	#	plt.plot(spec_file_original['wls'], spec_file_original['flux']+(a-1)*scale,
	#			 label='Raw spec %i'%(mj-offset2), lw=1.0, color='k')
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=2, color=PyCoCo_info.color_dict[b])
	#plt.xlim(1600,11000)
	#plt.title(GP2DIM_Class.snname)
	#plt.xlabel('Wavelength')
	#plt.ylabel('Calibrated Flux + offset')
	#fig.savefig(GP2DIM_Class.save_plot_path+'/to_be_extended_spec1.pdf', bbox_inches='tight')
	#plt.show()
	#plt.close(fig)
	
	fig = plt.figure(1, figsize=(8,5))

	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, gn)
		plt.plot(x1m + x1s * x1_fill[mask], mu_fill_conv[mask]+(a-1)*scale, 
				 label='Extrapolated %.2f'%(mj-x2_train_min), lw=0.8, color='r')
		plt.fill_between(x1m + x1s * x1_fill[mask], (mu_fill_conv[mask]-std_fill_conv[mask])+(a-1)*scale , 
				 (mu_fill_conv[mask]+std_fill_conv[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	
	#colors_to_replace = plt.cm.viridis(np.linspace(0, 1, len(GP2DIM_Class.avail_filters)))
	#plt.xlim(1600,11000)
	#for i, b in enumerate(GP2DIM_Class.avail_filters):
	#	plt.vlines((GP2DIM_Class.lam_eff(b)), 0, 1., linestyle='--', lw=4, label=b, color=colors_to_replace[b])
	#RAV commented this out
	#plt.xlim(1600,11000)
	for b in GP2DIM_Class.avail_filters:
		plt.vlines((10**GP2DIM_Class.lam_eff(b)), 0, 1., linestyle='--', lw=4, label=b, color=color_dict[b])

	plt.title(GP2DIM_Class.snname)
	plt.xlabel('log10(wavelength)')
	plt.ylabel('Calibrated Flux + offset (linear)')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec_LOG_SPACE.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	fig = plt.figure(1, figsize=(14,6))
	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize=13)
	plt.rc('ytick', labelsize=13)
	
	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, gn)
		plt.plot(10**(x1m + x1s * x1_fill[mask]), mu_fill_conv[mask]+(a-1)*scale, 
				 label='Extrapolated %.2f'%(mj-x2_train_min), lw=0.8, color='r')
		plt.fill_between(10**(x1m + x1s * x1_fill[mask]), (mu_fill_conv[mask]-std_fill_conv[mask])+(a-1)*scale , 
				 (mu_fill_conv[mask]+std_fill_conv[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	a=0	
	mangled_original_list = GP2DIM_Class.mangledspec_list
	
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, gn)
		#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
		#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
		# Mangled files: linear Å in ``wls``, log10 flux in ``flux`` (see 5_Mangle_spectra_KN_log save)
		wls_lin_m = mangled_wls_linear_angstrom(spec_file_original)
		flx_lin_m = mangled_flux_linear_from_log10(spec_file_original['flux'])
		plt.plot(wls_lin_m, flx_lin_m+(a-1)*scale,
				label='Raw spec %.2f'%(mj-x2_train_min), lw=1, color='k')
	
	#plt.xlim(1600,11000)
	
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=4, color=PyCoCo_info.color_dict[b])
	plt.title(GP2DIM_Class.snname)
	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	return (mu_fill_conv, std_fill_conv, y_data_conv)


# list_mjds_tot = grid_ext_columns
# list_mjds_tot = extrap_mjds


def save_plots_files(GP2DIM_Class, list_mjds_tot, y_data_conv, x1_fill, x2_fill, mu_fill_conv, std_fill_conv):
	"""Write extended spectra.

	With ``save_dual_products=True`` (set in notebook 6 when using the ``twodim/<mode>/`` layout),
	writes under ``…/spliced/`` and ``…/full_gp/`` beneath ``save_plot_path``, plus optional
	``…/diagnostics/gp_vs_spliced_<phase>.pdf``. Default ``save_dual_products`` is False (single
	flat directory, legacy behavior).
	"""
	_base = GP2DIM_Class.save_plot_path
	_dual = bool(getattr(GP2DIM_Class, "save_dual_products", False))
	if _dual:
		_sub_sp = getattr(GP2DIM_Class, "subdir_spliced", "spliced")
		_sub_fg = getattr(GP2DIM_Class, "subdir_full_gp", "full_gp")
		_sub_dg = getattr(GP2DIM_Class, "subdir_diagnostics", "diagnostics")
		dir_spliced = os.path.join(_base, _sub_sp)
		dir_full_gp = os.path.join(_base, _sub_fg)
		dir_diag = os.path.join(_base, _sub_dg)
		for _d in (dir_spliced, dir_full_gp, dir_diag):
			os.makedirs(_d, exist_ok=True)
	else:
		dir_spliced = dir_full_gp = _base
		dir_diag = None

	gn = GP2DIM_Class.grid_norm_info
	x1m, x1s, x2m, x2s = _legacy_axis_params(gn)
	x2_train_min = float(gn.get("x2_train_min", x2m))

	def _write_spec_txt(out_dir, mj, fname_suffix, wls_a, flx, flxerr):
		pth = os.path.join(out_dir, '%.6f%s' % (float(mj), fname_suffix))
		with open(pth, 'w') as fout:
			fout.write('#wls\tflux\tfluxerr\n')
			for w, f, ferr in zip(wls_a, flx, flxerr):
				fout.write('%E\t%E\t%E\n' % (w, f, ferr))

	fig = plt.figure(1, figsize=(11, 8))

	max_val = np.max(y_data_conv)
	med_val = np.median(y_data_conv)

	scale = (max_val - med_val) / 5.

	list_mjds_spec = np.array(GP2DIM_Class.get_spec_mjd())
	list_mjds_spec_file = np.array(GP2DIM_Class.mangledspec_list)

	min_mjd = min(list_mjds_tot)
	a = 0
	for j in range(len(list_mjds_tot)):
		mj = list_mjds_tot[j]
		mask = x2_mask_for_phase(x2_fill, mj, gn)
		wls = x1m + x1s * x1_fill[mask]
		smooth_ext_spec = mu_fill_conv[mask]
		smooth_ext_spec_err = std_fill_conv[mask]

		a = a - 1

		if not phases_close(mj, list_mjds_spec):
			wls_lin_out = 10 ** wls
			plt.plot(wls_lin_out, smooth_ext_spec + (a + 1) * scale, label='Extrapolated %i' % (mj - x2_train_min), lw=0.8, color='r')
			plt.fill_between(
				wls_lin_out,
				(smooth_ext_spec - smooth_ext_spec_err) + (a + 1) * scale,
				(smooth_ext_spec + smooth_ext_spec_err) + (a + 1) * scale,
				alpha=0.3, facecolor='r',
			)
			plt.text(wls_lin_out[0], (a + 1) * scale, '%.6f' % (mj - min_mjd))
			_write_spec_txt(dir_spliced, mj, '_spec_extended_FL.txt', wls_lin_out, smooth_ext_spec, smooth_ext_spec_err)
			_write_spec_txt(dir_full_gp, mj, '_spec_extended_FL.txt', wls_lin_out, smooth_ext_spec, smooth_ext_spec_err)

	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	plt.title(GP2DIM_Class.snname)
	plt.show()
	plt.close(fig)
