"""Extended TwoD / RE-mangle spectra: notebook 6 newlog format uses linear Å, log10 F in flux columns.

Internal physics (mangle, trapezoid photometry) uses linear F and linear σ(F). Use these helpers
from ``7_Rimangle_KN_log`` / ``ReMangle_SingleSpectrumClass``."""

from __future__ import annotations

import numpy as np

__all__ = [
	"auto_extended_flux_is_log10",
	"extended_to_linear_recarray",
	"linear_flux_to_log10_columns",
]

# Kept in sync with GP2dim_utils (avoid importing that module's heavy stack from rimangle only).


def _mangled_wls_max_is_linear_angstrom(wls_arr):
	"""True if wls look like linear Å (same as GP2dim_utils)."""
	w = np.asarray(wls_arr, dtype=float)
	mx = float(np.nanmax(w))
	return bool(np.isfinite(mx) and mx > 200.0)


def _mangled_wls_linear_angstrom(spec_rec):
	"""Return wavelength column in linear Å (linear or log10-Å columns)."""
	w = np.asarray(spec_rec['wls'], dtype=float)
	if _mangled_wls_max_is_linear_angstrom(w):
		return w
	return np.power(10.0, np.clip(w, -50.0, 8.0))


def _mangled_flux_linear_from_log10(flux_arr):
	"""log10 F → linear F (same as GP2dim_utils)."""
	f = np.asarray(flux_arr, dtype=float)
	return np.power(10.0, np.clip(f, -350.0, 300.0))


def auto_extended_flux_is_log10(wls, flux) -> bool:
	"""Heuristic: negative median => typical log10 F; small positive => likely linear F (see NB6 newlog)."""
	_ = wls
	f = np.asarray(flux, dtype=float)
	if not np.any(np.isfinite(f)):
		return True
	med = float(np.nanmedian(f[np.isfinite(f)]))
	if med < 0.0:
		return True
	if 0.0 < med < 1e-12:
		return False
	if med >= 0.1:
		return False
	# 1e-12 … 0.1 ambiguous; default to log10 to match newlog unless user overrides
	return True


def extended_to_linear_recarray(ext_spec, flux_in_log10: bool):
	"""Build structured array wls, flux, fluxerr with linear Å and linear F, σ(F).

	``ext_spec`` is the genfromtxt record with wls, flux, fluxerr as stored on disk.
	"""
	w = ext_spec['wls']
	f = ext_spec['flux']
	e = ext_spec['fluxerr']
	wlin = _mangled_wls_linear_angstrom({'wls': w})
	if flux_in_log10:
		flin = _mangled_flux_linear_from_log10(f)
		elin = np.abs(flin * np.log(10.0) * np.asarray(e, dtype=float))
	else:
		flin = np.asarray(f, dtype=float)
		elin = np.asarray(e, dtype=float)
	dt = [('wls', 'f8'), ('flux', 'f8'), ('fluxerr', 'f8')]
	out = np.empty(wlin.shape[0], dtype=dt)
	out['wls'] = wlin
	out['flux'] = flin
	out['fluxerr'] = elin
	return out


def linear_flux_to_log10_columns(f, fe, floor: float = 1e-300):
	"""Convert linear F, σ_F to log10 F, σ_log10 (error on dex) for on-disk log format."""
	f = np.maximum(np.asarray(f, dtype=float), float(floor))
	fe = np.asarray(fe, dtype=float)
	f_log = np.log10(f)
	fe_log = fe / (f * np.log(10.0))
	return f_log, fe_log
