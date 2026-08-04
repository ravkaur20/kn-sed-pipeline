"""Numpy-only helpers for 2D GP prediction-phase grids (no george dependency)."""

from __future__ import annotations

import numpy as np


def merge_extrap_mjds_dense_log_phase(extrap_mjds, n_dense, *, decimals=12):
	"""Union phase columns with samples evenly spaced in **log10(phase days)** between min and max.

	Uses ``numpy.logspace(lo, hi, n)`` (base 10), i.e. linear days ``10**np.linspace(lo, hi, n)``,
	uniform spacing in dex. Converts with ``log10`` to match grid keys. Original phases are never
	removed (concatenate + unique).

	Parameters
	----------
	extrap_mjds : array_like
		log10(phase in days) prediction columns.
	n_dense : int
		Number of log-spaced samples inclusive; if ``< 2``, returns input unchanged.
	"""
	base = np.asarray(extrap_mjds, dtype=float).ravel()
	if base.size == 0:
		return base
	n = int(n_dense)
	if n < 2:
		return base
	lo = float(np.nanmin(base))
	hi = float(np.nanmax(base))
	if not (np.isfinite(lo) and np.isfinite(hi)):
		return base
	if lo == hi:
		return base
	t_lin = np.logspace(lo, hi, n)
	dense = np.log10(np.asarray(t_lin, dtype=float))
	merged = np.concatenate([base, dense])
	merged = np.unique(np.sort(np.round(merged.astype(float), int(decimals))))
	return merged
