"""Inverse-variance binning of unsmoothed spectra, in linear flux.

Unsmoothed pixels are noisy (per-pixel S/N can be ~1) and some have flux <= 0,
so nothing should take the log of a single pixel or fit a ratio to raw
pixels. These helpers combine pixels into bins first:

    flux_bin = sum(f / s^2) / sum(1 / s^2),   err_bin = 1 / sqrt(sum(1 / s^2))

using only weight-1 pixels (weight 0 = clipped in notebook 0.2). A bin is left
empty (NaN) when fewer than ``min_good_frac`` of its pixels are weight 1, or
it has no weight-1 pixels at all.

``gp_log_grid`` is the 2D GP's wavelength grid (log10 Angstrom, constant log
step set by ``DELTA`` Angstrom at the middle of the span), shared by step 5,
step 6's re-mangle and step 6's grid prep so every spectrum lands on the same
nodes.
"""

from __future__ import annotations

import numpy as np


def ivar_bin(wl, flux, err, weight, edges, min_good_frac=0.5):
    """Bin pixels into ``[edges[i], edges[i+1])`` (same units as ``wl``).

    Returns a dict of arrays, one value per bin:
        flux, err   inverse-variance mean and its error (NaN if empty)
        n_good      weight-1 pixels used
        good_frac   weight-1 pixels / all pixels in the bin (NaN if none)
    """
    wl = np.asarray(wl, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    weight = np.ones(len(wl)) if weight is None else np.asarray(weight, float)
    edges = np.asarray(edges, float)
    nb = len(edges) - 1

    idx = np.searchsorted(edges, wl, side="right") - 1
    inside = (idx >= 0) & (idx < nb)
    use = inside & (weight > 0) & np.isfinite(flux) & np.isfinite(err) & (err > 0)

    n_all = np.bincount(idx[inside], minlength=nb).astype(float)
    n_good = np.bincount(idx[use], minlength=nb).astype(float)
    iv = 1.0 / err[use] ** 2
    s_iv = np.bincount(idx[use], weights=iv, minlength=nb)
    s_fiv = np.bincount(idx[use], weights=flux[use] * iv, minlength=nb)

    with np.errstate(invalid="ignore", divide="ignore"):
        good_frac = np.where(n_all > 0, n_good / n_all, np.nan)
        ok = (n_good > 0) & (good_frac >= min_good_frac)
        fbin = np.where(ok, s_fiv / s_iv, np.nan)
        ebin = np.where(ok, 1.0 / np.sqrt(s_iv), np.nan)
    return {"flux": fbin, "err": ebin, "n_good": n_good, "good_frac": good_frac}


def gp_log_grid(wl_min_A, wl_max_A, delta_A):
    """The 2D GP wavelength grid: log10(Angstrom) nodes with a constant step
    of log10(1 + delta / lambda_mid), lambda_mid = geometric middle of the
    span (same formula step 6's grid prep has always used)."""
    lo, hi = np.log10(wl_min_A), np.log10(wl_max_A)
    mid_lin = 10.0 ** (0.5 * (lo + hi))
    step = np.log10(1.0 + delta_A / max(mid_lin, 100.0))
    return np.round(np.arange(lo, hi + step, step), 8)


def node_edges(nodes):
    """Bin edges halfway between nodes (same units), half a step past each end."""
    nodes = np.asarray(nodes, float)
    if len(nodes) == 1:
        return np.array([nodes[0] - 0.5, nodes[0] + 0.5])
    mid = 0.5 * (nodes[1:] + nodes[:-1])
    return np.concatenate([[2 * nodes[0] - mid[0]], mid, [2 * nodes[-1] - mid[-1]]])


def bin_onto_log_grid(wl_A, flux, err, weight, grid_log10, min_good_frac=0.5):
    """Bin a linear-Angstrom spectrum onto log10 grid nodes. Only nodes inside
    the spectrum's span can be filled; the rest are NaN."""
    return ivar_bin(np.log10(np.clip(np.asarray(wl_A, float), 1.0, None)),
                    flux, err, weight, node_edges(grid_log10), min_good_frac)
