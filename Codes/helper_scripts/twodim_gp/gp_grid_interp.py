"""Interpolate latent GP values from a fill grid onto arbitrary (x₁, x₂) rows."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def interp_latent_gp_at_fill_rows(
    X_fill: np.ndarray,
    latent_vec: np.ndarray,
    X_rows: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Interpolate ``latent_vec`` defined on ``X_fill`` to ``X_rows`` (same 2-D coordinate space).

    Uses ``LinearNDInterpolator`` with ``NearestNDInterpolator`` fallback for points outside
    the convex hull (or non-finite linear samples), matching ``plot_bands_gp_overview``.

    Returns ``(values, n_nn_fallback)`` where ``n_nn_fallback`` counts rows that used NN fill.
    """
    X_fill = np.asarray(X_fill, dtype=float).reshape(-1, 2)
    latent_vec = np.asarray(latent_vec, dtype=float).ravel()
    X_rows = np.asarray(X_rows, dtype=float).reshape(-1, 2)
    if X_fill.shape[0] != latent_vec.shape[0]:
        raise ValueError("X_fill rows must match latent_vec length")

    lut = LinearNDInterpolator(X_fill, latent_vec)
    lut_nn = NearestNDInterpolator(X_fill, latent_vec)
    z = lut(X_rows)
    if z.ndim != 1 or z.shape[0] != X_rows.shape[0]:
        raise ValueError("interpolation shape mismatch")

    zn = lut_nn(X_rows).ravel()
    out = np.asarray(z.ravel(), dtype=float)
    bad = ~(np.isfinite(out))
    n_bad = int(np.sum(bad))
    out[bad] = zn[bad]

    bad2 = ~(np.isfinite(out))
    if np.any(bad2):
        out = np.asarray(zn, dtype=float)
        n_bad = int(out.shape[0])
    return out, n_bad
