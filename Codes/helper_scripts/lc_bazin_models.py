"""Pure Bazin / power-law rise functions for early LC extrapolation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "rise_func",
    "bazin_func",
    "bazin_forced_zero_factory",
]


def rise_func(x, a, t0, n):
    f_t = np.zeros(len(x))
    f_t[x > t0] = a * (x[x > t0] - t0) ** 1.5
    return f_t


def bazin_func(x, a, t0, t_fall, t_rise, c):
    """Bazin function (1109.0948v1 Eq. 1) with constant offset."""
    arg_fall = -(x - t0) / t_fall
    arg_rise = -(x - t0) / t_rise
    return a * np.exp(arg_fall) / (1 + np.exp(arg_rise)) + c


def bazin_forced_zero_factory(t_exp: float) -> Callable[..., np.ndarray]:
    """Return Bazin(x, a, t0, t_fall, t_rise) forced to zero at ``t_exp``."""

    def bazin_forced_zero(x, a, t0, t_fall, t_rise):
        arg_fall = np.clip(-(x - t0) / t_fall, -700, 700)
        arg_rise = np.clip(-(x - t0) / t_rise, -700, 700)
        val = a * np.exp(arg_fall) / (1 + np.exp(arg_rise))

        arg_fall_exp = np.clip(-(t_exp - t0) / t_fall, -700, 700)
        arg_rise_exp = np.clip(-(t_exp - t0) / t_rise, -700, 700)
        val_exp = a * np.exp(arg_fall_exp) / (1 + np.exp(arg_rise_exp))
        return val - val_exp

    return bazin_forced_zero
