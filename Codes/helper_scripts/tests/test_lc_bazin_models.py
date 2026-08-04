"""Unit tests for lc_bazin_models."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_bazin_models import bazin_forced_zero_factory, bazin_func, rise_func


def test_bazin_func_finite():
    t = np.linspace(0, 50, 100)
    y = bazin_func(t, a=1.0, t0=10.0, t_fall=20.0, t_rise=5.0, c=0.0)
    assert np.all(np.isfinite(y))


def test_bazin_forced_zero_at_explosion():
    t_exp = 5.0
    fn = bazin_forced_zero_factory(t_exp)
    t = np.array([t_exp, t_exp + 10.0, t_exp + 20.0])
    y = fn(t, 1.0, 8.0, 30.0, 5.0)
    assert abs(y[0]) < 1e-12


def test_rise_func_zero_before_t0():
    t = np.array([0.0, 5.0, 10.0])
    y = rise_func(t, a=1.0, t0=5.0, n=1.5)
    assert y[0] == 0.0
    assert y[1] == 0.0
    assert y[2] > 0.0
