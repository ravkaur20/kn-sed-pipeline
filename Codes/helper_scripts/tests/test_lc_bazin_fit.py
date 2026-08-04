"""Unit tests for lc_bazin_fit."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_config as pconf
from lc_bazin_fit import perform_bazin_fit
from lc_bazin_fit_config import default_bazin_fit_config


def test_perform_bazin_fit_returns_extrap_arrays():
    cfg = default_bazin_fit_config("AT2017gfo")
    t0 = pconf.explosion_date_mjd("AT2017gfo")
    t_ = np.array([t0 + 1.0, t0 + 2.0, t0 + 4.0, t0 + 6.0])
    flux_ = np.array([0.05, 0.15, 0.6, 1.0])
    fluxerr_ = np.full(4, 0.05)
    phase_ = t_ - t0
    mjd_ref = float(t0)

    _R, _cov, t_extrap, fit_, fit_err, t_newpts, newpts_, newpts_err, _labels, success = perform_bazin_fit(
        "AT2017gfo",
        "TestBand",
        t_,
        flux_,
        fluxerr_,
        phase_,
        mjd_ref,
        config=cfg,
        plot=False,
    )
    assert success
    assert len(t_extrap) > 0
    assert len(fit_) == len(t_extrap)
    assert len(fit_err) == len(t_extrap)
    if len(t_newpts):
        assert len(newpts_) == len(t_newpts)
