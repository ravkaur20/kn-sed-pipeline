"""Tests for the SED time range (pconf.SED_TIME_RANGE) and the step-3 log-phase grid."""

import _bootstrap_paths  # noqa: F401
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_config as pconf
from lc_gp_fit import SNPhotometryClass

T0 = 58000.0
DATA_LO = math.log10(0.5)
DATA_HI = math.log10(25.44)


@pytest.fixture
def sed_range(monkeypatch):
    def _set(rng, unit="phase"):
        monkeypatch.setattr(pconf, "SED_TIME_RANGE", {"TestSN": rng})
        monkeypatch.setattr(pconf, "SED_TIME_RANGE_UNIT", unit)

    return _set


def test_no_entry_uses_data_range(monkeypatch):
    monkeypatch.setattr(pconf, "SED_TIME_RANGE", {})
    assert pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI) == (DATA_LO, DATA_HI)


def test_wider_phase_range_extends_with_warning(sed_range):
    sed_range((0.1, 30.0))
    with pytest.warns(UserWarning, match="extrapolate"):
        lo, hi = pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI)
    assert lo == pytest.approx(math.log10(0.1))
    assert hi == pytest.approx(math.log10(30.0))


def test_narrower_range_falls_back_to_data(sed_range):
    sed_range((1.0, 10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo, hi = pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI)
    assert (lo, hi) == (DATA_LO, DATA_HI)


def test_one_sided_range(sed_range):
    sed_range((None, 30.0))
    with pytest.warns(UserWarning):
        lo, hi = pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI)
    assert lo == DATA_LO
    assert hi == pytest.approx(math.log10(30.0))


def test_mjd_range_converted_with_t0(sed_range):
    sed_range((None, T0 + 30.0), unit="mjd")
    with pytest.warns(UserWarning):
        _, hi = pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI, t0_mjd=T0)
    assert hi == pytest.approx(math.log10(30.0))


def test_range_before_explosion_raises(sed_range):
    sed_range((T0 - 1.0, None), unit="mjd")
    with pytest.raises(ValueError, match="before explosion"):
        pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI, t0_mjd=T0)


def test_bad_unit_raises(sed_range):
    sed_range((None, 30.0), unit="days")
    with pytest.raises(ValueError, match="SED_TIME_RANGE_UNIT"):
        pconf.sed_log_phase_range("TestSN", DATA_LO, DATA_HI)


@pytest.mark.parametrize("hi", [DATA_HI, 0.3, 0.30000001, 0.305, 0.31])
def test_log_phase_grid_includes_both_ends(hi):
    grid = pconf.log_phase_grid(0.0, hi, 0.01)
    assert grid[0] == 0.0
    assert grid[-1] == hi
    assert np.all(np.diff(grid) > 0.005)


def _photometry_sn(tmp_path, spec_log_phase):
    sn = SNPhotometryClass(
        "TestSN",
        datalc_path=str(tmp_path),
        dataspec_path=str(tmp_path),
        output_dir=str(tmp_path) + "/",
        exclude_filt=[],
        anchor_t0_in_lc_gp=False,
    )
    phase = np.array([0.6, 2.0, 10.0, 25.44])
    sn.phot = pd.DataFrame(
        {"MJD": T0 + phase, "Phase": phase, "Log_Phase": np.log10(phase)}
    )
    sn.clipped_phot = sn.phot
    sn.get_spec_log_phase = lambda verbose=False: np.asarray(spec_log_phase)
    return sn


def test_grid_reaches_latest_log_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(pconf, "SED_TIME_RANGE", {})
    sn = _photometry_sn(tmp_path, np.log10([0.5, 4.0]))
    lo, hi = sn.sed_log_phase_bounds()
    grid = pconf.log_phase_grid(lo, hi)
    assert lo == pytest.approx(math.log10(0.5))  # first spectrum precedes photometry
    assert grid[-1] >= np.max(sn.phot["Log_Phase"])


def test_bounds_use_photometry_t0_for_mjd(tmp_path, sed_range):
    sed_range((None, T0 + 40.0), unit="mjd")
    sn = _photometry_sn(tmp_path, np.log10([0.5]))
    with pytest.warns(UserWarning):
        _, hi = sn.sed_log_phase_bounds()
    assert hi == pytest.approx(math.log10(40.0))
