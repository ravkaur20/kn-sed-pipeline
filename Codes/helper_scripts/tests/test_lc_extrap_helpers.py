"""Unit tests for lc_extrap_helpers."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_extrap_helpers import (
    bazin_forced_zero_t0_bounds_guess,
    clip_extrap_uncertainties,
    covariance_is_bad,
    decode_band,
    early_bands_stage_a,
    filters_within_explosion_window,
    normalize_bands_keep,
    phase_reference_mjd,
    pick_reference_band,
    validate_bands_keep,
)


def test_clip_extrap_abs_cap():
    ee = np.array([0.01, 10.0, np.nan])
    out = clip_extrap_uncertainties(
        ee, flux_=np.ones(5), fluxerr_=np.full(5, 0.02), abs_cap=0.5, rel_med_max=None
    )
    assert out[0] == 0.01
    assert out[1] == 0.5
    assert np.isfinite(out[2])


def test_pick_reference_band_skips_excluded():
    avail = ["Swope_V", "DECam_r", "VISTA_J"]
    exclude = {"Swope_V"}
    candidates = ("Swope_V", "Sinistro_V", "DECam_r")
    assert pick_reference_band(avail, exclude, candidates) == "DECam_r"


def test_pick_reference_band_all_excluded():
    avail = ["Swope_V", "EFOSC2_V"]
    exclude = {"Swope_V", "EFOSC2_V", "Sinistro_V", "DECam_r", "Skymapper_r"}
    candidates = ("Swope_V", "Sinistro_V", "EFOSC2_V")
    assert pick_reference_band(avail, exclude, candidates) is None


def test_filters_within_explosion_window_respects_exclude():
    phot = pd.DataFrame(
        {
            "band": ["Swope_V", "VISTA_J", "VISTA_J"],
            "MJD": [57982.6, 57982.7, 57990.0],
        }
    )
    within, outside = filters_within_explosion_window(
        phot, explosion_mjd=57982.52851852, window_days=1.5, exclude={"Swope_V"}
    )
    assert "Swope_V" not in within
    assert "Swope_V" not in outside
    assert "VISTA_J" in within


def test_filters_within_explosion_window_numpy_structured_array():
    """SNPhotometryClass.load() returns genfromtxt-style structured arrays, not DataFrames."""
    phot = np.array(
        [
            ("Swope_V", 57982.6),
            ("VISTA_J", 57982.7),
            ("VISTA_J", 57990.0),
            ("DECam_i", 57990.5),
        ],
        dtype=[("band", "U20"), ("MJD", "f8")],
    )
    within, outside = filters_within_explosion_window(
        phot, explosion_mjd=57982.52851852, window_days=1.5, exclude={"Swope_V"}
    )
    assert "Swope_V" not in within
    assert "Swope_V" not in outside
    assert "VISTA_J" in within
    assert "DECam_i" in outside


def test_early_bands_stage_a_never_returns_excluded():
    out = early_bands_stage_a(
        filters_within=["DECam_i", "Swope_V", "VISTA_J"],
        exclude={"Swope_V", "EFOSC2_V"},
        include=["UVOT_U"],
        reference_band="Sinistro_V",
        se_sne=[],
        snname="AT2017gfo",
    )
    assert "Swope_V" not in out
    assert "EFOSC2_V" not in out
    assert "DECam_i" in out
    assert "UVOT_U" in out


def test_validate_bands_keep_raises():
    with pytest.raises(ValueError, match="not in preview cache"):
        validate_bands_keep(["DECam_i", "BAD_BAND"], ["DECam_i", "VISTA_J"])


def test_phase_reference_mjd_without_reference():
    mjd = phase_reference_mjd("AT2017gfo", None, 57982.52851852, {})
    assert mjd == 57982.52851852


def test_decode_band_bytes():
    assert decode_band(b"Swope_V") == "Swope_V"


def test_bazin_forced_zero_t0_bounds_guess():
    t_ = np.array([57983.0, 57984.5, 57986.0])
    flux_ = np.array([0.2, 0.8, 1.0])
    t0_lo, t0_hi, p_t0 = bazin_forced_zero_t0_bounds_guess(t_, flux_)
    assert t0_lo <= p_t0 <= t0_hi
    assert p_t0 == 57986.0


def test_bazin_bounds_wider_than_explosion():
    """Regression: band peak after explosion must lie inside Bazin t0 bounds."""
    explosion_mjd = 57982.52851852
    t_ = np.array([57983.5, 57985.0, 57987.2])
    flux_ = np.array([0.3, 0.9, 1.0])
    t0_lo, t0_hi, p_t0 = bazin_forced_zero_t0_bounds_guess(t_, flux_)
    peak_mjd = float(t_[np.argmax(flux_)])
    assert peak_mjd > explosion_mjd
    assert t0_lo <= peak_mjd <= t0_hi
    assert t0_hi > explosion_mjd


def test_normalize_bands_keep_flattens_nested():
    assert normalize_bands_keep([["DECam_i", "VISTA_J"]]) == ["DECam_i", "VISTA_J"]
    assert normalize_bands_keep(["DECam_i", ["VISTA_J", "DECam_i"]]) == ["DECam_i", "VISTA_J"]
