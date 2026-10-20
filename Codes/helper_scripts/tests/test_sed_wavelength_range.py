"""Tests for the SED wavelength range (pconf.SED_WAVELENGTH_RANGE) and the 2D GP prediction grid."""

import _bootstrap_paths  # noqa: F401
import math
import os
import warnings

import numpy as np
import pytest

import pipeline_config as pconf
import GP2dim_utils as GP2dim
from mangle_spectra_log import save_mangled_spectrum
from twodim_grid_prep import (
    FullMangledSeries_Class,
    filter_wavelength_span,
    raw_photometry_bands,
)

DATA_LO = 3300.0
DATA_HI = 24500.0


@pytest.fixture
def wl_range(monkeypatch):
    def _set(rng):
        monkeypatch.setattr(pconf, "SED_WAVELENGTH_RANGE", {"TestSN": rng})

    return _set


def test_no_entry_uses_data_range(monkeypatch):
    monkeypatch.setattr(pconf, "SED_WAVELENGTH_RANGE", {})
    assert pconf.sed_wavelength_range("TestSN", DATA_LO, DATA_HI) == (DATA_LO, DATA_HI)


def test_wider_range_extends_with_warning(wl_range):
    wl_range((1000.0, 30000.0))
    with pytest.warns(UserWarning, match="extrapolate"):
        lo, hi = pconf.sed_wavelength_range("TestSN", DATA_LO, DATA_HI)
    assert (lo, hi) == (1000.0, 30000.0)


def test_narrower_range_falls_back_to_data(wl_range):
    wl_range((4000.0, 10000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo, hi = pconf.sed_wavelength_range("TestSN", DATA_LO, DATA_HI)
    assert (lo, hi) == (DATA_LO, DATA_HI)


def test_one_sided_range(wl_range):
    wl_range((2000.0, None))
    with pytest.warns(UserWarning):
        lo, hi = pconf.sed_wavelength_range("TestSN", DATA_LO, DATA_HI)
    assert (lo, hi) == (2000.0, DATA_HI)


def test_nonpositive_wavelength_raises(wl_range):
    wl_range((0.0, None))
    with pytest.raises(ValueError, match="positive"):
        pconf.sed_wavelength_range("TestSN", DATA_LO, DATA_HI)


def _write_filter(root, band, wls, trans):
    d = os.path.join(root, "GeneralFilters")
    os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, "%s.dat" % band), np.column_stack([wls, trans]))


def test_filter_span_uses_nonzero_transmission(tmp_path):
    _write_filter(tmp_path, "Blue", [1500.0, 1600.0, 2000.0, 2500.0, 2600.0], [0, 0.1, 1, 0.1, 0])
    _write_filter(tmp_path, "Red", [18000.0, 19000.0, 25000.0, 26000.0], [0, 1, 0.2, 0])
    span = filter_wavelength_span(["Blue", "Red"], str(tmp_path), "TestSN")
    assert span == (1600.0, 25000.0)


def test_filter_span_skips_missing_band(tmp_path):
    _write_filter(tmp_path, "Blue", [1600.0, 2000.0], [0.5, 1])
    with pytest.warns(UserWarning, match="No filter curve"):
        span = filter_wavelength_span(["Blue", "Missing"], str(tmp_path), "TestSN")
    assert span == (1600.0, 2000.0)
    with pytest.warns(UserWarning):
        assert filter_wavelength_span(["Missing"], str(tmp_path), "TestSN") is None


def test_raw_photometry_bands(tmp_path):
    p = tmp_path / "TestSN.dat"
    p.write_text("MJD,Flux,band\n1,1e-16,Red\n2,1e-16,Blue\n3,1e-16,Red\n")
    assert raw_photometry_bands(str(p)) == ["Blue", "Red"]
    with pytest.warns(UserWarning, match="No raw photometry"):
        assert raw_photometry_bands(str(tmp_path / "none.dat")) == []


def _grid_class(td, filters_dir, raw_path):
    for mjd in ("57983.969000", "57984.969003"):
        wls = np.linspace(4000, 8000, 20)
        save_mangled_spectrum(
            os.path.join(td, "%s_mangled_spec.txt" % mjd),
            wls, np.full(20, -15.0), np.full(20, 0.05), np.zeros(20),
        )
    spec_class = FullMangledSeries_Class(
        "TestSN",
        57982.52851852,
        output_dir=td + os.sep,
        filters_dir=filters_dir,
        mangled_spectra_dir=td,
        prepare_output_dir=False,
        raw_photometry_path=raw_path,
    )
    spec_class.plot_grid_rebin = False
    return spec_class


def test_grid_range_reaches_filter_edges_past_spectra(tmp_path, monkeypatch):
    monkeypatch.setattr(pconf, "SED_WAVELENGTH_RANGE", {})
    filt = tmp_path / "Filters"
    _write_filter(filt, "Blue", [2000.0, 3000.0, 5000.0], [0.2, 1, 0])
    _write_filter(filt, "Red", [7000.0, 9000.0], [1, 0.5])
    raw = tmp_path / "TestSN.dat"
    raw.write_text("MJD,Flux,band\n57984,1e-16,Blue\n57984,1e-16,Red\n")
    spec_dir = tmp_path / "mangled"
    spec_dir.mkdir()

    spec_class = _grid_class(str(spec_dir), str(filt), str(raw))
    spec_class.grid_all_spectraltimeseries()

    # Blue edge from the filter (spectra start at 4000 A), red edge from the Red filter.
    assert (spec_class.sed_wl_min_a, spec_class.sed_wl_max_a) == (2000.0, 9000.0)
    lo, hi = GP2dim.prediction_log_wavelength_range(spec_class)
    assert lo == pytest.approx(math.log10(2000.0))
    assert hi == pytest.approx(math.log10(9000.0))
    # The training grid still covers only the spectra.
    assert 10 ** spec_class.grids[0].min() == pytest.approx(4000.0, rel=1e-6)


def test_grid_range_without_photometry_is_spectra(tmp_path, monkeypatch):
    monkeypatch.setattr(pconf, "SED_WAVELENGTH_RANGE", {})
    spec_dir = tmp_path / "mangled"
    spec_dir.mkdir()
    spec_class = _grid_class(str(spec_dir), str(tmp_path), str(tmp_path / "none.dat"))
    with pytest.warns(UserWarning, match="No raw photometry"):
        spec_class.grid_all_spectraltimeseries()
    assert spec_class.sed_wl_min_a == pytest.approx(4000.0, rel=1e-6)
    assert spec_class.sed_wl_max_a == pytest.approx(8000.0, rel=1e-6)
