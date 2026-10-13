"""Unit tests for lc_rising_photometry."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_rising_photometry import LC_COLUMNS, EarlyLCPhotometry, read_lc_columns


def test_load_and_clip_fixture():
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures"
    sn = EarlyLCPhotometry(str(fixture_dir) + "/", "test_rising_lc", exclude_filt=())
    sn.load()
    bands = sn.get_availfilter()
    assert "TestBand" in bands
    sn.clip_photometry()
    clipped = sn.clipped_phot
    assert len(clipped) >= 3
    one = sn.get_singlefilter("TestBand")
    assert len(one) >= 3


RAW_FORMAT = """MJD,Mag,Mag_err,Flux,Flux_err,band,Instr,FilterSet,MagSys
57983.0,20.1,0.05,1.0e-15,1.0e-16,TestBand,TestCam,GeneralFilters,AB
57983.005,20.0,0.05,1.1e-15,1.0e-16,TestBand,TestCam,GeneralFilters,
57984.0,19.5,0.05,2.0e-15,1.5e-16,TestBand,TestCam,GeneralFilters,Vega
57985.0,19.0,0.05,3.0e-15,2.0e-16,TestBand,TestCam,GeneralFilters,AB
57985.0,19.2,0.05,2.5e-15,2.0e-16,Other,OtherCam,GeneralFilters,AB
"""


def test_raw_step0_format_loads_and_clips(tmp_path):
    """Notebook 2 reads 1_LCs_flux_raw directly (9 columns, different order)."""
    (tmp_path / "RawSN.dat").write_text(RAW_FORMAT)
    sn = EarlyLCPhotometry(str(tmp_path) + "/", "RawSN", exclude_filt=())
    sn.load()
    assert sn.phot.dtype.names == LC_COLUMNS
    assert list(sn.phot["Flux"]) == [1.0e-15, 1.1e-15, 2.0e-15, 3.0e-15, 2.5e-15]
    assert list(sn.phot["Instr"][:1]) == ["TestCam"]
    sn.clip_photometry()
    one = sn.clipped_phot[sn.clipped_phot["band"] == "TestBand"]
    # the two TestBand points 0.005 d apart are averaged into one
    assert len(one) == 3
    assert one["FilterSet"][0] == "GeneralFilters" and one["Instr"][0] == "TestCam"


def test_old_six_column_format_still_loads():
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures"
    arr = read_lc_columns(str(fixture_dir / "test_rising_lc.dat"))
    assert arr.dtype.names == LC_COLUMNS


def test_missing_column_raises(tmp_path):
    p = tmp_path / "Bad.dat"
    p.write_text("MJD,Flux,Flux_err,band\n57983.0,1e-15,1e-16,TestBand\n")
    import pytest

    with pytest.raises(KeyError, match="FilterSet"):
        read_lc_columns(str(p))
