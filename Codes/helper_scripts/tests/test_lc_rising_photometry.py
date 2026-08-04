"""Unit tests for lc_rising_photometry."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_rising_photometry import EarlyLCPhotometry


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
