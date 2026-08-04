"""Unit tests for lc_bazin_fit_config."""
import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_config as pconf
from lc_bazin_fit_config import default_bazin_fit_config, load_se_sne_names


def test_at2017gfo_default_explosion_date():
    cfg = default_bazin_fit_config("AT2017gfo")
    t0, lo, hi = cfg.explosion_dates["AT2017gfo"]
    assert t0 == pconf.explosion_date_mjd("AT2017gfo")
    assert lo is None and hi is None


def test_load_se_sne_names_on_fixture_info():
    datainfo = pconf.bootstrap_runtime().datainfo_path
    names = load_se_sne_names(datainfo)
    assert isinstance(names, list)
