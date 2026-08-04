"""Tests for LC GP export methods (fitted_phot_logspace / fitted_phot4mangling)."""

import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_gp_fit import SNPhotometryClass


def _make_sn(tmp_path, bands=("BandA", "BandB")):
    out_root = tmp_path / "Outputs"
    out_root.mkdir()
    sn = SNPhotometryClass(
        "TestSN",
        datalc_path=str(tmp_path / "Photometry"),
        dataspec_path=str(tmp_path / "Spectroscopy"),
        output_dir=str(out_root) + "/",
        exclude_filt=[],
        anchor_t0_in_lc_gp=False,
    )
    sn.avail_filters = np.array(bands)

    log_phase_grid = np.array([0.0, 0.1, 0.2, 0.3])
    spec_log_phase = np.array([0.05, 0.15, 0.25])

    sn.fitted_phot = {}
    for band in bands:
        mu = np.array([-16.0, -15.5, -15.0, -14.5])
        std = np.full(4, 0.05)
        mu_spec = np.array([-16.1, -15.6, -15.1])
        sn.fitted_phot[band] = {
            "fit_highcadence": [log_phase_grid, mu.copy(), std.copy()],
            "fit_mjdspec": [spec_log_phase.copy(), mu_spec.copy(), std.copy()[:3]],
        }

    def _get_singlefilter(single_filter, extended_clipped=False, verbose=False):
        lp = np.array([0.05, 0.15, 0.25])
        return pd.DataFrame({"Log_Phase": lp})

    sn.get_singlefilter = _get_singlefilter
    sn.get_spec_list = lambda verbose=False: np.array(["spec1.dat", "spec2.dat", "spec3.dat"])
    sn.get_spec_mjd = lambda verbose=False: np.array([58000.0, 58001.0, 58002.0])
    sn.get_spec_log_phase = lambda verbose=False: spec_log_phase.copy()
    return sn


def test_full_fitted_LC_file_writes_logspace_table(tmp_path):
    sn = _make_sn(tmp_path)
    sn.create_results_folder()
    df = sn.full_fitted_LC_file()

    out = Path(sn.results_mainpath) / "fitted_phot_logspace_TestSN.dat"
    assert out.is_file()
    assert "Log_Phase" in df.columns
    assert "BandA_log_flux" in df.columns
    assert "BandB_log_flux_err" in df.columns

    loaded = pd.read_csv(out, sep="\t")
    assert "Log_Phase" in loaded.columns
    assert "BandA_log_flux" in loaded.columns

    mu_a = sn.fitted_phot["BandA"]["fit_highcadence"][1]
    assert np.isnan(mu_a).any()
    assert np.isfinite(mu_a).any()


def test_mangling_GPfile_writes_mangling_table(tmp_path):
    sn = _make_sn(tmp_path)
    sn.create_results_folder()
    df = sn.mangling_GPfile()

    out = Path(sn.results_mainpath) / "fitted_phot4mangling_TestSN.dat"
    assert out.is_file()
    assert "spec_mjd" in df.columns
    assert "BandA_fit_log_flux" in df.columns
    assert "BandB_inrange" in df.columns

    loaded = pd.read_csv(out, sep="\t")
    assert "spec_file" in loaded.columns
    assert "BandA_fit_log_fluxerr" in loaded.columns
