import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np


from gp_full_spectra_export import distinct_phase_columns, export_full_gp_spectra


class _StubSpecClass:
    mode = "extrapolate_spectra"
    grid_norm_info = {
        "offset": -5.0,
        "scale_factor": 1.0,
        "norm1": 4.0,
        "norm2": 2.0,
        "offset2": -3.0,
    }

    def get_spec_mjd(self):
        return []


class _StubSpecClassZscore:
    mode = "extrapolate_spectra"
    grid_norm_info = {
        "offset": -5.0,
        "scale_factor": 1.0,
        "x1_mean": 3.2,
        "x1_std": 0.15,
        "x2_mean": -1.5,
        "x2_std": 0.4,
        "coord_parametrization": "zscore",
    }

    def get_spec_mjd(self):
        return []


class TestGpFullSpectraExport(unittest.TestCase):
    def test_writes_spec_extended_no_spliced(self):
        n = 6
        x1 = np.linspace(0.85, 0.9, 3)
        x1_fill = np.tile(x1, 2)
        mu = np.zeros(n)
        std = np.ones(n) * 0.01
        y = np.zeros(n)
        phases = np.array([-1.5, -1.4])
        gn = _StubSpecClass.grid_norm_info  # legacy coords: x2 = (phase - offset2) / norm2
        x2_fill = np.repeat((phases - gn["offset2"]) / gn["norm2"], 3)
        with tempfile.TemporaryDirectory() as td:
            paths = export_full_gp_spectra(
                _StubSpecClass(),
                x1_fill=x1_fill,
                x2_fill=x2_fill,
                mu_fill=mu,
                std_fill=std,
                grid_ext_columns=phases,
                y_data_nonan=y,
                out_dir=td,
            )
            self.assertTrue(paths)
            full_gp = os.path.join(td, "full_gp")
            self.assertTrue(os.path.isdir(full_gp))
            self.assertFalse(os.path.isdir(os.path.join(td, "spliced")))
            txts = [f for f in os.listdir(full_gp) if f.endswith(".txt")]
            self.assertGreaterEqual(len(txts), 1)
            self.assertTrue(any(f.endswith("_spec_extended_FL.txt") for f in txts))

    def test_writes_spec_extended_zscore_grid_norm_info(self):
        gn = _StubSpecClassZscore.grid_norm_info
        x1 = (np.array([3.0, 3.25, 3.5]) - gn["x1_mean"]) / gn["x1_std"]
        phases = np.array([-1.5, -1.4])
        x2_vals = (phases - gn["x2_mean"]) / gn["x2_std"]
        n = 6
        x1_fill = np.tile(x1, 2)
        x2_fill = np.concatenate([np.full(3, x2_vals[0]), np.full(3, x2_vals[1])])
        mu = np.zeros(n)
        std = np.ones(n) * 0.01
        y = np.zeros(n)
        with tempfile.TemporaryDirectory() as td:
            paths = export_full_gp_spectra(
                _StubSpecClassZscore(),
                x1_fill=x1_fill,
                x2_fill=x2_fill,
                mu_fill=mu,
                std_fill=std,
                grid_ext_columns=phases,
                y_data_nonan=y,
                out_dir=td,
            )
            self.assertTrue(paths)
            full_gp = os.path.join(td, "full_gp")
            txts = [f for f in os.listdir(full_gp) if f.endswith(".txt")]
            self.assertGreaterEqual(len(txts), 1)
            self.assertTrue(any(f.endswith("_spec_extended_FL.txt") for f in txts))

    def test_spectroscopic_epochs_are_exported(self):
        """Columns at observed-spectrum epochs get a file too (no holes in the SED)."""
        gn = _StubSpecClassZscore.grid_norm_info

        class _WithSpectra(_StubSpecClassZscore):
            def get_spec_mjd(self):
                return [-1.4]  # second column is a spectrum epoch

        x1 = (np.array([3.0, 3.25, 3.5]) - gn["x1_mean"]) / gn["x1_std"]
        phases = np.array([-1.5, -1.4, -1.3])
        x2_vals = (phases - gn["x2_mean"]) / gn["x2_std"]
        x1_fill = np.tile(x1, 3)
        x2_fill = np.repeat(x2_vals, 3)
        n = x1_fill.size
        with tempfile.TemporaryDirectory() as td:
            paths = export_full_gp_spectra(
                _WithSpectra(),
                x1_fill=x1_fill,
                x2_fill=x2_fill,
                mu_fill=np.zeros(n),
                std_fill=np.ones(n) * 0.01,
                grid_ext_columns=phases,
                y_data_nonan=np.zeros(n),
                out_dir=td,
            )
            stems = sorted(float(os.path.basename(p).split("_")[0]) for p in paths)
            np.testing.assert_allclose(stems, phases)

    def test_near_identical_columns_become_one_file(self):
        """Copies of one epoch (e.g. X-shooter arms) write a single file with one row per wavelength."""
        gn = _StubSpecClassZscore.grid_norm_info
        x1 = (np.array([3.0, 3.25, 3.5]) - gn["x1_mean"]) / gn["x1_std"]
        # -1.4 twice (1e-12 apart: share the mask) and once 2e-6 dex away: all one epoch
        phases = np.array([-1.5, -1.4, -1.4 + 1e-12, -1.4 + 2e-6])
        x2_vals = (phases - gn["x2_mean"]) / gn["x2_std"]
        x1_fill = np.tile(x1, 4)
        x2_fill = np.repeat(x2_vals, 3)
        n = x1_fill.size
        with tempfile.TemporaryDirectory() as td:
            paths = export_full_gp_spectra(
                _StubSpecClassZscore(),
                x1_fill=x1_fill,
                x2_fill=x2_fill,
                mu_fill=np.zeros(n),
                std_fill=np.ones(n) * 0.01,
                grid_ext_columns=phases,
                y_data_nonan=np.zeros(n),
                out_dir=td,
            )
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertEqual(np.loadtxt(p).shape, (3, 3))

    def test_distinct_phase_columns_keeps_close_but_separate_epochs(self):
        # 7.453 d and 7.471 d spectra are ~1e-3 dex apart: both kept
        cols = np.log10([7.45332, 7.47149, 8.47147, 8.47149])
        out = distinct_phase_columns(cols)
        np.testing.assert_allclose(10 ** out, [7.45332, 7.47149, 8.47147], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
