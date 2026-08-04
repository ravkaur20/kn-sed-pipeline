import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np


from gp_full_spectra_export import export_full_gp_spectra


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
        x2 = np.full(3, -0.5)
        x1_fill = np.tile(x1, 2)
        x2_fill = np.concatenate([np.full(3, -0.5), np.full(3, -0.4)])
        mu = np.zeros(n)
        std = np.ones(n) * 0.01
        y = np.zeros(n)
        phases = np.array([-1.5, -1.4])
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


if __name__ == "__main__":
    unittest.main()
