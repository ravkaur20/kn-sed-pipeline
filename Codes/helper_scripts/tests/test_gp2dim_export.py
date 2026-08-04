"""Tests for ``gp2dim_export`` (no George required)."""
import _bootstrap_paths  # noqa: F401
import json
import os
import sys
import tempfile
import unittest

import numpy as np


import gp2dim_export as ex


class TestGp2dimExport(unittest.TestCase):
    def test_save_and_load_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            X = np.random.randn(40, 2)
            y = np.random.randn(40)
            yerr = np.abs(np.random.randn(40)) * 0.1 + 1e-3
            yc = np.sqrt(yerr**2 + 1e-6**2)
            Xf = np.random.randn(100, 2)
            ex.save_gp_minimal_bundle(
                td,
                X=X,
                y=y,
                yerr=yerr,
                y_compute=yc,
                X_fill=Xf,
                kernel_wls_scale=0.01,
                kernel_time_scale=0.04,
                y_var_scale=float(np.var(y)),
                white_noise_variance=0.01,
                prior=False,
                prior_points=np.zeros((0, 2)),
                prior_values=np.zeros((0,)),
                grid_norm_info={"norm1": 4.0, "offset2": -2.0, "norm2": 1.0},
                gp_module="test",
                mode="extrapolate_spectra",
                snname="TEST",
                kernel_layout="per_axis_Matern32_product",
                assign_spec_bundle_ids=False,
            )
            d = np.load(os.path.join(td, "gp_minimal_bundle.npz"), allow_pickle=False)
            self.assertEqual(d["X"].shape, (40, 2))
            self.assertEqual(d["y"].shape, (40,))
            self.assertEqual(d["X_fill"].shape, (100, 2))
            np.testing.assert_allclose(d["y_compute"], yc)
            with open(os.path.join(td, "gp_minimal_bundle_meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["snname"], "TEST")
            self.assertIn("grid_norm_info", meta)

    def test_export_includes_spec_bundle_id_and_train_obs(self):
        """Default export adds collaborator keys for ``iterate_gp_surface_bundle_scale``."""
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((32, 2))
            y = rng.standard_normal(32)
            yerr = np.abs(rng.standard_normal(32)) * 0.1 + 1e-3
            yc = np.sqrt(yerr**2 + 1e-6**2)
            Xf = rng.standard_normal((50, 2))
            gn = {
                "x1_mean": 3.5,
                "x1_std": 0.2,
                "x2_mean": 0.5,
                "x2_std": 0.4,
                "coord_parametrization": "zscore",
            }
            ex.save_gp_minimal_bundle(
                td,
                X=X,
                y=y,
                yerr=yerr,
                y_compute=yc,
                X_fill=Xf,
                kernel_wls_scale=0.01,
                kernel_time_scale=0.04,
                y_var_scale=float(np.var(y)),
                white_noise_variance=0.01,
                prior=False,
                prior_points=np.zeros((0, 2)),
                prior_values=np.zeros((0,)),
                grid_norm_info=gn,
                gp_module="test",
                mode="extrapolate_spectra",
                snname="TEST",
                kernel_layout="per_axis_Matern32_product",
                assign_spec_bundle_ids=True,
            )
            d = np.load(os.path.join(td, "gp_minimal_bundle.npz"), allow_pickle=False)
            try:
                self.assertIn("spec_bundle_id", d.files)
                self.assertIn("train_obs_class", d.files)
                self.assertEqual(d["spec_bundle_id"].shape, (32,))
                self.assertEqual(d["train_obs_class"].shape, (32,))
                self.assertEqual(d["spec_bundle_id"].dtype, np.int32)
            finally:
                d.close()


if __name__ == "__main__":
    unittest.main()
