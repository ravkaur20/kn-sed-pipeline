import _bootstrap_paths  # noqa: F401
import json
import os
import sys
import tempfile
import unittest

import numpy as np


from iter_gp_predictions_export import save_iter_gp_config_json, save_iter_predictions_npz


class TestIterPredictionsExport(unittest.TestCase):
    def test_save_predictions_has_x_fill_and_mu_train(self):
        n = 12
        x1 = np.linspace(0.8, 0.9, 4)
        x2 = np.linspace(-0.5, -0.4, 3)
        x1f = np.tile(x1, 3)
        x2f = np.repeat(x2, 4)
        merged = {
            "X_fill": np.column_stack([x1f, x2f]),
            "point_class_train": np.array([0, 1, 0, 1] * 3),
            "sigma_eff_train": np.ones(n) * 0.01,
            "mu_train": np.zeros(n),
            "config_final": {"log_amp": -1.0},
            "log_likelihood": 123.0,
        }
        gn = {"norm1": 4.0, "norm2": 2.0, "offset2": -3.0}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "predictions.npz")
            save_iter_predictions_npz(
                path,
                mu=np.zeros(n),
                mu_raw=np.zeros(n),
                std=np.ones(n) * 0.01,
                x1_fill=x1f,
                x2_fill=x2f,
                grid_norm_info=gn,
                merged=merged,
            )
            save_iter_gp_config_json(os.path.join(td, "config.json"), merged=merged, grid_norm_info=gn)
            z = dict(np.load(path))
            self.assertIn("X_fill", z)
            self.assertEqual(z["X_fill"].shape[0], n)
            self.assertIn("mu_train", z)
            self.assertTrue(os.path.isfile(os.path.join(td, "config.json")))
            with open(os.path.join(td, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.assertIn("config", cfg)


class TestIterDenseGridFlag(unittest.TestCase):
    def test_apply_iter_dense_grid_flags(self):
        import iter_gp_grid as ig

        class S:
            pass

        s = S()
        ig._apply_iter_dense_grid_flags(s)
        self.assertTrue(s.gp_predict_dense_log_phase)
        self.assertGreaterEqual(int(s.gp_predict_dense_log_phase_n), 64)


if __name__ == "__main__":
    unittest.main()
