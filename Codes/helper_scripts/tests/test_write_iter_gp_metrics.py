import _bootstrap_paths  # noqa: F401
import json
import os
import tempfile
import unittest

import numpy as np


class TestWriteIterGpMetrics(unittest.TestCase):
    def test_collect_and_write_metrics(self):
        from diagnostics.write_iter_gp_metrics import run_iter_gp_metrics

        with tempfile.TemporaryDirectory() as td:
            iter_root = os.path.join(td, "twodim_iter")
            gp_runs = os.path.join(iter_root, "iter_00", "gp_runs")
            os.makedirs(gp_runs)
            cfg = {
                "config": {"metric_t": 0.1, "sigma_phot": 0.012, "sigma_spec": 0.005},
                "log_likelihood_at_compute": -100.0,
                "n_phot": 10,
                "n_spec": 20,
            }
            with open(os.path.join(gp_runs, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            out = os.path.join(iter_root, "metrics")
            result = run_iter_gp_metrics(iter_root, out)
            self.assertEqual(result["n_records"], 1)
            self.assertTrue(os.path.isfile(os.path.join(out, "iter_metrics.json")))
            self.assertTrue(os.path.isfile(os.path.join(out, "chi2_vs_iter.png")))


if __name__ == "__main__":
    unittest.main()
