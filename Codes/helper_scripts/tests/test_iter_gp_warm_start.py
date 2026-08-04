import _bootstrap_paths  # noqa: F401
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np



class TestIterGpWarmStart(unittest.TestCase):
    def test_warm_start_config_attached_to_spec_class(self):
        import iter_gp_grid as ig

        captured: list = []

        class CapturingSpec:
            def __init__(self, *args, **kwargs):
                self.warm_start_config_json = None
                self.grids = (np.linspace(3.5, 3.7, 3), [])
                self.mode = "extrapolate_spectra"
                captured.append(self)

            def get_spec_mjd(self):
                return []

        def _fake_prepare_grid(sn, spec_class):
            spec_class.grid_norm_info = {
                "offset": -5.0,
                "scale_factor": 1.0,
                "norm1": 4.0,
                "norm2": 2.0,
                "offset2": -3.0,
            }
            n = 4
            return (
                np.zeros((2, 2)),
                np.ones((2, 2)),
                np.zeros(2),
                np.zeros(2),
                np.array([-1.5, -1.4]),
            )

        def _fake_transform(*_a, **_k):
            n = 4
            return np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)

        def _fake_run(*_a, **_k):
            n = 4
            return (
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.ones(n) * 0.01,
                np.zeros(n),
            )

        with tempfile.TemporaryDirectory() as td:
            warm = os.path.join(td, "gp_inference_config.json")
            with open(warm, "w", encoding="utf-8") as fh:
                json.dump({"test": 1}, fh)
            GP2dim = ig.resolve_gp2dim_module()
            with patch.object(ig, "FullMangledSeries_Class", CapturingSpec):
                with patch.object(GP2dim, "prepare_grid", _fake_prepare_grid):
                    with patch.object(GP2dim, "transform2LOG_reshape", _fake_transform):
                        with patch.object(GP2dim, "setPRIOR", lambda *a, **k: (np.array([]), np.array([]))):
                            with patch("iter_gp_grid.gpiter.run_2DGP_GRID_iter", _fake_run):
                                with patch(
                                    "gp_full_spectra_export.export_full_gp_spectra",
                                    lambda *a, **k: [],
                                ):
                                    ig.run_iter_gp_fit(
                                        "SN",
                                        t0_fix=57982.0,
                                        mode="extrapolate_spectra",
                                        mangled_spectra_dir=os.path.join(td, "m"),
                                        gp_runs_dir=os.path.join(td, "gp"),
                                        output_dir=td,
                                        coco_path=td,
                                        warm_start_config_path=warm,
                                    )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].warm_start_config_json, warm)


if __name__ == "__main__":
    unittest.main()
