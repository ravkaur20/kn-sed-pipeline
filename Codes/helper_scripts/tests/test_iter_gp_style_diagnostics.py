import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np



class TestIterGpStyleDiagnosticsPaths(unittest.TestCase):
    def test_plot_results_parent_tag_layout(self):
        import iter_gp_style_diagnostics as isd

        with tempfile.TemporaryDirectory() as td:
            iter_dir = os.path.join(td, "iter_00")
            gp_runs = os.path.join(iter_dir, "gp_runs")
            export = os.path.join(gp_runs, "gp_minimal_export")
            os.makedirs(export)
            np.savez_compressed(
                os.path.join(gp_runs, "predictions.npz"),
                mu=np.zeros(4),
                std=np.ones(4),
                X_fill=np.zeros((4, 2)),
            )
            with open(os.path.join(export, "gp_minimal_bundle_meta.json"), "w") as fh:
                fh.write("{}")
            np.savez_compressed(
                os.path.join(export, "gp_minimal_bundle.npz"),
                X=np.zeros((2, 2)),
                y=np.zeros(2),
                yerr=np.ones(2),
            )
            calls: list[list[str]] = []

            def _fake_run(cmd, **_kw):
                calls.append(list(cmd))
                if "--output-dir" in cmd and "--tag" in cmd:
                    pred_parent = cmd[cmd.index("--output-dir") + 1]
                    tag = cmd[cmd.index("--tag") + 1]
                    run_dir = os.path.join(pred_parent, tag)
                    figs = os.path.join(run_dir, "figs")
                    os.makedirs(figs, exist_ok=True)
                    with open(os.path.join(figs, "gp_mu_heatmap.png"), "wb") as wf:
                        wf.write(b"png")
                return type("R", (), {"returncode": 0})()

            with patch("subprocess.run", _fake_run):
                isd.run_iter_gp_style_diagnostics(
                    iter_dir,
                    gp_runs,
                    os.path.join(iter_dir, "figs"),
                    coco_path=td,
                    snname="SN",
                )
            self.assertTrue(calls)
            pr = calls[0]
            self.assertEqual(pr[pr.index("--output-dir") + 1], iter_dir)
            self.assertEqual(pr[pr.index("--tag") + 1], "gp_runs")
            self.assertIn("--heatmap-normalized", pr)
            self.assertIn("--heatmap-raw", pr)
            self.assertTrue(
                os.path.isfile(os.path.join(iter_dir, "figs", "gp_surface", "gp_mu_heatmap.png"))
            )


if __name__ == "__main__":
    unittest.main()
