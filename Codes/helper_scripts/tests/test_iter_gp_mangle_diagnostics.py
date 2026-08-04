import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np



class TestIterGpMangleDiagnostics(unittest.TestCase):
    @unittest.skipUnless(
        __import__("iter_gp_mangle_diagnostics", fromlist=["_HAS_MPL"])._HAS_MPL,
        "matplotlib required",
    )
    def test_iteration_diagnostics_files(self):
        import iter_gp_mangle_diagnostics as igd

        wls = np.linspace(4000, 8000, 20)
        chain = [
            {
                "basename": "a.dat",
                "mjd": 57983.97,
                "prescaled_log": np.full(20, -15.0),
                "mangled_log": np.full(20, -14.9),
                "extracted_log": np.full(20, -14.85),
                "demangled_log": np.full(20, -15.0),
                "wls": wls,
                "wls_prescaled": wls,
                "old_mask": np.zeros(20),
            }
        ]
        gp_result = {
            "mu": np.zeros(30),
            "mu_raw": np.ones(30) * 0.5,
            "std": np.ones(30) * 0.01,
            "x1_fill": np.linspace(0.8, 0.9, 30),
            "x2_fill": np.linspace(-0.5, 0.0, 30),
            "grid_norm_info": {
                "norm1": 4.0,
                "norm2": 2.0,
                "offset2": -3.0,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            igd.save_iteration_diagnostics(
                td,
                0,
                chain_data=chain,
                old_masks={"a.dat": np.zeros(20)},
                new_masks={"a.dat": np.full(20, 0.01)},
                gp_result=gp_result,
                metrics={"iteration": 0, "max_rel_phot_err": 0.1},
            )
            figs = os.path.join(td, "figs")
            self.assertTrue(os.path.isfile(os.path.join(figs, "iteration_summary.json")))
            gp_vs = os.path.join(figs, "gp_vs_mangled")
            self.assertTrue(os.path.isdir(gp_vs))
            pdfs = [f for f in os.listdir(gp_vs) if f.endswith(".pdf")]
            self.assertGreaterEqual(len(pdfs), 1)


if __name__ == "__main__":
    unittest.main()
