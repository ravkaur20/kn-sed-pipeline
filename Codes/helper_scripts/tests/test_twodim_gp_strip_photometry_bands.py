"""Smoke: ``strip_photometry_bands`` drops photometric rows at default rounded x₁ targets."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import TWODIM_GP


def _have_george() -> bool:
    try:
        import george  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_george(), "strip_photometry_bands imports gp_utils/george")
class TestTwodimGpStripBands(unittest.TestCase):
    def test_strip_removes_matching_photometry_only(self):
        strip_py = os.path.join(TWODIM_GP, "strip_photometry_bands.py")
        self.assertTrue(os.path.isfile(strip_py))

        lo = [-0.8767, -0.8217, -0.5, -0.8767, 0.1]
        X = np.array([[xv, -1.0] for xv in lo], dtype=np.float64)
        n = X.shape[0]
        to = ["phot"] * (n - 1) + ["spec"]
        z = dict(
            X=X,
            y=np.linspace(1, 5, n),
            yerr=np.ones(n) * 0.1,
            y_compute=np.ones(n) * 0.2,
            train_obs_class=np.asarray(to, dtype="<U8"),
            spec_bundle_id=np.arange(n, dtype=np.int32),
        )

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.npz")
            out = os.path.join(td, "out.npz")
            np.savez_compressed(inp, **z)
            cmd = [
                sys.executable,
                strip_py,
                "-i",
                inp,
                "-o",
                out,
                "--bands=-0.8767,-0.8217",
                "--round-digits=4",
            ]
            subprocess.run(cmd, cwd=TWODIM_GP, check=True)
            dd = dict(np.load(out, allow_pickle=False))
            try:
                self.assertEqual(dd["X"].shape[0], 2)
                self.assertAlmostEqual(float(dd["X"][0, 0]), -0.5)
                self.assertAlmostEqual(float(dd["X"][1, 0]), 0.1)
            finally:
                for k in dd:
                    pass


if __name__ == "__main__":
    unittest.main()
