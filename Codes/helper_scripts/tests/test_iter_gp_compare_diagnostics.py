import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest import mock

import numpy as np


class TestIterGpCompareDiagnostics(unittest.TestCase):
    @unittest.skipUnless(
        __import__("iter_gp_compare_diagnostics", fromlist=["_HAS_MPL"])._HAS_MPL,
        "matplotlib required",
    )
    def test_mangled_vs_gp_interpolates_gp_onto_prescaled_grid(self):
        import iter_gp_compare_diagnostics as igc

        wls_pre = np.linspace(4000, 8000, 200)
        wls_gp = np.linspace(4200, 7800, 12)
        chain = [
            {
                "basename": "a.dat",
                "mjd": 57983.97,
                "prescaled_log": np.full(200, -15.0),
                "mangled_log": np.full(200, -14.9),
                "extracted_log": np.full(12, -14.85),
                "demangled_log": np.full(12, -15.0),
                "wls": wls_gp,
                "wls_prescaled": wls_pre,
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            figs = os.path.join(td, "figs")
            written = igc.save_mangled_vs_gp_diagnostics(
                os.path.join(figs, "gp_vs_mangled"), chain
            )
            self.assertEqual(len(written), 1)
            self.assertTrue(os.path.isfile(written[0]))

    @unittest.skipUnless(
        __import__("iter_plot_suite", fromlist=["_HAS_MPL"])._HAS_MPL,
        "matplotlib required",
    )
    def test_individual_plot_uses_full_prescaled_wavelength_span(self):
        import iter_plot_suite as ips

        wls_pre = np.linspace(4000, 8000, 200)
        wls_gp = np.linspace(4200, 7800, 12)
        chain = [
            {
                "basename": "a.dat",
                "mjd": 57983.97,
                "mangled_log": np.full(200, -14.9),
                "extracted_log": np.full(12, -14.85),
                "wls": wls_gp,
                "wls_prescaled": wls_pre,
            }
        ]
        captured: dict[str, np.ndarray] = {}

        def _capture(wls_man, man, wls_gp, ext, **kwargs):
            captured["wls_man"] = np.asarray(wls_man, dtype=float)
            captured["wls_gp"] = np.asarray(wls_gp, dtype=float)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(ips, "_plot_gp_vs_mangled_panel", side_effect=_capture):
                ips.plot_gp_vs_mangled(td, chain)
        self.assertIn("wls_man", captured)
        self.assertAlmostEqual(float(captured["wls_man"].min()), 4000.0)
        self.assertAlmostEqual(float(captured["wls_man"].max()), 8000.0)
        self.assertEqual(captured["wls_man"].size, 200)
        self.assertEqual(captured["wls_gp"].size, 12)

    @unittest.skipUnless(
        __import__("iter_plot_suite", fromlist=["_HAS_MPL"])._HAS_MPL,
        "matplotlib required",
    )
    def test_residual_plot_uses_full_prescaled_wavelength_span(self):
        import iter_plot_suite as ips

        wls_pre = np.linspace(4000, 8000, 200)
        wls_gp = np.linspace(4200, 7800, 12)
        chain = [
            {
                "basename": "a.dat",
                "mjd": 57983.97,
                "mangled_log": np.linspace(-15.0, -14.5, 200),
                "extracted_log": np.full(12, -14.85),
                "extracted_err": np.full(12, 0.05),
                "mangled_err": np.full(200, 0.04),
                "wls": wls_gp,
                "wls_prescaled": wls_pre,
            }
        ]
        captured: dict[str, np.ndarray] = {}

        def _capture(x, y, *args, **kwargs):
            if "wls" not in captured:
                captured["wls"] = np.asarray(x, dtype=float)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_ax = mock.MagicMock()
                mock_ax.plot.side_effect = _capture
                mock_fig = mock.MagicMock()
                mock_subplots.return_value = (mock_fig, [mock_ax, mock_ax])
                with mock.patch("matplotlib.pyplot.close"):
                    with mock.patch("matplotlib.pyplot.savefig"):
                        ips.plot_residuals_vs_unc(td, chain)
        self.assertIn("wls", captured)
        self.assertAlmostEqual(float(captured["wls"].min()), 4000.0)
        self.assertAlmostEqual(float(captured["wls"].max()), 8000.0)
        self.assertEqual(captured["wls"].size, 200)


if __name__ == "__main__":
    unittest.main()
