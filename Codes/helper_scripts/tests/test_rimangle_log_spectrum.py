"""Unit tests for rimangle_log_spectrum (log10 extended flux on disk, linear F internally)."""
import os
import unittest

import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import CODES

import numpy as np


import rimangle_log_spectrum as rml


class TestAutoExtendedLog(unittest.TestCase):
    def test_negative_median_is_log10(self):
        w = np.linspace(3000.0, 8000.0, 10)
        f = np.full(10, -20.0)
        self.assertTrue(rml.auto_extended_flux_is_log10(w, f))

    def test_tiny_positive_is_linear(self):
        w = np.linspace(3000.0, 8000.0, 10)
        f = np.full(10, 1e-20)
        self.assertFalse(rml.auto_extended_flux_is_log10(w, f))


class TestExtendedToLinear(unittest.TestCase):
    def test_log10_roundtrip_order(self):
        w = np.array([3000.0, 4000.0], dtype=np.float64)
        logf = np.array([-15.0, -14.5])
        loge = np.array([0.1, 0.2])
        dt = [("wls", "f8"), ("flux", "f8"), ("fluxerr", "f8")]
        ext = np.empty(2, dtype=dt)
        ext["wls"] = w
        ext["flux"] = logf
        ext["fluxerr"] = loge
        lin = rml.extended_to_linear_recarray(ext, True)
        np.testing.assert_allclose(lin["wls"], w)
        np.testing.assert_allclose(lin["flux"], 10.0**logf)
        expect_err = np.abs(lin["flux"] * np.log(10.0) * loge)
        np.testing.assert_allclose(lin["fluxerr"], expect_err, rtol=1e-6)

    def test_linear_unchanged(self):
        w = np.array([3000.0, 4000.0])
        f = np.array([1e-15, 2e-15])
        e = np.array([1e-16, 2e-16])
        dt = [("wls", "f8"), ("flux", "f8"), ("fluxerr", "f8")]
        ext = np.empty(2, dtype=dt)
        ext["wls"] = w
        ext["flux"] = f
        ext["fluxerr"] = e
        lin = rml.extended_to_linear_recarray(ext, False)
        np.testing.assert_allclose(lin["flux"], f)
        np.testing.assert_allclose(lin["fluxerr"], e)

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(
                CODES,
                "..",
                "Outputs",
                "AT2017gfo",
                "FINAL_spectra_2dim",
                "57982.530000_FINAL_spec_FL.txt",
            )
        ),
        "sample AT2017gfo file not in repo",
    )
    def test_sample_at2017gfo_file_median_flux(self):
        path = os.path.join(
            CODES,
            "..",
            "Outputs",
            "AT2017gfo",
            "FINAL_spectra_2dim",
            "57982.530000_FINAL_spec_FL.txt",
        )
        d = np.genfromtxt(
            path, names=["wls", "flux", "fluxerr"], dtype=None, encoding="utf-8"
        )
        ex = d
        if not hasattr(ex, "wls") and d.dtype and d.dtype.names:
            ex = d.view(np.recarray)
        self.assertTrue(rml.auto_extended_flux_is_log10(d["wls"], d["flux"]))
        lin = rml.extended_to_linear_recarray(d, True)
        self.assertTrue(np.isfinite(lin["flux"]).all())
        self.assertGreater(float(np.nanmedian(lin["flux"])), 0.0)
        f_log, _ = rml.linear_flux_to_log10_columns(lin["flux"], lin["fluxerr"])
        np.testing.assert_allclose(f_log, d["flux"], rtol=0.01, atol=0.02)


class TestLinearToLog10Columns(unittest.TestCase):
    def test_matches_derivative(self):
        f = np.array([1e-20, 1e-10])
        fe = np.array([1e-21, 1e-11])
        fl, fel = rml.linear_flux_to_log10_columns(f, fe)
        np.testing.assert_allclose(fl, np.log10(f))
        np.testing.assert_allclose(fel, fe / (f * np.log(10.0)))


if __name__ == "__main__":
    unittest.main()
