"""Tests for synthetic_photometry_export (AB synthetic photometry from FINAL GP spectra)."""
import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import synthetic_photometry_export as sp  # noqa: E402


def _ab_flat_spectrum(wls, mag=0.0):
    """F_lambda of a source with constant AB magnitude ``mag``."""
    return sp.AB_FNU_CGS * sp.C_AA_PER_S / wls**2 * 10 ** (-0.4 * mag)


def _tophat(lo=5000.0, hi=6000.0):
    w = np.arange(lo - 100.0, hi + 101.0, 10.0)
    t = ((w >= lo) & (w <= hi)).astype(float)
    return w, t


class TestSynthAbMag(unittest.TestCase):
    def test_flat_ab_source_both_methods(self):
        w = np.geomspace(3000.0, 25000.0, 120)
        f = _ab_flat_spectrum(w, mag=21.3)
        fw, ft = _tophat()
        for method in sp.SYNTH_METHODS:
            mag, err, cov = sp.synth_ab_mag(w, f, 0.1 * f, fw, ft, method=method)
            self.assertAlmostEqual(mag, 21.3, places=3, msg=method)
            self.assertAlmostEqual(err, 2.5 / np.log(10) * 0.1, places=3)
            self.assertAlmostEqual(cov, 1.0)

    def test_partial_coverage_is_nan(self):
        w = np.geomspace(5500.0, 25000.0, 120)
        f = _ab_flat_spectrum(w)
        fw, ft = _tophat()
        mag, err, cov = sp.synth_ab_mag(w, f, f * 0, fw, ft)
        self.assertTrue(np.isnan(mag) and np.isnan(err))
        self.assertLess(cov, 0.6)

    def test_methods_differ_for_sloped_spectrum(self):
        w = np.geomspace(3000.0, 25000.0, 400)
        f = 1e-17 * (w / 5500.0) ** -3
        fw, ft = _tophat(4000.0, 7000.0)
        m1 = sp.synth_ab_mag(w, f, f * 0, fw, ft, method="true_ab")[0]
        m2 = sp.synth_ab_mag(w, f, f * 0, fw, ft, method="wtf_band_ab")[0]
        self.assertGreater(abs(m1 - m2), 1e-3)

    def test_bad_method(self):
        w = np.geomspace(3000.0, 25000.0, 50)
        with self.assertRaises(ValueError):
            sp.synth_ab_mag(w, _ab_flat_spectrum(w), w * 0, *_tophat(), method="vega")


class TestExtinction(unittest.TestCase):
    def test_zero_ebv_is_zero(self):
        self.assertEqual(sp.nb1_band_extinction_mag(*_tophat(), ebv_mw=0.0), 0.0)

    def test_bluer_band_more_extinction(self):
        a_blue = sp.nb1_band_extinction_mag(*_tophat(4000.0, 5000.0), ebv_mw=0.1)
        a_red = sp.nb1_band_extinction_mag(*_tophat(8000.0, 9000.0), ebv_mw=0.1)
        self.assertGreater(a_blue, a_red)
        self.assertGreater(a_red, 0.0)


class TestGridAndInterpolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.final_dir = os.path.join(d, "as_observed")
        os.makedirs(self.final_dir)
        w = np.geomspace(3000.0, 25000.0, 120)
        # mag increases linearly in log10(phase): m = 20 + log10(phase)
        for lp in (-0.5, 0.0, 0.5, 1.0):
            f = _ab_flat_spectrum(w, mag=20.0 + lp)
            np.savetxt(
                os.path.join(self.final_dir, "%.6f_FINAL_spec_FL.txt" % lp),
                np.column_stack([w, f, 0.05 * f]),
                header="wls\tflux\tfluxerr",
            )
        fw, ft = _tophat()
        self.filt = os.path.join(d, "Top_hat.dat")
        np.savetxt(self.filt, np.column_stack([fw, ft]))
        self.t0 = 57982.5

    def tearDown(self):
        self.tmp.cleanup()

    def test_grid_columns_and_values(self):
        g = sp.synth_lightcurves_on_grid(
            self.final_dir, ["Top_hat"], {"Top_hat": self.filt}, t0_mjd=self.t0,
            apply_extinction=True, ext_kwargs={"ebv_mw": 0.1}, excluded_bands=["Top_hat"],
        )
        self.assertEqual(len(g), 4)
        np.testing.assert_allclose(g["mag_AB_dustcorr"], 20.0 + g["log10_phase"], atol=1e-3)
        np.testing.assert_allclose(g["mag_AB_obs"] - g["mag_AB_dustcorr"], g["A_ext_band"])
        np.testing.assert_allclose(g["MJD"], self.t0 + 10 ** g["log10_phase"])
        self.assertTrue(g["excluded_from_fit"].all())

        g2 = sp.synth_lightcurves_on_grid(
            self.final_dir, ["Top_hat"], {"Top_hat": self.filt}, t0_mjd=self.t0,
            apply_extinction=False,
        )
        self.assertIn("mag_AB_model", g2.columns)
        self.assertNotIn("mag_AB_obs", g2.columns)

    def test_interpolate_to_observations(self):
        g = sp.synth_lightcurves_on_grid(
            self.final_dir, ["Top_hat"], {"Top_hat": self.filt}, t0_mjd=self.t0,
            apply_extinction=False,
        )
        obs = pd.DataFrame({
            "MJD": self.t0 + np.array([10 ** 0.25, 10 ** 2.0, 2.0, 0.0 + 1.0]),
            "band": ["Top_hat", "Top_hat", "Other", "Top_hat"],
        })
        out = sp.interpolate_to_observations(g, obs, t0_mjd=self.t0)
        self.assertEqual(list(out["input_row"]), [0, 1, 2, 3])
        self.assertAlmostEqual(out.loc[0, "mag_AB_model"], 20.25, places=3)
        self.assertTrue(np.isnan(out.loc[1, "mag_AB_model"]))  # beyond last epoch
        self.assertTrue(np.isnan(out.loc[2, "mag_AB_model"]))  # band not synthesized
        self.assertAlmostEqual(out.loc[3, "mag_AB_model"], 20.0, places=3)


if __name__ == "__main__":
    unittest.main()
