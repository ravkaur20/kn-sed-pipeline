"""Tests for photometry_mag_to_flux (pipeline step 0: AB mag -> pivot-wavelength band flux)."""
import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import CODES, REPO_ROOT
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import photometry_mag_to_flux as m2f  # noqa: E402
from mangle_spectra_log import band_flux_trapz  # noqa: E402
from synthetic_photometry_export import synth_ab_mag  # noqa: E402

AB_FNU = 3631e-23


def _tophat(lo, hi, step=5.0):
    w = np.arange(lo - 50.0, hi + 51.0, step)
    return w, ((w >= lo) & (w <= hi)).astype(float)


def _ab_flat(wls, mag):
    return AB_FNU * m2f.C_AA_PER_S / wls**2 * 10 ** (-0.4 * mag)


class TestPivotAndConversion(unittest.TestCase):
    def test_pivot_tophat_analytic(self):
        a, b = 5000.0, 6000.0
        w = np.linspace(a, b, 20001)
        expected = np.sqrt((b**2 - a**2) / 2.0 / np.log(b / a))
        self.assertAlmostEqual(m2f.pivot_wavelength(w, np.ones_like(w)), expected, places=3)

    def test_pivot_independent_of_normalization_and_order(self):
        w, t = _tophat(4000.0, 5500.0)
        p = m2f.pivot_wavelength(w, t)
        self.assertAlmostEqual(m2f.pivot_wavelength(w[::-1], 100.0 * t[::-1]), p, places=8)

    def test_mag_flux_roundtrip(self):
        mags = np.array([15.0, 18.3, 22.7])
        f = m2f.abmag_to_flambda(mags, 6200.0)
        np.testing.assert_allclose(m2f.flambda_to_abmag(f, 6200.0), mags, atol=1e-12)

    def test_zero_mag_flat_fnu(self):
        # m_AB = 0 <=> f_nu = 3631 Jy; at lam_piv f_lam = f_nu c / lam^2
        self.assertAlmostEqual(
            m2f.abmag_to_flambda(0.0, 5000.0) / (AB_FNU * m2f.C_AA_PER_S / 5000.0**2), 1.0, places=3
        )

    def test_error_propagation(self):
        f = m2f.abmag_to_flambda(20.0, 5000.0)
        self.assertAlmostEqual(m2f.abmag_err_to_flambda_err(f, 0.1) / f, 0.4 * np.log(10) * 0.1)


class TestConsistencyWithPipeline(unittest.TestCase):
    """Pivot flux of a true-AB synthetic mag == photon-weighted band mean used by mangling."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.filt_w, self.filt_t = _tophat(4000.0, 7000.0)  # wide band: SED slope matters
        # smooth, non-trivial bandpass shape
        self.filt_t = self.filt_t * np.exp(-0.5 * ((self.filt_w - 5200.0) / 900.0) ** 2)
        # band_flux_trapz takes the Inputs/Filters root and looks in GeneralFilters/
        self.filt_path = self.tmp.name
        os.makedirs(os.path.join(self.tmp.name, "GeneralFilters"))
        np.savetxt(os.path.join(self.tmp.name, "GeneralFilters", "Test_band.dat"),
                   np.column_stack([self.filt_w, self.filt_t]))
        self.lam_piv = m2f.pivot_wavelength(self.filt_w, self.filt_t)

    def tearDown(self):
        self.tmp.cleanup()

    def _check(self, wls, flux):
        mag, _, cov = synth_ab_mag(wls, flux, 0 * flux, self.filt_w, self.filt_t, method="true_ab")
        self.assertGreater(cov, 0.999)
        _, _, band_mean, _, _, _ = band_flux_trapz(
            wls, flux, 0 * flux, "Test_band", filter_path=self.filt_path
        )
        self.assertAlmostEqual(m2f.abmag_to_flambda(mag, self.lam_piv) / band_mean, 1.0, places=4)

    def test_blue_power_law(self):
        w = np.linspace(3000.0, 9000.0, 60001)
        self._check(w, 1e-16 * (w / 5000.0) ** -4)

    def test_red_blackbody_like(self):
        w = np.linspace(3000.0, 9000.0, 60001)
        self._check(w, 1e-17 / (w**5 * (np.exp(1.4388e8 / (w * 2500.0)) - 1.0)) * 1e20)

    def test_median_wavelength_is_wrong(self):
        """Regression for the old bug: a blue-shifted 'effective' wavelength over-predicts flux."""
        w = np.linspace(3000.0, 9000.0, 60001)
        f = _ab_flat(w, 19.0)
        _, _, band_mean, _, _, _ = band_flux_trapz(w, f, 0 * f, "Test_band", filter_path=self.filt_path)
        too_blue = m2f.abmag_to_flambda(19.0, 0.87 * self.lam_piv)
        self.assertGreater(too_blue / band_mean, 1.2)
        self.assertAlmostEqual(m2f.abmag_to_flambda(19.0, self.lam_piv) / band_mean, 1.0, places=4)


class TestConvertRawPhotometry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.filters_root = os.path.join(self.tmp.name, "Filters")
        os.makedirs(os.path.join(self.filters_root, "GeneralFilters"))
        for name, (lo, hi) in {"X_g": (4000.0, 5500.0), "X_i": (6900.0, 8500.0)}.items():
            np.savetxt(os.path.join(self.filters_root, "GeneralFilters", name + ".dat"),
                       np.column_stack(_tophat(lo, hi)))
        self.csv = (
            "MJD,Mag,Mag_err,Flux,Flux_err,band,Instr,FilterSet\n"
            "57982.981,17.48,0.02,9.9e-16,1e-17,X_g,Cam,GeneralFilters\n"
            "57983.5,,0.1,1.234e-17,2e-18,X_i,Cam,GeneralFilters\n"
            "57984.0,19.1,0.05,1e-17,1e-18,X_i,Cam,GeneralFilters\n"
        )
        self.df = pd.read_csv(io.StringIO(self.csv), dtype=str, keep_default_na=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_convert_values_and_untouched_columns(self):
        new, report = m2f.convert_raw_photometry(self.df, self.filters_root)
        piv_g = m2f.pivot_wavelength(*_tophat(4000.0, 5500.0))
        self.assertAlmostEqual(float(new.loc[0, "Flux"]) / m2f.abmag_to_flambda(17.48, piv_g), 1.0, places=10)
        self.assertAlmostEqual(float(new.loc[0, "Flux_err"]) / float(new.loc[0, "Flux"]),
                               0.4 * np.log(10) * 0.02, places=10)
        # missing Mag -> Flux left as-is
        self.assertEqual(new.loc[1, "Flux"], "1.234e-17")
        self.assertEqual(new.loc[1, "Flux_err"], "2e-18")
        for col in ("MJD", "Mag", "Mag_err", "band", "Instr", "FilterSet"):
            self.assertTrue((new[col] == self.df[col]).all(), col)
        self.assertEqual({r.band for r in report}, {"X_g", "X_i"})

    def test_check_detects_stale_then_passes(self):
        self.assertEqual(len(m2f.check_raw_photometry(self.df, self.filters_root)), 2)
        new, _ = m2f.convert_raw_photometry(self.df, self.filters_root)
        self.assertEqual(m2f.check_raw_photometry(new, self.filters_root), [])

    def test_non_ab_system_rejected(self):
        df = self.df.assign(MagSys=["AB", "Vega", "AB"])
        with self.assertRaises(ValueError):
            m2f.convert_raw_photometry(df, self.filters_root)

    def test_missing_filter_raises(self):
        df = self.df.copy()
        df.loc[0, "band"] = "Nope_g"
        with self.assertRaises(FileNotFoundError):
            m2f.convert_raw_photometry(df, self.filters_root)


class TestShippedPhotometry(unittest.TestCase):
    """The committed raw photometry must already match the pivot-wavelength AB conversion."""

    def test_step0_check_passes_on_repo_inputs(self):
        raw_dir = os.path.join(REPO_ROOT, "Inputs", "Photometry", "1_LCs_flux_raw")
        sns = sorted(f[:-4] for f in os.listdir(raw_dir) if f.endswith(".dat"))
        self.assertTrue(sns)
        for sn in sns:
            r = subprocess.run(
                [sys.executable, os.path.join(CODES, "0_Mag_to_Flux_KN.py"),
                 "--snname", sn, "--coco-path", REPO_ROOT, "--check"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, msg="%s:\n%s" % (sn, r.stdout[-2000:] + r.stderr[-2000:]))


if __name__ == "__main__":
    unittest.main()
