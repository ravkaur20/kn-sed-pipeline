"""Tests for spec_binning and the weight-aware parts of step 4 prescale."""
import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np

import spec_binning as sb  # noqa: E402
import spectra_pre_scale as sps  # noqa: E402


def _spec(wl, flux, err, weight=None):
    out = np.zeros(len(wl), dtype=sps.SPEC_W_DTYPE)
    out["wls"], out["flux"], out["fluxerr"] = wl, flux, err
    out["weight"] = 1.0 if weight is None else weight
    return out


class TestIvarBin(unittest.TestCase):
    def test_mean_and_error(self):
        wl = np.arange(0.5, 100.0, 1.0)              # 100 pixels, 10 per bin
        flux = np.full(100, 2.0)
        err = np.full(100, 0.5)
        r = sb.ivar_bin(wl, flux, err, None, np.arange(0, 101, 10))
        np.testing.assert_allclose(r["flux"], 2.0)
        np.testing.assert_allclose(r["err"], 0.5 / np.sqrt(10))

    def test_weight_zero_excluded_and_mostly_clipped_bin_empty(self):
        wl = np.arange(0.5, 20.0, 1.0)
        flux = np.ones(20)
        flux[:3] = 1e6                                # clipped junk
        w = np.ones(20)
        w[:3] = 0
        w[10:17] = 0                                  # 70% of bin 2 clipped
        r = sb.ivar_bin(wl, flux, np.ones(20), w, [0, 10, 20])
        self.assertAlmostEqual(r["flux"][0], 1.0)     # junk never used
        self.assertTrue(np.isnan(r["flux"][1]))       # mostly clipped -> empty
        self.assertAlmostEqual(r["good_frac"][1], 0.3)

    def test_negative_pixels_kept_unbiased(self):
        rng = np.random.default_rng(0)
        wl = np.arange(0.0, 20000.0, 1.0)
        flux = 1.0 + rng.normal(0, 1.0, len(wl))      # S/N 1 per pixel, ~16% negative
        r = sb.ivar_bin(wl, flux, np.ones(len(wl)), None, [0, 20000])
        self.assertAlmostEqual(r["flux"][0], 1.0, delta=0.03)

    def test_gp_log_grid_matches_grid_prep_formula(self):
        g = sb.gp_log_grid(3500.0, 24000.0, 30.0)
        lo, hi = np.log10(3500.0), np.log10(24000.0)
        step = np.log10(1.0 + 30.0 / 10 ** (0.5 * (lo + hi)))
        np.testing.assert_allclose(np.diff(g), step, atol=2e-8)
        self.assertAlmostEqual(g[0], round(lo, 8))

    def test_bin_onto_log_grid(self):
        grid = sb.gp_log_grid(5000.0, 6000.0, 30.0)
        wl = np.arange(5000.0, 6000.0, 0.2)
        r = sb.bin_onto_log_grid(wl, np.ones(len(wl)), np.ones(len(wl)), None, grid)
        self.assertEqual(len(r["flux"]), len(grid))
        self.assertTrue(np.nanmax(np.abs(r["flux"] - 1)) < 1e-12)


class TestPrescaleWeights(unittest.TestCase):
    def test_overlap_fit_not_biased_by_noisy_pixels(self):
        # true scale 2; per-pixel S/N ~2 in both arms. A pixel-by-pixel fit
        # comes out ~17% low here; the binned fit should average to 2.
        wl = np.arange(5600.0, 5900.0, 0.2)
        ms = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            ref = _spec(wl, 2.0 + rng.normal(0, 1.0, len(wl)), np.ones(len(wl)))
            arm = _spec(wl, 1.0 + rng.normal(0, 0.5, len(wl)), np.full(len(wl), 0.5))
            m, n = sps.overlap_scale_factor_wls(ref, arm)
            self.assertGreater(n, 0)
            ms.append(m)
        self.assertAlmostEqual(np.mean(ms), 2.0, delta=0.02)

    def test_overlap_fit_ignores_weight_zero(self):
        wl = np.arange(5600.0, 5900.0, 1.0)
        ref = _spec(wl, np.full(len(wl), 2.0), np.ones(len(wl)))
        f = np.ones(len(wl))
        w = np.ones(len(wl))
        f[100:200] = 50.0                             # clipped junk in the arm
        w[100:200] = 0
        arm = _spec(wl, f, np.ones(len(wl)), w)
        m, _ = sps.overlap_scale_factor_wls(ref, arm)
        self.assertAlmostEqual(m, 2.0, places=6)

    def test_load_save_round_trip_keeps_weight_and_negatives(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.dat")
            with open(p, "w") as fh:
                fh.write("# wls\tflux\tflux_err\tweight\n")
                fh.write("5000.0\t-1.0e-17\t2.0e-17\t1\n5000.2\t3.0e-17\t2.0e-17\t0\n")
            s = sps.load_spectrum_array(p)
            self.assertEqual(len(s), 2)
            self.assertEqual(list(s["weight"]), [1.0, 0.0])
            self.assertLess(s["flux"][0], 0)
            q = os.path.join(d, "out.dat")
            sps.save_spectrum_array(q, sps.apply_flux_scale(s, 2.0))
            t = sps.load_spectrum_array(q)
            self.assertEqual(list(t["weight"]), [1.0, 0.0])
            self.assertAlmostEqual(t["flux"][1], 6.0e-17)

    def test_legacy_three_column_file_gets_weight_one(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.dat")
            with open(p, "w") as fh:
                fh.write("#wls\tflux\tflux_err\n5000.0\t1.0\t0.1\n5001.0\t-1.0\t0.1\n")
            s = sps.load_spectrum_array(p)
            self.assertEqual(len(s), 1)               # legacy: flux <= 0 dropped
            self.assertEqual(s["weight"][0], 1.0)

    def test_median_snr_per_angstrom(self):
        # same per-Angstrom S/N, different pixel sizes -> same value
        a = _spec(np.arange(0, 100, 0.2), np.ones(500), np.full(500, 0.1))
        b = _spec(np.arange(0, 100, 2.0), np.ones(50), np.full(50, 0.1 / np.sqrt(10)))
        self.assertAlmostEqual(sps.median_snr(a), sps.median_snr(b), places=6)


if __name__ == "__main__":
    unittest.main()
