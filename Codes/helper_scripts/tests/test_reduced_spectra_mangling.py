"""Steps 5-6 with reduced (unsmoothed, weighted) spectra.

Covers the synthetic-photometry error, the clipped-fraction rule, binning
mangled spectra onto the GP grid before the log, and step 6's grid prep
copying binned nodes instead of interpolating.
"""
import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np

import mangle_spectra_log as msl  # noqa: E402
import spec_binning as sb  # noqa: E402
from twodim_grid_prep import FullMangledSeries_Class  # noqa: E402


def _write_filter(d, name="Test_band", lo=5600.0, hi=6400.0):
    sub = os.path.join(d, "GeneralFilters")
    os.makedirs(sub, exist_ok=True)
    w = np.linspace(lo, hi, 400)
    t = np.exp(-0.5 * ((w - 6000.0) / 150.0) ** 2)
    np.savetxt(os.path.join(sub, name + ".dat"), np.column_stack([w, t]))
    return d + os.sep


class TestSynphotError(unittest.TestCase):
    def test_error_matches_scatter_for_any_pixel_size(self):
        with tempfile.TemporaryDirectory() as d:
            fp = _write_filter(d)
            for dl in (0.2, 2.0, 30.0):
                w = np.arange(5000.0, 7000.0 + dl, dl)
                sig = np.full(len(w), 0.1)
                vals = []
                for seed in range(400):
                    f = 1.0 + np.random.default_rng(seed).normal(0, 0.1, len(w))
                    vals.append(msl.band_flux_trapz(w, f, sig, "Test_band", filter_path=fp)[2])
                err = msl.band_flux_trapz(w, np.ones(len(w)), sig, "Test_band", filter_path=fp)[3]
                self.assertAlmostEqual(err / np.std(vals), 1.0, delta=0.1, msg="pixel %g A" % dl)

    def test_flux_unchanged_from_trapezoid(self):
        from scipy import integrate

        with tempfile.TemporaryDirectory() as d:
            fp = _write_filter(d)
            w = np.arange(5000.0, 7000.0, 1.3)
            f = 1.0 + 0.3 * np.sin(w / 200.0)
            got = msl.band_flux_trapz(w, f, 0.1 * f, "Test_band", filter_path=fp)[2]
            filt = np.loadtxt(os.path.join(fp, "GeneralFilters", "Test_band.dat"))
            cut = (w > filt[0, 0]) & (w < filt[-1, 0])
            t = np.interp(w[cut], filt[:, 0], filt[:, 1])
            ref = integrate.trapezoid(t * w[cut] * f[cut], w[cut]) / integrate.trapezoid(t * w[cut], w[cut])
            self.assertAlmostEqual(got, ref, places=12)


class TestClippedFraction(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.fp = _write_filter(self.td.name)
        self.w = np.arange(5000.0, 7000.0, 1.0)
        self.f = np.ones(len(self.w))
        self.e = np.full(len(self.w), 0.1)

    def tearDown(self):
        self.td.cleanup()

    def test_mostly_clipped_band_is_skipped(self):
        wt = np.where((self.w > 5700) & (self.w < 6300), 0.0, 1.0)
        syn = msl.band_flux_trapz(self.w, self.f, self.e, "Test_band",
                                  filter_path=self.fp, spec_weight=wt)[2]
        self.assertTrue(np.isnan(syn))

    def test_partly_clipped_band_error_inflated(self):
        wt = np.where((self.w > 6150) & (self.w < 6400), 0.0, 1.0)
        base = msl.band_flux_trapz(self.w, self.f, self.e, "Test_band", filter_path=self.fp)
        got = msl.band_flux_trapz(self.w, self.f, self.e, "Test_band",
                                  filter_path=self.fp, spec_weight=wt)
        self.assertTrue(np.isfinite(got[2]))
        self.assertGreater(got[3], base[3] * 1.05)

    def test_weight_column_threads_through_constraints(self):
        # a filter that is mostly clipped gives no mangling constraint
        import pandas as pd

        row = pd.Series({"spec_mjd": 1.0, "Test_band_fit_log_flux": 0.0,
                         "Test_band_fit_log_fluxerr": 0.01, "Test_band_inrange": True})
        mjd = {"Test_band": {"min_mjd": 0.0, "max_mjd": 2.0}}
        wt = np.where((self.w > 5700) & (self.w < 6300), 0.0, 1.0)
        args = (self.w, self.f, self.e, row, ["Test_band"], mjd)
        self.assertIsNotNone(msl._collect_mangle_constraints(*args, filter_path=self.fp))
        self.assertIsNone(msl._collect_mangle_constraints(*args, filter_path=self.fp, weight=wt))


class TestBinMangledToGrid(unittest.TestCase):
    def test_noisy_pixels_binned_before_log_unbiased(self):
        # per-pixel S/N 1 (many negative pixels); log of the bin, not of pixels
        rng = np.random.default_rng(3)
        w = np.arange(5000.0, 6000.0, 0.2)
        f = 1.0 + rng.normal(0, 1.0, len(w))
        grid = sb.gp_log_grid(5000.0, 6000.0, 30.0)
        wl, lf, le, m = msl.bin_mangled_to_gp_grid(
            w, f, np.ones(len(w)), np.ones(len(w)), np.zeros(len(w)), np.zeros(len(w)), grid)
        self.assertGreater(len(wl), 25)
        self.assertAlmostEqual(np.average(lf, weights=1 / le ** 2), 0.0, delta=0.01)
        # log error ~ 1 / (S/N per bin * ln 10)
        self.assertTrue(np.all(le < 0.1))

    def test_mask_applied_in_linear_flux_and_clipped_bins_dropped(self):
        w = np.arange(5000.0, 6000.0, 1.0)
        f = np.full(len(w), 2.0)
        wt = np.where((w > 5400) & (w < 5600), 0.0, 1.0)
        grid = sb.gp_log_grid(5000.0, 6000.0, 30.0)
        mask = np.full(len(w), 0.3)
        wl, lf, le, m = msl.bin_mangled_to_gp_grid(
            w, f, np.full(len(w), 0.01), wt, mask, np.zeros(len(w)), grid)
        np.testing.assert_allclose(lf, np.log10(2.0) + 0.3, atol=1e-10)
        np.testing.assert_allclose(m, 0.3)
        self.assertFalse(np.any((wl > 5410) & (wl < 5590)))


class TestGridPrepBinned(unittest.TestCase):
    def test_binned_files_copied_onto_nodes_with_gaps_left_empty(self):
        grid = sb.gp_log_grid(5000.0, 8000.0, 30.0)
        with tempfile.TemporaryDirectory() as td:
            vals = {}
            for mjd, (i0, i1) in {"57983.969000": (2, 40), "57984.969003": (10, 60)}.items():
                nodes = grid[i0:i1]
                keep = np.ones(len(nodes), bool)
                keep[5:9] = False                       # a clipped gap
                lf = -16.0 + 0.001 * np.arange(len(nodes))
                msl.save_mangled_spectrum(
                    os.path.join(td, mjd + "_mangled_spec.txt"),
                    10 ** nodes[keep], lf[keep], np.full(keep.sum(), 0.01),
                    np.zeros(keep.sum()), gp_grid_binned=True)
                vals[mjd] = (nodes, keep, lf)
            sc = FullMangledSeries_Class("TESTSN", 57982.52851852, mode="extrapolate_spectra",
                                         output_dir=td + os.sep, mangled_spectra_dir=td,
                                         prepare_output_dir=False)
            sc.plot_grid_rebin = False
            gw, gm, ga, ge = sc.grid_all_spectraltimeseries()
            for mjd, (nodes, keep, lf) in vals.items():
                # columns are keyed by log10 phase, in file-listing order
                j = int(np.argmin(np.abs(np.asarray(gm) - np.log10(float(mjd) - 57982.52851852))))
                col = ga.iloc[:, j]
                for n, k, v in zip(nodes, keep, lf):
                    i = int(np.argmin(np.abs(gw - n)))
                    self.assertAlmostEqual(gw[i], n, places=6)
                    if k:
                        self.assertAlmostEqual(col.iloc[i], v, places=9)
                    else:
                        self.assertTrue(np.isnan(col.iloc[i]))   # not interpolated
            # one GP point per binned row, nothing extra
            self.assertEqual(int(np.isfinite(ga.values).sum()),
                             sum(int(k.sum()) for _, k, _ in vals.values()))


if __name__ == "__main__":
    unittest.main()
