"""Step 5's minimum filter coverage (MANGLE_MIN_FILTER_COVERAGE).

A filter is used for mangling only if the spectrum covers enough of its
photon weight, since the synthetic flux of a partly covered filter is the mean
over the covered part but is compared with full-band photometry.
"""
import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import mangle_spectra_log as msl  # noqa: E402
import pipeline_config as pconf  # noqa: E402


def _write_box(d, name, lo, hi):
    sub = os.path.join(d, "GeneralFilters")
    os.makedirs(sub, exist_ok=True)
    w = np.linspace(lo, hi, 801)
    np.savetxt(os.path.join(sub, name + ".dat"), np.column_stack([w, np.ones_like(w)]))


def _spec(w, f, weight=None):
    names = [("wls", float), ("flux", float), ("fluxerr", float)]
    if weight is not None:
        names.append(("weight", float))
    s = np.zeros(len(w), dtype=names)
    s["wls"], s["flux"], s["fluxerr"] = w, f, 0.01 * f
    if weight is not None:
        s["weight"] = weight
    return s


def _full_band_logflux(lo, hi, beta):
    """Photon-weighted mean of F = (lambda/6000)**beta over a flat filter."""
    w = np.linspace(lo, hi, 20001)
    f = (w / 6000.0) ** beta
    return float(np.log10(np.trapz(w * f, w) / np.trapz(w, w)))


class _Base(unittest.TestCase):
    """Two flat filters, A (5000-6000) and B (6500-7500); photometry is the
    exact full-band flux of a power-law spectrum, so a correct mask is 0."""

    beta = -3.0

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        _write_box(self.td.name, "Box_A", 5000.0, 6000.0)
        _write_box(self.td.name, "Box_B", 6500.0, 7500.0)
        self.fp = self.td.name + os.sep
        self.row = pd.Series({
            "spec_mjd": 1.0,
            "Box_A_fit_log_flux": _full_band_logflux(5000, 6000, self.beta),
            "Box_A_fit_log_fluxerr": 0.01, "Box_A_inrange": True,
            "Box_B_fit_log_flux": _full_band_logflux(6500, 7500, self.beta),
            "Box_B_fit_log_fluxerr": 0.01, "Box_B_inrange": True,
        })
        self.mjd = {"Box_A": {"min_mjd": 0.0, "max_mjd": 2.0},
                    "Box_B": {"min_mjd": 0.0, "max_mjd": 2.0}}
        self.filters = ["Box_A", "Box_B"]

    def tearDown(self):
        self.td.cleanup()

    def power_law(self, lo, hi, step=1.0):
        w = np.arange(lo, hi + step / 2, step)
        return w, (w / 6000.0) ** self.beta


class TestCoverageFraction(_Base):
    def test_full_partial_and_union(self):
        cov = lambda ivs: msl.filter_coverage_fraction("Box_A", ivs, filter_path=self.fp)
        self.assertAlmostEqual(cov([(4000.0, 8000.0)]), 1.0, places=10)
        # photon weight T * lambda: the red half holds more than half
        expect = (5750.0 ** 2 - 5500.0 ** 2 + 6000.0 ** 2 - 5750.0 ** 2) / (6000.0 ** 2 - 5000.0 ** 2)
        self.assertAlmostEqual(cov([(5500.0, 9000.0)]), expect, places=6)
        self.assertAlmostEqual(cov([(5000.0, 5500.0), (5500.0, 6000.0)]), 1.0, places=10)
        self.assertAlmostEqual(cov([(5000.0, 5400.0), (5600.0, 6000.0)]),
                               1.0 - (5600.0 ** 2 - 5400.0 ** 2) / (6000.0 ** 2 - 5000.0 ** 2), places=6)
        self.assertEqual(cov([(7000.0, 8000.0)]), 0.0)

    def test_data_interval_ignores_clipped_ends_not_interior(self):
        w = np.arange(4000.0, 8000.0)
        wt = np.ones_like(w)
        wt[w < 4500] = 0.0          # clipped block at the blue end
        wt[(w > 6000) & (w < 6200)] = 0.0  # interior telluric band
        self.assertEqual(msl.spectrum_data_interval(w, wt), (4500.0, 7999.0))
        self.assertEqual(msl.spectrum_data_interval(w), (4000.0, 7999.0))


class TestSingleSpectrum(_Base):
    def test_partly_covered_filter_is_skipped(self):
        w, f = self.power_law(4500.0, 7000.0)   # covers A fully, B about 50%
        spec = _spec(w, f)
        # without the rule B is used, with a biased offset
        old = msl.compute_mangling_mask(spec, self.row, self.filters, self.mjd,
                                        filter_path=self.fp, min_filter_coverage=0.0)
        self.assertEqual(old[3]["used_filters"], ["Box_A", "Box_B"])
        c = msl._collect_mangle_constraints(w, f, 0.01 * f, self.row, self.filters, self.mjd,
                                            filter_path=self.fp, min_filter_coverage=0.0)
        self.assertGreater(abs(c[0][1]), 0.02)   # B's log_diff is off by > 0.02 dex
        self.assertLess(abs(c[0][0]), 1e-3)      # A's is right
        new = msl.compute_mangling_mask(spec, self.row, self.filters, self.mjd,
                                        filter_path=self.fp, min_filter_coverage=0.99)
        meta = new[3]
        self.assertEqual(meta["used_filters"], ["Box_A"])
        self.assertIn("Box_B", meta["skipped_low_coverage"])
        self.assertAlmostEqual(meta["filter_coverage"]["Box_A"], 1.0)

    def test_one_filter_gives_constant_mask_equal_to_its_offset(self):
        w, f = self.power_law(4500.0, 7000.0)
        f = f * 10 ** -0.2                       # spectrum 0.2 dex too faint
        mask, log_f, _, meta = msl.compute_mangling_mask(
            _spec(w, f), self.row, self.filters, self.mjd,
            filter_path=self.fp, min_filter_coverage=0.99)
        self.assertTrue(meta["constant_mask"])
        np.testing.assert_allclose(mask, 0.2, atol=1e-4)
        np.testing.assert_allclose(log_f, np.log10(f) + 0.2, atol=1e-4)
        self.assertEqual(np.ptp(mask), 0.0)

    def test_two_filters_still_fit_a_curve(self):
        w, f = self.power_law(4500.0, 8000.0)
        out = msl.compute_mangling_mask(_spec(w, f), self.row, self.filters, self.mjd,
                                        filter_path=self.fp, min_filter_coverage=0.99)
        self.assertEqual(out[3]["used_filters"], ["Box_A", "Box_B"])
        self.assertFalse(out[3]["constant_mask"])
        self.assertTrue(np.all(np.isfinite(out[0])))
        self.assertLess(np.max(np.abs(out[0])), 0.01)

    def test_clipped_interior_counts_as_covered_clipped_end_does_not(self):
        w, f = self.power_law(4500.0, 8000.0)
        wt = np.ones_like(w)
        wt[(w > 5400) & (w < 5450)] = 0.0       # interior clip inside A
        out = msl.compute_mangling_mask(_spec(w, f, wt), self.row, self.filters, self.mjd,
                                        filter_path=self.fp, min_filter_coverage=0.99)
        self.assertIn("Box_A", out[3]["used_filters"])
        wt2 = np.ones_like(w)
        wt2[w < 5100] = 0.0                      # clipped blue end eats 9% of A
        out2 = msl.compute_mangling_mask(_spec(w, f, wt2), self.row, self.filters, self.mjd,
                                         filter_path=self.fp, min_filter_coverage=0.99)
        self.assertNotIn("Box_A", out2[3]["used_filters"])

    def test_default_threshold_comes_from_config(self):
        self.assertEqual(pconf.MANGLE_MIN_FILTER_COVERAGE, 0.99)
        w, f = self.power_law(4500.0, 7000.0)
        out = msl.compute_mangling_mask(_spec(w, f), self.row, self.filters, self.mjd,
                                        filter_path=self.fp)
        self.assertEqual(out[3]["used_filters"], ["Box_A"])
        self.assertEqual(out[3]["min_filter_coverage"], 0.99)

    def test_no_covered_filter_returns_none(self):
        w, f = self.power_law(5500.0, 7000.0)
        self.assertIsNone(msl.compute_mangling_mask(_spec(w, f), self.row, self.filters, self.mjd,
                                                    filter_path=self.fp, min_filter_coverage=0.99))


class TestBundle(_Base):
    def setUp(self):
        super().setUp()
        self.phot = pd.DataFrame([self.row])

    def bundle(self, arms, cov):
        specs = {name: _spec(*self.power_law(lo, hi)) for name, (lo, hi) in arms.items()}
        return msl.compute_mangling_mask_bundle(
            specs, self.phot, self.filters, self.mjd, filter_path=self.fp,
            member_mjd={n: 1.0 for n in specs}, min_filter_coverage=cov)

    def test_filter_straddling_overlapping_arms_is_used_and_stitched(self):
        out = self.bundle({"UVB_x": (4500.0, 5520.0), "VIS_x": (5480.0, 8000.0)}, 0.99)
        meta = out[2]
        self.assertEqual(meta["used_filters"], ["Box_A", "Box_B"])
        self.assertEqual(meta["mode_for_filter"]["Box_A"], "stitched")
        self.assertAlmostEqual(meta["filter_coverage"]["Box_A"], 1.0)

    def test_filter_over_an_arm_gap_is_skipped(self):
        arms = {"UVB_x": (4500.0, 5400.0), "VIS_x": (5600.0, 8000.0)}
        self.assertEqual(self.bundle(arms, 0.0)[2]["used_filters"], ["Box_A", "Box_B"])
        meta = self.bundle(arms, 0.99)[2]
        self.assertEqual(meta["used_filters"], ["Box_B"])
        self.assertTrue(meta["constant_mask"])

    def test_single_arm_partial_filter_is_skipped(self):
        meta = self.bundle({"UVB_x": (4500.0, 5400.0), "VIS_x": (6000.0, 7000.0)}, 0.99)
        self.assertIsNone(meta)    # A: 4500-5400 only; B: 6000-7000 only


if __name__ == "__main__":
    unittest.main()
