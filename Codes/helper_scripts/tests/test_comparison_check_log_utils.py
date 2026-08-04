"""Tests for comparison_check_log_utils (7.5 log comparison notebook helpers)."""
import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import REPO_ROOT as ROOT
import os
import sys
import tempfile
import unittest

import numpy as np


import comparison_check_log_utils as cc  # noqa: E402


def _final_as_observed_dir_with_spectra(coco: str, sn: str) -> str | None:
    """Resolve a directory containing FINAL ``as_observed`` spectra."""
    d = cc.resolve_final_directory_legacy(coco, sn, "as_observed")
    return d


class TestDeduplicateWavelength(unittest.TestCase):
    def test_merges_exact_duplicate_lambda(self):
        w = np.array([4000.0, 4000.0, 4100.0])
        f = np.array([1.0, 3.0, 2.0])
        wu, fu, eu = cc.deduplicate_wavelength_flux(w, f, None)
        np.testing.assert_allclose(wu, [4000.0, 4100.0])
        np.testing.assert_allclose(fu, [2.0, 2.0])
        np.testing.assert_allclose(eu, [0.0, 0.0])


class TestParseAndStem(unittest.TestCase):
    def test_parse_final_stem_signed(self):
        self.assertAlmostEqual(
            cc.parse_final_stem("0.842480_FINAL_spec_FL.txt"), 0.842480
        )
        self.assertAlmostEqual(
            cc.parse_final_stem("-0.078320_FINAL_spec.txt"), -0.078320
        )

    def test_calendar_mjd_stem_roundtrip(self):
        self.assertTrue(cc.stem_looks_like_calendar_mjd(57982.52851852))
        self.assertFalse(cc.stem_looks_like_calendar_mjd(0.842480))

    def test_stem_to_spec_mjd_at2017gfo(self):
        coco = ROOT
        sn = "AT2017gfo"
        mjd = cc.stem_to_spec_mjd(0.842480, coco, sn)
        self.assertGreater(mjd, 57982.0)
        self.assertLess(mjd, 58000.0)
        np.testing.assert_allclose(mjd, 57989.46904913394, rtol=0.0, atol=1e-4)


class TestSpectraListAugment(unittest.TestCase):
    def test_prepend_only_if_later(self):
        wl = np.linspace(3000, 8000, 50)
        t0 = 57982.0
        sl = [(t0 + 1.0, wl.copy(), np.ones_like(wl))]
        out = cc.augment_spectra_list_explosion_mjd(sl, t0, wl, flux_floor_linear=1e-50)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][0], t0)

    def test_dense_axis_monotonic(self):
        mjd0 = 58000.0
        st = np.array([58000.1, 58001.0, 58010.0])
        mag = np.array([20.0, 19.0, 18.0])
        u, v = cc.dense_plot_axis_log_days(st, mag, mjd0, n_points=32)
        self.assertEqual(u.shape, v.shape)
        self.assertGreater(u.size, 2)

    def test_lookup_table_prepend_explosion_adds_row(self):
        coco = ROOT
        sn = "AT2017gfo"
        tdir = tempfile.mkdtemp(prefix="cclog_prepend_")
        try:
            for stem, scale in [(0.842480, 1e-17), (-0.078320, 2e-17)]:
                path = os.path.join(tdir, "%s_FINAL_spec_FL.txt" % stem)
                wl = np.linspace(4000.0, 8000.0, 20)
                fl = np.full_like(wl, scale)
                fe = np.full_like(wl, scale * 0.1)
                np.savetxt(path, np.column_stack([wl, fl, fe]))
            lt0, mj0, _w, sl0 = cc.create_lookup_table(
                tdir,
                coco,
                sn,
                flux_on_disk="linear",
                wavelength_bins=40,
            )
            t_early = float(np.min(mj0)) - 5.0
            lt1, mj1, _w2, sl1 = cc.create_lookup_table(
                tdir,
                coco,
                sn,
                flux_on_disk="linear",
                wavelength_bins=40,
                prepend_explosion_mjd=t_early,
                prepend_flux_floor_linear=1e-50,
            )
            self.assertEqual(lt1.shape[0], lt0.shape[0] + 1)
            self.assertAlmostEqual(mj1[0], t_early)
            self.assertEqual(len(sl1), len(sl0) + 1)
        finally:
            for fn in os.listdir(tdir):
                os.unlink(os.path.join(tdir, fn))
            os.rmdir(tdir)


class TestCreateLookupSmoke(unittest.TestCase):
    def test_real_as_observed_subset(self):
        coco = ROOT
        sn = "AT2017gfo"
        final_dir = _final_as_observed_dir_with_spectra(coco, sn)
        if final_dir is None:
            self.skipTest("no FINAL as_observed .txt (legacy or twodim layout)")

        lt, spec_mjds, wls, slist = cc.create_lookup_table(
            final_dir,
            coco,
            sn,
            flux_on_disk="auto",
            wavelength_bins=200,
        )
        self.assertTrue(cc.lookup_index_is_mjd(lt))
        self.assertEqual(len(spec_mjds), lt.shape[0])
        self.assertGreater(spec_mjds.max(), 40000.0)
        self.assertEqual(len(slist), len(spec_mjds))
        self.assertEqual(slist[0][0], spec_mjds[0])
        self.assertIn("spec_mjd", lt.attrs)

    def test_synthetic_temp_linear_files(self):
        coco = ROOT
        sn = "AT2017gfo"
        tdir = tempfile.mkdtemp(prefix="cclog_")
        try:
            # Two small linear-F_lambda files with phase-like stems (must exist in logspace table)
            for stem, scale in [(0.842480, 1e-17), (-0.078320, 2e-17)]:
                path = os.path.join(tdir, "%s_FINAL_spec_FL.txt" % stem)
                wl = np.linspace(4000.0, 8000.0, 20)
                fl = np.full_like(wl, scale)
                fe = np.full_like(wl, scale * 0.1)
                np.savetxt(path, np.column_stack([wl, fl, fe]))
            lt, spec_mjds, _w, slist = cc.create_lookup_table(
                tdir,
                coco,
                sn,
                flux_on_disk="linear",
                wavelength_bins=50,
            )
            self.assertEqual(lt.shape[0], 2)
            self.assertEqual(len(spec_mjds), 2)
            self.assertTrue(np.all(np.diff(spec_mjds) >= 0))
            self.assertEqual(len(slist), 2)
        finally:
            for fn in os.listdir(tdir):
                os.unlink(os.path.join(tdir, fn))
            os.rmdir(tdir)

    def test_two_near_stems_keep_separate_rows(self):
        """Close filename stems can share one Log_Phase bin — still one row per file (no averaging)."""
        coco = ROOT
        sn = "AT2017gfo"
        tdir = tempfile.mkdtemp(prefix="cclog_twostems_")
        try:
            for stem in (0.842480, 0.842481):
                path = os.path.join(tdir, "%s_FINAL_spec_FL.txt" % stem)
                wl = np.array([4000.0, 5000.0])
                fl = np.array([1e-17, 2e-17])
                np.savetxt(path, np.column_stack([wl, fl]))
            lt, spec_mjds, _w, slist = cc.create_lookup_table(
                tdir,
                coco,
                sn,
                flux_on_disk="linear",
                wavelength_bins=10,
            )
            self.assertEqual(lt.shape[0], 2)
            self.assertEqual(len(spec_mjds), 2)
            self.assertEqual(len(slist), 2)
            self.assertIn("spec_mjd", lt.attrs)
            self.assertEqual(len(lt.attrs["spec_mjd"]), 2)
        finally:
            for fn in os.listdir(tdir):
                os.unlink(os.path.join(tdir, fn))
            os.rmdir(tdir)


if __name__ == "__main__":
    unittest.main()
