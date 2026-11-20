"""Step 6 "gp_correction" update rule (ITER_REMANGLE_MODE).

Each iteration: GP spectrum at the epoch (full grid) minus the previous
correction where the spectrum has data -> mangled to the photometry -> new
correction C over the full grid -> next GP input = step-5 mangled + C.
XSHOOTER arms of a group share one C.
"""
import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import iterative_gp_mangle as igm  # noqa: E402
import mangle_spectra_log as msl  # noqa: E402

BOXES = {"Box_A": (4600.0, 5600.0), "Box_B": (6300.0, 7300.0), "Box_C": (8000.0, 9000.0),
         "Box_D": (11000.0, 12500.0)}
BETA = -2.0


def truth(w):
    return 1e-16 * (np.asarray(w, float) / 6000.0) ** BETA


def band_logflux(lo, hi):
    w = np.linspace(lo, hi, 20001)
    return float(np.log10(np.trapz(w * truth(w), w) / np.trapz(w, w)))


class _Setup(unittest.TestCase):
    """A group (two arms, 4500-6400 and 6200-9500 A) and a single spectrum
    (4500-7400 A). Photometry is the truth; GP extraction on a 0.01-dex grid."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        d = self.td.name
        os.makedirs(os.path.join(d, "GeneralFilters"))
        for n, (lo, hi) in BOXES.items():
            w = np.linspace(lo, hi, 401)
            np.savetxt(os.path.join(d, "GeneralFilters", n + ".dat"), np.column_stack([w, np.ones_like(w)]))
        self.fp = d + os.sep
        self.step5 = os.path.join(d, "step5")
        self.out = os.path.join(d, "out")
        self.members = {
            "UVB_x.dat": (58000.000002, 4500.0, 6400.0),
            "VIS_x.dat": (58000.000003, 6200.0, 9500.0),
            "LDSS_y.dat": (58000.2, 4500.0, 7400.0),
        }
        self.entries, self.pre = {}, {}
        rows = []
        for bn, (mjd, lo, hi) in self.members.items():
            w = np.arange(lo, hi, 2.0)
            p = os.path.join(d, bn)
            np.savetxt(p, np.column_stack([w, truth(w), 0.01 * truth(w), np.ones_like(w)]),
                       header="wls flux fluxerr weight")
            self.pre[bn] = p
            self.entries[bn] = SimpleNamespace(basename=bn, mjd=mjd, path=p)
            nodes = 10 ** np.arange(np.log10(lo) + 0.001, np.log10(hi), 0.0015)
            msl.save_mangled_spectrum(os.path.join(self.step5, "%.6f_mangled_spec.txt" % mjd),
                                      nodes, np.log10(truth(nodes)) + 0.1, np.full(nodes.size, 0.01),
                                      np.full(nodes.size, 0.1), gp_grid_binned=True)
            row = {"spec_mjd": mjd}
            for n, (lo_, hi_) in BOXES.items():
                row.update({n + "_fit_log_flux": band_logflux(lo_, hi_), n + "_fit_log_fluxerr": 0.01,
                            n + "_inrange": True})
            rows.append(row)
        self.phot = pd.DataFrame(rows)
        self.mjd_dict = {n: {"min_mjd": 57999.0, "max_mjd": 58001.0} for n in BOXES}
        self.units = [("g1", [self.entries["UVB_x.dat"], self.entries["VIS_x.dat"]]),
                      ("LDSS_y.dat", [self.entries["LDSS_y.dat"]])]
        self.lw = np.arange(np.log10(1600.0), np.log10(25000.0), 0.01)

    def tearDown(self):
        self.td.cleanup()

    def extraction(self, offset_fn=lambda w: 0.0 * w):
        w = 10 ** self.lw
        ex = {"log10_wls": self.lw, "log10_flux": np.log10(truth(w)) + offset_fn(w),
              "log10_fluxerr": np.full(self.lw.size, 0.005), "target_wls_linear": w}
        return {bn: {"basename": bn, "extracted": ex} for bn in self.members}

    def run_rule(self, extractions, prev=None, out=None):
        return igm.gp_correction_remangle(
            "TEST", units=self.units, extractions_by_bn=extractions, prev_corrections=prev or {},
            step5_dir=self.step5, prescaled_paths=self.pre, mangled_out_dir=out or self.out,
            corrections_out_dir=os.path.join(out or self.out, "corr"), phot4mangling=self.phot,
            avail_filters=list(BOXES), filter_mjd_dict=self.mjd_dict, filter_path=self.fp,
            fine_step_log10=0.0015, min_filter_coverage=0.99)

    def mangled(self, bn, out=None):
        d = np.loadtxt(os.path.join(out or self.out, "%.6f_mangled_spec.txt" % self.members[bn][0]))
        return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


class TestFirstIteration(_Setup):
    def test_gp_too_faint_gives_full_range_correction_added_to_step5(self):
        res = self.run_rule(self.extraction(lambda w: -0.2 + 0.0 * w))   # GP 0.2 dex too faint
        wls, c, s = res["corrections"]["g1"]
        self.assertLess(wls.min(), 1700.0)
        self.assertGreater(wls.max(), 24000.0)                         # full SED grid
        np.testing.assert_allclose(np.interp([5000, 7000, 8500, 12000], wls, c), 0.2, atol=2e-3)
        self.assertEqual(sorted(res["report"]["units"][0]["used_filters"]), sorted(BOXES))
        for bn in ("UVB_x.dat", "VIS_x.dat"):
            w, lf, le, m = self.mangled(bn)
            np.testing.assert_allclose(lf, np.log10(truth(w)) + 0.1 + 0.2, atol=2e-3)
            np.testing.assert_allclose(m, 0.1 + 0.2, atol=2e-3)       # step-5 mask + C
            self.assertTrue(np.all(le > 0.01))                          # C's std added

    def test_xshooter_arms_share_one_correction(self):
        res = self.run_rule(self.extraction(lambda w: 0.05 * np.log10(w / 6000.0)))
        self.assertEqual(set(res["corrections"]), {"g1", "LDSS_y.dat"})
        files = os.listdir(os.path.join(self.out, "corr"))
        self.assertEqual(len(files), 2)
        wls, c, _ = res["corrections"]["g1"]
        for bn in ("UVB_x.dat", "VIS_x.dat"):
            w, lf, le, m = self.mangled(bn)
            np.testing.assert_allclose(m - 0.1, np.interp(np.log10(w), np.log10(wls), c), atol=1e-6)
        # the overlap of the two arms (6200-6400 A) gets the same correction from both
        uw, _, _, um = self.mangled("UVB_x.dat")
        vw, _, _, vm = self.mangled("VIS_x.dat")
        x = np.log10([6250.0, 6350.0])
        np.testing.assert_allclose(np.interp(x, np.log10(uw), um), np.interp(x, np.log10(vw), vm), atol=1e-6)

    def test_masks_for_closure_are_on_prescaled_grid(self):
        res = self.run_rule(self.extraction())
        for bn, p in self.pre.items():
            self.assertEqual(res["masks"][bn].size, np.loadtxt(p).shape[0])
            np.testing.assert_allclose(res["masks"][bn], 0.1, atol=2e-3)   # GP right: C ~ 0


class TestLaterIterations(_Setup):
    def test_previous_correction_removed_only_where_data_exist(self):
        # the previous correction (+0.3) reached the GP only where the group has data
        prev = {"g1": (10 ** self.lw, np.full(self.lw.size, 0.3), np.zeros(self.lw.size)),
                "LDSS_y.dat": (10 ** self.lw, np.full(self.lw.size, 0.3), np.zeros(self.lw.size))}
        lo, hi = 4500.0, 9498.0
        ext = self.extraction(lambda w: np.where((w >= lo) & (w <= hi), 0.3, 0.0))
        res = self.run_rule({k: v for k, v in ext.items() if k != "LDSS_y.dat"} |
                            {"LDSS_y.dat": self.extraction(
                                lambda w: np.where((w >= 4500) & (w <= 7398), 0.3, 0.0))["LDSS_y.dat"]},
                            prev=prev)
        for uid in ("g1", "LDSS_y.dat"):
            wls, c, _ = res["corrections"][uid]
            np.testing.assert_allclose(np.interp([5000, 7000, 12000], wls, c), 0.0, atol=3e-3)
        # removing it everywhere would have shifted the J-like filter (11000-12500 A)
        p = igm.previous_correction_on(np.array([5000.0, 12000.0]), "g1", prev, [(4500.0, 9500.0)])
        np.testing.assert_allclose(p, [0.3, 0.0])

    def test_total_is_step5_plus_current_correction_only(self):
        r1 = self.run_rule(self.extraction(lambda w: -0.2 + 0.0 * w), out=os.path.join(self.td.name, "it1"))
        # next GP: follows its inputs where there are data (truth + 0.1 + 0.2), photometry elsewhere
        def follows(w):
            return np.where((w >= 4500) & (w <= 9500), 0.1 + 0.2, 0.0)
        r2 = self.run_rule(self.extraction(follows), prev=r1["corrections"], out=os.path.join(self.td.name, "it2"))
        wls2, c2, _ = r2["corrections"]["g1"]
        w, lf, le, m = self.mangled("UVB_x.dat", out=os.path.join(self.td.name, "it2"))
        s5 = np.loadtxt(os.path.join(self.step5, "%.6f_mangled_spec.txt" % self.members["UVB_x.dat"][0]))
        np.testing.assert_allclose(lf, s5[:, 1] + np.interp(np.log10(w), np.log10(wls2), c2), atol=2e-5)
        np.testing.assert_allclose(m, s5[:, 3] + np.interp(np.log10(w), np.log10(wls2), c2), atol=2e-5)
        # GP minus C1 (0.2) leaves the step-5 spectrum, 0.1 dex too bright: C2 ~ -0.1
        np.testing.assert_allclose(np.interp([5000, 8500], wls2, c2), -0.1, atol=1e-2)
        # (the filter outside the data, correction 0, pulls the smooth curve a little near 9000 A)

    def test_previous_correction_is_zero_on_first_iteration(self):
        self.assertTrue(np.all(igm.previous_correction_on(np.array([5000.0]), "g1", {}, [(0, 1e5)]) == 0))


class TestModeAndUnits(unittest.TestCase):
    def test_mode_validation(self):
        import pipeline_config as pconf
        old = pconf.ITER_REMANGLE_MODE
        try:
            self.assertEqual(igm.remangle_mode(), "gp_correction")
            pconf.ITER_REMANGLE_MODE = "nonsense"
            with self.assertRaises(ValueError):
                igm.remangle_mode()
        finally:
            pconf.ITER_REMANGLE_MODE = old

    def test_units_without_groups_are_single_spectra(self):
        es = [SimpleNamespace(basename="b.dat", mjd=2.0), SimpleNamespace(basename="a.dat", mjd=1.0)]
        with tempfile.TemporaryDirectory() as d:
            units = igm.mangle_units(es, output_dir=d, snname="TEST", bundle_aware=False)
        self.assertEqual([u[0] for u in units], ["a.dat", "b.dat"])


if __name__ == "__main__":
    unittest.main()
