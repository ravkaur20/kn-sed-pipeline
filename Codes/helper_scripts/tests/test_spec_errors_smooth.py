"""Tests for spec_errors_smooth (notebook 0.3 errors + smoothing)."""
import _bootstrap_paths  # noqa: F401
import unittest

import numpy as np

import spec_errors_smooth as se  # noqa: E402


def _fake_spec(pix_A, sigma=0.05, seed=0, wl0=4000.0, wl1=9000.0, err=True):
    """Smooth continuum with broad features plus white noise of known sigma."""
    rng = np.random.default_rng(seed)
    wl = np.arange(wl0, wl1, pix_A)
    truth = 1.0 + 0.3 * np.sin(wl / 400.0) + 0.2 * np.exp(-0.5 * ((wl - 6000) / 150) ** 2)
    flux = truth + rng.normal(0, sigma, len(wl))
    return {"name": "fake", "mjd": 0.0, "wl": wl, "flux": flux,
            "err": np.full(len(wl), sigma) if err else None,
            "weight": np.ones(len(wl))}, truth


class TestGsmooth(unittest.TestCase):
    def test_constant_is_preserved(self):
        wl = np.linspace(4000, 5000, 500)
        r = se.gsmooth(wl, np.full(500, 3.0), np.ones(500), 0.002)
        np.testing.assert_allclose(r["y"], 3.0)

    def test_bad_pixels_get_no_weight(self):
        wl = np.linspace(4000, 5000, 500)
        y = np.ones(500)
        y[200:220] = 1e6
        good = np.ones(500, bool)
        good[200:220] = False
        r = se.gsmooth(wl, y, np.ones(500), 0.002, good=good)
        np.testing.assert_allclose(r["y"], 1.0)
        self.assertGreater(r["bad_frac"][210], 0.5)
        self.assertEqual(r["bad_frac"][0], 0.0)

    def test_propagated_sigma_matches_scatter(self):
        # smoothed value scatter over many noise draws == propagated sigma
        wl = np.arange(5000.0, 5600.0, 2.0)
        x = np.array([5300.0])
        vals = []
        for s in range(300):
            y = np.random.default_rng(s).normal(0, 0.1, len(wl))
            r = se.gsmooth(wl, y, np.full(len(wl), 100.0), 0.002, x_out=x)
            vals.append(r["y"][0])
        pred = np.sqrt(r["var"][0])
        self.assertAlmostEqual(np.std(vals) / pred, 1.0, delta=0.12)


class TestErrorEstimate(unittest.TestCase):
    def test_estimate_recovers_sigma_for_any_pixel_size(self):
        for pix in (0.2, 0.6, 2.0, 4.0):
            spec, _ = _fake_spec(pix, sigma=0.05)
            sig, info = se.estimate_errors(spec)
            ratio = np.median(sig[500:-500]) / 0.05
            self.assertAlmostEqual(ratio, 1.0, delta=0.07, msg="pixel %g A" % pix)

    def test_dynamic_factor_grows_for_coarse_pixels(self):
        f_fine = np.median(se.estimate_errors(_fake_spec(0.2)[0])[1]["factor"])
        f_coarse = np.median(se.estimate_errors(_fake_spec(4.0)[0])[1]["factor"])
        self.assertAlmostEqual(f_fine, np.sqrt(np.pi / 2), delta=0.02)
        self.assertGreater(f_coarse, f_fine + 0.03)

    def test_rescale_constant_fixes_wrong_errors(self):
        spec, _ = _fake_spec(1.0, sigma=0.05)
        spec["err"] = spec["err"] * 3.0          # file errors 3x too big
        p = dict(se.DEFAULTS, err_mode="rescale")
        sigma, info = se.choose_errors(spec, p)
        self.assertAlmostEqual(np.median(sigma) / 0.05, 1.0, delta=0.07)
        self.assertAlmostEqual(info["scale"]["scale"], 1 / 3.0, delta=0.03)

    def test_rescale_linear_returns_coeffs(self):
        spec, _ = _fake_spec(1.0)
        p = dict(se.DEFAULTS, err_mode="rescale", rescale_shape="linear")
        _, info = se.choose_errors(spec, p)
        self.assertEqual(len(info["scale"]["coeffs"]), 2)

    def test_trust_keeps_file_errors(self):
        spec, _ = _fake_spec(1.0)
        sigma, info = se.choose_errors(spec, dict(se.DEFAULTS))
        np.testing.assert_array_equal(sigma, spec["err"])

    def test_no_errors_uses_estimate(self):
        spec, _ = _fake_spec(1.0, err=False)
        sigma, info = se.choose_errors(spec, dict(se.DEFAULTS, err_mode="trust"))
        self.assertTrue(info["mode"].startswith("estimate"))
        self.assertAlmostEqual(np.median(sigma) / 0.05, 1.0, delta=0.07)


class TestReduceOne(unittest.TestCase):
    def test_end_to_end(self):
        spec, truth = _fake_spec(0.5, sigma=0.05)
        spec["weight"][4000:4400] = 0           # a clipped region
        res = se.reduce_one(spec, dict(se.DEFAULTS))
        self.assertLessEqual(res["vexp"], res["vexp_max"])
        # one point per kernel sigma
        step = np.median(np.diff(np.log(res["out_wl"])))
        self.assertAlmostEqual(step, res["vexp"], delta=res["vexp"] * 0.01)
        # errors right -> residual/sigma scatter ~ 1
        self.assertAlmostEqual(res["chi_std"], 1.0, delta=0.1)
        # points mostly inside the clipped region are weight 0
        lo, hi = spec["wl"][4000], spec["wl"][4399]
        inside = (res["out_wl"] > lo + 30) & (res["out_wl"] < hi - 30)
        self.assertTrue(inside.any())
        self.assertTrue(np.all(res["out_weight"][inside] == 0))
        # smoothed points follow the truth within ~3 sigma
        t = np.interp(res["out_wl"], spec["wl"], truth)
        w = res["out_weight"] > 0
        pull = (res["out_flux"][w] - t[w]) / res["out_err"][w]
        self.assertLess(np.median(np.abs(pull)), 2.0)

    def test_manual_vexp_and_kaepora_rule(self):
        spec, _ = _fake_spec(1.0)
        r1 = se.reduce_one(spec, dict(se.DEFAULTS, vexp=0.003))
        self.assertEqual(r1["vexp"], 0.003)
        self.assertEqual(r1["width_rule_used"], "manual")
        r2 = se.reduce_one(spec, dict(se.DEFAULTS, width_rule="kaepora"))
        self.assertEqual(r2["vexp"], r2["vexp_kaepora"])
        self.assertTrue(0.001 <= r2["vexp_kaepora"] <= 0.0045)

    def test_knee_finds_bend(self):
        v = np.geomspace(1e-4, 1e-2, 30)
        stats = np.where(v < 3e-3, (v / 3e-3) ** -2.5, (v / 3e-3) ** -0.5)
        k = se.knee_vexp(v, stats)
        self.assertTrue(2e-3 < k < 5e-3, k)

    def test_knee_falls_back_to_cap(self):
        v = np.geomspace(1e-4, 1e-2, 30)
        self.assertEqual(se.knee_vexp(v, v ** -2.5), v[-1])

    def test_cap_rule(self):
        spec, _ = _fake_spec(1.0)
        r = se.reduce_one(spec, dict(se.DEFAULTS, width_rule="cap"))
        self.assertEqual(r["vexp"], r["vexp_max"])

    def test_cap_is_100A_at_red_end(self):
        spec, _ = _fake_spec(1.0)
        self.assertAlmostEqual(se.vexp_cap(spec["wl"]) * spec["wl"].max(), 100.0)


if __name__ == "__main__":
    unittest.main()
