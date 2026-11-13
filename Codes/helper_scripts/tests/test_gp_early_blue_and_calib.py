"""Early-time blue step modes and the spectral calibration-error options."""
import _bootstrap_paths  # noqa: F401
import unittest

import numpy as np

from twodim_gp import gp_utils as gu  # noqa: E402
from twodim_gp import run_inference as ri  # noqa: E402


def _grid(mu_fn):
    w = np.linspace(-3.0, 2.0, 41)
    t = np.linspace(-7.0, 1.0, 81)
    W, T = np.meshgrid(w, t)
    X = np.column_stack([W.ravel(), T.ravel()])
    return X, mu_fn(X[:, 0], X[:, 1])


# red brighter than the UV at every phase: the case that made the shelf
def _bump(w, t):
    return 1.0 + 0.05 * np.exp(-0.5 * ((w + 1.5) / 0.7) ** 2) + 0.001 * t


class TestEarlyBlue(unittest.TestCase):
    def _cols(self, X, mu, ph):
        m = np.isclose(X[:, 1], ph)
        o = np.argsort(X[m, 0])
        return X[m, 0][o], mu[m][o]

    def test_hard_leaves_step_blend_does_not(self):
        X, mu = _grid(_bump)
        ph = np.unique(X[:, 1])
        lo, hi = ph[ph < -4].max(), ph[ph >= -4].min()
        hard, _ = ri._enforce_blue_early(X, mu, -4.0, mode="hard")
        blend, _ = ri._enforce_blue_early(X, mu, -4.0, mode="blend", wl_start=-3.0)
        jump = lambda m: np.abs(self._cols(X, m, lo)[1] - self._cols(X, m, hi)[1]).max()
        self.assertGreater(jump(hard), 0.03)
        self.assertLess(jump(blend), 0.015)
        self.assertLess(jump(blend), jump(hard) / 3)

    def test_blend_leaves_uv_and_late_times_alone(self):
        X, mu = _grid(_bump)
        out, _ = ri._enforce_blue_early(X, mu, -4.0, mode="blend", wl_start=-1.0)
        uv = X[:, 0] < -1.0
        np.testing.assert_array_equal(out[uv], mu[uv])
        late = X[:, 1] > -1.0
        np.testing.assert_allclose(out[late], mu[late], atol=1e-6)
        # well before the cutoff, redder than wl_start never brighter than bluer
        w, m = self._cols(X, out, -7.0)
        sel = w >= -1.0
        self.assertTrue(np.all(np.diff(m[sel]) <= 1e-12))

    def test_off_and_legacy_default(self):
        X, mu = _grid(_bump)
        out, n = ri._enforce_blue_early(X, mu, -4.0, mode="off")
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(out, mu)
        self.assertEqual(ri._resolve_early_blue_mode({"enforce_blue_early": True}), "hard")
        self.assertEqual(ri._resolve_early_blue_mode({"enforce_blue_early": False}), "off")
        self.assertEqual(ri._resolve_early_blue_mode({"early_blue_mode": "blend"}), "blend")


def _two_spectra_bundle(offset):
    """Two spectra 0.01 apart in phase, one +offset and one -offset."""
    w = np.linspace(0.0, 1.0, 60)
    X = np.vstack([np.column_stack([w, np.full(60, 0.50)]),
                   np.column_stack([w, np.full(60, 0.51)])])
    y = 1.0 + 0.05 * np.sin(3 * X[:, 0]) + np.r_[np.full(60, offset), np.full(60, -offset)]
    t = np.linspace(0.3, 0.7, 41)
    Wf, Tf = np.meshgrid(w[::6], t)
    return dict(X=X, y=y, yerr=np.full(120, 1e-4), X_fill=np.column_stack([Wf.ravel(), Tf.ravel()]),
                kernel_wls_scale=np.array(0.05), kernel_time_scale=np.array(0.01),
                y_var_scale=np.array(0.01), y_scale_factor=np.array(np.log(10.0)),
                prior_points=np.zeros((0, 2)), prior_values=np.zeros(0))


class TestSpecCalib(unittest.TestCase):
    KW = dict(optimize=False, mean="constant", phot_spec_threshold=10, sigma_spec=1e-3,
              sigma_spec_adaptive_frac=None, early_blue_mode="off", enforce_mono_early=False,
              lt=0.01, lw=0.05)

    def _wiggle(self, **kw):
        out = run_gp_from_bundle_kw(_two_spectra_bundle(0.05), **dict(self.KW, **kw))
        X, mu = out["X_fill"], out["mu"]
        sel = np.isclose(X[:, 0], 0.0)
        return np.ptp(mu[sel][np.argsort(X[sel, 1])])

    def test_correlated_absorbs_spectrum_offsets(self):
        none = self._wiggle()
        corr = self._wiggle(spec_calib_mode="correlated", spec_calib_err_dex=0.1)
        self.assertGreater(none, 0.05)          # follows the +/-0.05 offsets
        self.assertLess(corr, none / 3)         # offsets go into the shared term

    def test_dex_converted_to_scaled_units(self):
        cfg = dict(spec_calib_mode="independent", spec_calib_err_dex=0.01)
        self.assertAlmostEqual(ri._spec_calib_sigma_scaled(cfg, 31.0), 0.01 * np.log(10) / 31.0)
        self.assertEqual(ri._spec_calib_sigma_scaled(dict(cfg, spec_calib_mode="none"), 31.0), 0.0)
        with self.assertRaises(ValueError):
            ri._spec_calib_sigma_scaled(cfg, None)

    def test_kernel_offset_only_within_a_spectrum(self):
        X = np.array([[0, 0], [0.1, 0], [0, 1], [0.5, 2.0]], float)
        pc = np.array(["spec", "spec", "spec", "phot"])
        cfg = gu.KernelConfig()
        d = gu.build_kernel(cfg, 0.1).get_value(gu.add_spec_calib_columns(X, pc)) \
            - gu.build_kernel(cfg).get_value(X)
        np.testing.assert_allclose(d, 0.01 * np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]), atol=1e-12)


def run_gp_from_bundle_kw(bundle, **kw):
    return ri.run_gp_from_bundle(bundle, **kw)


if __name__ == "__main__":
    unittest.main()
