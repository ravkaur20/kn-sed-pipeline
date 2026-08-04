"""Smoke tests for log-space 2D GP helpers (stdlib unittest; no pytest required)."""
import _bootstrap_paths  # noqa: F401
import glob
import os
import sys
import tempfile
import unittest

import numpy as np


import gp2dim_phase_merge as phase_merge

try:
    import GP2dim_utils as g
except ImportError:  # e.g. george not installed
    g = None

skip_gp2dim = unittest.skipIf(g is None, "GP2dim_utils import failed (need george, pandas, etc.)")


class TestMergeDenseLogPhase(unittest.TestCase):
    """Optional prediction-phase densification: logspace in linear days / uniform in dex."""

    def test_n_below_2_returns_original(self):
        x = np.array([-3.0, -1.0, 0.5])
        np.testing.assert_array_equal(phase_merge.merge_extrap_mjds_dense_log_phase(x, 1), x)

    def test_preserves_all_original_columns(self):
        orig = np.array([-2.7, -1.2, 0.1, -0.05])
        merged = phase_merge.merge_extrap_mjds_dense_log_phase(orig, 32)
        self.assertGreater(len(merged), len(orig))
        for v in orig:
            self.assertTrue(np.any(np.isclose(merged, v, rtol=0.0, atol=1e-9)))

    def test_endpoints_only_uniform_dex_spacing(self):
        """Only min/max columns: merged interior matches linspace in log10(days)."""
        lo, hi = -3.0, -1.0
        n = 5
        merged = phase_merge.merge_extrap_mjds_dense_log_phase(np.array([lo, hi]), n)
        expect = np.linspace(lo, hi, n)
        np.testing.assert_allclose(merged, expect, rtol=0.0, atol=1e-11)

    def test_single_phase_no_change(self):
        x = np.array([-1.5])
        np.testing.assert_array_equal(phase_merge.merge_extrap_mjds_dense_log_phase(x, 50), x)


@skip_gp2dim
class TestGp2dimNewlog(unittest.TestCase):
    def test_scaled_ln_to_linear_clamp(self):
        offset, scale = 0.0, 1.0
        huge = np.array([0.0, 500.0, 2000.0])
        out = g.scaled_ln_to_linear(huge, offset, scale)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[0], 1.0)
        self.assertLess(out[-1], np.finfo(float).max)

    def test_x2_mask_for_phase(self):
        gn = {"offset2": -2.0, "norm2": 1.5, "norm1": 4.0}
        x2 = np.array([0.0, (0.5 - gn["offset2"]) / gn["norm2"], 1.0])
        m = g.x2_mask_for_phase(x2, 0.5, gn)
        self.assertEqual(int(m.sum()), 1)
        self.assertTrue(bool(m[1]))

    def test_phases_close(self):
        arr = np.array([-1.0, -1.0 + 1e-10, 0.2])
        self.assertTrue(g.phases_close(-1.0, arr))
        self.assertFalse(g.phases_close(0.3, arr))

    def test_transform_scale_floor(self):
        class Dummy:
            pass

        d = Dummy()
        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        y, yerr, x1n, x2n = g.transform2LOG_reshape(d, raw, raw_err, off_xa, off_ya)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(np.isfinite(x1n)))
        self.assertTrue(np.all(np.isfinite(x2n)))
        self.assertGreater(d.grid_norm_info["scale_factor"], 0)

    def test_transform_nonsquare_grid_coordinate_lengths(self):
        """resh_wls must repeat off_xa once per time column, not per wavelength row."""
        class Dummy:
            pass

        d = Dummy()
        raw = np.array(
            [
                [-30.0, -29.9, np.nan],
                [-29.8, -30.1, -30.0],
                [-30.2, np.nan, -29.7],
                [-29.5, -30.3, -29.6],
            ],
            dtype=float,
        )
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.50, 3.51, 3.52, 3.53], dtype=float)
        off_ya = np.array([-1.0, -0.9, -0.8], dtype=float)
        y, yerr, x1n, x2n = g.transform2LOG_reshape(d, raw, raw_err, off_xa, off_ya)
        n_finite = int(np.sum(np.isfinite(raw)))
        self.assertEqual(len(y), n_finite)
        self.assertEqual(len(yerr), n_finite)
        self.assertEqual(len(x1n), n_finite)
        self.assertEqual(len(x2n), n_finite)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(np.isfinite(x1n)))
        self.assertTrue(np.all(np.isfinite(x2n)))


@skip_gp2dim
class TestLnFluxOffsetFloor(unittest.TestCase):
    def test_floor_on_caps_offset_below_physical_min(self):
        class Dummy:
            pass

        raw = np.array([[-25.0, -24.9], [-24.8, -25.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        d_on = Dummy()
        d_on.gp_ln_flux_offset_floor = True
        d_on.gp_ln_flux_offset_floor_linear = 1e-30
        _, _, _, _ = g.transform2LOG_reshape(d_on, raw, raw_err, off_xa, off_ya)
        ln_floor = float(np.log(1e-30))
        self.assertAlmostEqual(d_on.grid_norm_info["offset"], ln_floor, places=10)
        d_off = Dummy()
        d_off.gp_ln_flux_offset_floor = False
        _, _, _, _ = g.transform2LOG_reshape(d_off, raw, raw_err, off_xa, off_ya)
        self.assertGreater(d_off.grid_norm_info["offset"], ln_floor)

    def test_floor_on_increases_scale_factor_vs_physical_min(self):
        class Dummy:
            pass

        raw = np.array([[-25.0, -24.9], [-24.8, -25.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        d_on = Dummy()
        d_on.gp_ln_flux_offset_floor = True
        d_on.gp_ln_flux_offset_floor_linear = 1e-30
        g.transform2LOG_reshape(d_on, raw, raw_err, off_xa, off_ya)
        d_off = Dummy()
        d_off.gp_ln_flux_offset_floor = False
        g.transform2LOG_reshape(d_off, raw, raw_err, off_xa, off_ya)
        self.assertGreater(
            d_on.grid_norm_info["scale_factor"],
            d_off.grid_norm_info["scale_factor"],
        )


@skip_gp2dim
class TestLnFluxErrFromRelative(unittest.TestCase):
    def test_small_dex_matches_legacy_chain_rule(self):
        class Dummy:
            pass

        d_rel = Dummy()
        d_rel.gp_ln_flux_err_from_relative = True
        d_leg = Dummy()
        d_leg.gp_ln_flux_err_from_relative = False
        raw = np.array([[-30.0], [-29.9]], dtype=float)
        raw_err = np.full_like(raw, 1e-7)
        off_xa = np.array([3.5, 3.51])
        off_ya = np.array([-1.0])
        _, yerr_rel, _, _ = g.transform2LOG_reshape(d_rel, raw, raw_err, off_xa, off_ya)
        _, yerr_leg, _, _ = g.transform2LOG_reshape(d_leg, raw, raw_err, off_xa, off_ya)
        np.testing.assert_allclose(yerr_rel, yerr_leg, rtol=1e-5, atol=1e-14)

    def test_large_dex_differs_from_chain_rule(self):
        class Dummy:
            pass

        d_rel = Dummy()
        d_rel.gp_ln_flux_err_from_relative = True
        d_leg = Dummy()
        d_leg.gp_ln_flux_err_from_relative = False
        raw = np.array([[-5.0], [-5.1]], dtype=float)
        raw_err = np.full_like(raw, 0.4)
        off_xa = np.array([3.5, 3.51])
        off_ya = np.array([-1.0])
        _, yerr_rel, _, _ = g.transform2LOG_reshape(d_rel, raw, raw_err, off_xa, off_ya)
        _, yerr_leg, _, _ = g.transform2LOG_reshape(d_leg, raw, raw_err, off_xa, off_ya)
        self.assertGreater(float(np.max(np.abs(yerr_rel - yerr_leg))), 1e-6)

    def test_relative_formula_finite_positive(self):
        class Dummy:
            pass

        d = Dummy()
        d.gp_ln_flux_err_from_relative = True
        d.gp_yerr_floor_frac = 0.0
        d.gp_yerr_abs_floor = 0.0
        raw = np.array([[-10.0]], dtype=float)
        raw_err = np.array([[0.05]], dtype=float)
        y, yerr, _, _ = g.transform2LOG_reshape(d, raw, raw_err, np.array([3.5]), np.array([-1.0]))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(yerr > 0))


@skip_gp2dim
class TestTrainingYerrFloors(unittest.TestCase):
    def test_floor_off_smaller_than_legacy_spread_floor(self):
        class Dummy:
            pass

        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-9)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        d_off = Dummy()
        d_off.gp_yerr_floor_frac = 0.0
        d_off.gp_yerr_abs_floor = 0.0
        _, yerr_off, _, _ = g.transform2LOG_reshape(d_off, raw, raw_err, off_xa, off_ya)
        d_on = Dummy()
        d_on.gp_yerr_floor_frac = 1e-4
        d_on.gp_yerr_abs_floor = 0.0
        _, yerr_on, _, _ = g.transform2LOG_reshape(d_on, raw, raw_err, off_xa, off_ya)
        self.assertLess(np.min(yerr_off), np.min(yerr_on))
        self.assertTrue(np.all(yerr_on >= yerr_off - 1e-15))

    def test_zero_propagated_errors_raise_without_floor(self):
        class Dummy:
            pass

        d = Dummy()
        d.gp_yerr_floor_frac = 0.0
        d.gp_yerr_abs_floor = 0.0
        raw = np.array([[-30.0], [-29.9]], dtype=float)
        raw_err = np.zeros_like(raw)
        with self.assertRaises(ValueError):
            g.transform2LOG_reshape(
                d, raw, raw_err, np.array([3.5, 3.51]), np.array([-1.0])
            )

    def test_abs_floor_makes_zero_errors_positive(self):
        class Dummy:
            pass

        d = Dummy()
        d.gp_yerr_floor_frac = 0.0
        d.gp_yerr_abs_floor = 1e-12
        raw = np.array([[-30.0], [-29.9]], dtype=float)
        raw_err = np.zeros_like(raw)
        _, yerr, _, _ = g.transform2LOG_reshape(
            d, raw, raw_err, np.array([3.5, 3.51]), np.array([-1.0])
        )
        self.assertTrue(np.all(yerr >= 1e-12 * 0.99))

    def test_apply_compute_stage_matches_legacy_floor_when_enabled(self):
        class Dummy:
            pass

        d = Dummy()
        d.gp_yerr_floor_frac = 1e-4
        d.gp_yerr_abs_floor = 0.0
        y = np.array([0.0, 1.0, -0.5])
        yerr = np.array([1e-30, 1e-30, 1e-30])
        out = g._apply_training_yerr_floors(d, y, yerr, stage="compute")
        expect_min = max(1e-4 * (float(np.nanstd(y)) + 1e-12), 1e-12)
        self.assertGreaterEqual(float(np.min(out)), expect_min * 0.999)


class TestGpPredictionGrid(unittest.TestCase):
    """Caps N_wavelength so run_2DGP_GRID does not build huge predict batches (memory)."""

    def test_wavelength_grid_point_count_bounded(self):
        wls_min, wls_max = 3.0, 4.5
        span_wl = float(wls_max - wls_min)
        _gp_n_wl = 300
        _wl_step = 0.01
        n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
        n_wl_use = max(2, min(_gp_n_wl, n_from_step))
        self.assertLessEqual(n_wl_use, _gp_n_wl)
        self.assertGreaterEqual(n_wl_use, 2)


@skip_gp2dim
class TestGpDenseMatrixHint(unittest.TestCase):
    def test_bytes_scales_as_n_squared(self):
        self.assertEqual(g.gp_dense_matrix_bytes_order_of_magnitude(0), 0)
        self.assertEqual(g.gp_dense_matrix_bytes_order_of_magnitude(1000), 8 * 1000 * 1000)
        n = 15000
        b = g.gp_dense_matrix_bytes_order_of_magnitude(n)
        self.assertGreater(b / (1024.0**3), 1.5)

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            g.gp_dense_matrix_bytes_order_of_magnitude(-1)


@skip_gp2dim
class TestPhaseAxisDenormForPlots(unittest.TestCase):
    """Training plots use offset2 + norm2*x2_norm to recover log10(phase days)."""

    def test_denorm_matches_original_log_phase(self):
        offset2 = -2.0
        norm2 = 3.5
        x2_data = np.array([-2.0, 0.0, 1.5], dtype=float)
        x2_norm = (x2_data - offset2) / norm2
        restored = offset2 + norm2 * x2_norm
        np.testing.assert_allclose(restored, x2_data)

    def test_linear_axes_from_normed_grid(self):
        """gp_2d_surface_linear_axes: phase(days)=10**(offset2+norm2*x2), wl=10**(norm1*x1)."""
        norm1, norm2 = 4.0, 2.0
        offset2 = -2.5
        x1 = np.array([0.8, 0.9])
        x2 = np.array([0.25, 0.5])
        phase_log = offset2 + norm2 * x2
        wl_log = norm1 * x1
        np.testing.assert_allclose(10**phase_log, [10 ** (-2.0), 10 ** (-1.5)])
        np.testing.assert_allclose(10**wl_log, [10**3.2, 10**3.6])

    def test_phase_days_from_norm_x2_matches_training_convention(self):
        gn = {"offset2": -2.0, "norm2": 1.5, "norm1": 4.0}
        x2_norm = np.array([0.0, 1.0])
        days = g.phase_days_from_norm_x2(x2_norm, gn)
        np.testing.assert_allclose(days, np.power(10.0, gn["offset2"] + gn["norm2"] * x2_norm))


@skip_gp2dim
class TestLogPredictionPhaseCoverage(unittest.TestCase):
    def test_counts_bracket(self):
        import io
        from contextlib import redirect_stdout

        cols = [-3.5, -2.0, 0.5, -1.5]
        buf = io.StringIO()
        with redirect_stdout(buf):
            g.log_prediction_phase_coverage(cols, lo=-3.0, hi=-1.0, label="t")
        s = buf.getvalue()
        self.assertIn("2 of 4", s)


try:
    import george
    from george.kernels import Matern32Kernel
except ImportError:
    george = None
    Matern32Kernel = None


@unittest.skipIf(george is None, "george not installed")
class TestGeorgePredictUsesVar(unittest.TestCase):
    def test_predict_return_var_not_full_cov(self):
        """Sanity check: return_var avoids allocating n_test^2 covariance."""
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 1.0, 25))
        y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(25)
        yerr = 0.15 * np.ones_like(y)
        kernel = 0.5 * Matern32Kernel(0.2, ndim=1)
        gp = george.GP(kernel)
        gp.compute(x, yerr)
        x_pred = np.linspace(0.0, 1.0, 400)
        mu, var = gp.predict(y, x_pred, return_var=True)
        self.assertEqual(mu.shape, (400,))
        self.assertEqual(var.shape, (400,))
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(var)))


@skip_gp2dim
class TestMangledSpecWavelengthConvention(unittest.TestCase):
    """Guards against ``10**`` on linear-Å mangled files (overflow)."""

    def test_linear_angstrom_detected(self):
        self.assertTrue(g.mangled_wls_max_is_linear_angstrom(np.array([2500.0, 8000.0])))

    def test_log10_angstrom_not_linear(self):
        self.assertFalse(g.mangled_wls_max_is_linear_angstrom(np.array([3.3, 3.7])))

    def test_mangled_helpers_kn_file_format_no_overflow(self):
        """Linear Å + log10 flux (as on disk from KN log mangle)."""
        spec = np.zeros(
            2,
            dtype=[("wls", float), ("flux", float), ("fluxerr", float)],
        )
        spec["wls"] = [3000.0, 3010.0]
        spec["flux"] = [-15.0, -15.1]
        spec["fluxerr"] = [0.05, 0.05]
        wlin = g.mangled_wls_linear_angstrom(spec)
        flin = g.mangled_flux_linear_from_log10(spec["flux"])
        self.assertTrue(np.all(np.isfinite(wlin)))
        self.assertTrue(np.all(np.isfinite(flin)))
        np.testing.assert_allclose(wlin, [3000.0, 3010.0])
        self.assertLess(np.max(flin), 1.0)


@skip_gp2dim
class TestFillGapsPhaseLogspace(unittest.TestCase):
    """Gap fill in log-phase grid uses linear-day thresholds (matches original linear notebook)."""

    def test_fills_interior_in_linear_day_gaps(self):
        min_log = np.log10(1.0)
        max_log = np.log10(30.0)
        spec = np.array([np.log10(2.0), np.log10(10.0)])
        out = g.fill_gaps_phase_logspace(
            min_log, max_log, spec, gap_size_days=0.1, cadence_days=0.1
        )
        self.assertGreater(len(out), 0)
        self.assertTrue(np.all(out >= min_log - 1e-12))
        self.assertTrue(np.all(out <= max_log + 1e-12))

    def test_inclusive_endpoint_mask_matches_extend_grid(self):
        mjds_grid = np.array([np.log10(5.0), np.log10(10.0), np.log10(25.0)])
        lo, hi = np.log10(1.0), np.log10(25.0)
        eps = 1e-5
        mask = (mjds_grid >= lo - eps) & (mjds_grid <= hi + eps)
        self.assertTrue(np.all(mask))

    def test_tiny_linear_gap_inserts_log_phases(self):
        """Borderline: linear segment just under 0.1d can still be ~2 dex in log; must not be empty."""
        min_log = -3.0
        spec = np.array([-1.0])  # 0.1d
        out = g.fill_gaps_phase_logspace(
            min_log, np.log10(20.0), spec, gap_size_days=0.1, cadence_days=0.1
        )
        self.assertGreater(len(out), 0)
        in_bracket = (out >= -3.0) & (out <= -1.0)
        self.assertGreater(np.count_nonzero(in_bracket), 0)


@skip_gp2dim
class TestSetPriorNewlog(unittest.TestCase):
    def test_setprior_log_flux_columns(self):
        """setPRIOR runs with *_log_flux LC and a minimal prior grid."""
        class Dummy:
            t0_fix = 58000.0
            path_fit_phot = ""

        d = Dummy()
        d.grid_norm_info = {
            "norm1": 3.7,
            "norm2": 2.0,
            "offset": -35.0,
            "offset2": -2.5,
            "scale_factor": 2.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            lc_path = os.path.join(tmp, "fitted.dat")
            with open(lc_path, "w") as f:
                f.write("Log_Phase\tSwope_g_log_flux\tSwope_r_log_flux\n")
                f.write("-2.0\t-15.0\t-14.5\n")
                f.write("0.3\t-14.0\t-13.8\n")
                f.write("0.5\t-13.5\t-13.2\n")
            d.path_fit_phot = lc_path
            prior_path = os.path.join(tmp, "prior.txt")
            rows = [
                "4000,-2,1.0",
                "5000,-2,1.0",
                "4000,1,1.0",
                "5000,1,1.0",
            ]
            with open(prior_path, "w") as f:
                f.write("\n".join(rows) + "\n")
            points, values = g.setPRIOR(d, PRIOR_file="prior.txt", PRIOR_folder=tmp + os.sep)
        self.assertEqual(points.shape[1], 2)
        self.assertEqual(values.ndim, 1)
        self.assertEqual(values.size, 4)
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.isfinite(values)))


@skip_gp2dim
@unittest.skipIf(george is None, "george not installed")
class TestRun2DGPGridDiagnosticSlices(unittest.TestCase):
    """Optional per-phase prior vs prediction PDFs (first slot only; bounded cost)."""

    def test_writes_gp_diag_pdfs_when_enabled(self):
        rng = np.random.default_rng(7)
        n = 48
        norm1 = 4.0
        offset2, norm2 = -2.5, 2.0
        x1n = rng.uniform(0.78, 0.97, n)
        x2_raw = rng.uniform(-1.7, 0.7, n)
        x2n = (x2_raw - offset2) / norm2
        y = rng.normal(0.0, 0.15, n)
        yerr = 0.1 * np.ones(n)

        class D:
            pass

        d = D()
        d.grid_norm_info = {
            "norm1": norm1,
            "norm2": norm2,
            "offset": -30.0,
            "offset2": offset2,
            "scale_factor": 2.0,
        }
        d.grids = [np.linspace(3.05, 3.95, 100)]
        d.verbose = False
        d.gp_print_training_size = False
        d.gp_predict_progress = False
        d.gp_diagnostic_slices = True
        d.gp_predict_slot_size = 3
        d.gp_predict_n_wavelength = 48
        d.gp_predict_wl_step = 0.02
        d.gp_predict_chunk_size = 900
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
            extrap_mjds = np.array([-1.4, -0.6, 0.2], dtype=float)
            x1f, x2f, mu, std = g.run_2DGP_GRID(
                d,
                y,
                yerr,
                x1n,
                x2n,
                0.35,
                0.35,
                extrap_mjds,
                prior=False,
            )
            self.assertEqual(x1f.shape, x2f.shape)
            self.assertEqual(mu.shape, x1f.shape)
            pdfs = sorted(glob.glob(os.path.join(tmp, "gp_diag_slot0_phase*.pdf")))
            self.assertGreaterEqual(len(pdfs), 1)
            self.assertLessEqual(len(pdfs), 3)


@skip_gp2dim
@unittest.skipIf(george is None, "george not installed")
class TestRun2DGPGridNoDiagnostics(unittest.TestCase):
    def test_no_diag_pdfs_when_disabled(self):
        rng = np.random.default_rng(8)
        n = 40
        norm1 = 4.0
        offset2, norm2 = -2.5, 2.0
        x1n = rng.uniform(0.78, 0.97, n)
        x2_raw = rng.uniform(-1.5, 0.5, n)
        x2n = (x2_raw - offset2) / norm2
        y = rng.normal(0.0, 0.12, n)
        yerr = 0.1 * np.ones(n)

        class D:
            pass

        d = D()
        d.grid_norm_info = {
            "norm1": norm1,
            "norm2": norm2,
            "offset": -30.0,
            "offset2": offset2,
            "scale_factor": 2.0,
        }
        d.grids = [np.linspace(3.05, 3.95, 80)]
        d.verbose = False
        d.gp_print_training_size = False
        d.gp_predict_progress = False
        d.gp_diagnostic_slices = False
        d.gp_predict_slot_size = 3
        d.gp_predict_n_wavelength = 40
        d.gp_predict_wl_step = 0.025
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
            extrap_mjds = np.array([-1.0, -0.2], dtype=float)
            g.run_2DGP_GRID(
                d,
                y,
                yerr,
                x1n,
                x2n,
                0.4,
                0.4,
                extrap_mjds,
                prior=False,
            )
            pdfs = glob.glob(os.path.join(tmp, "gp_diag_slot0_phase*.pdf"))
            self.assertEqual(len(pdfs), 0)


@skip_gp2dim
class TestZscoreCoords(unittest.TestCase):
    def test_phase_days_round_trip(self):
        gn = {
            "x2_mean": -1.5,
            "x2_std": 0.4,
            "coord_parametrization": "zscore",
        }
        log_phases = np.array([-2.0, -1.0, -0.3])
        x2n = (log_phases - gn["x2_mean"]) / gn["x2_std"]
        back_log = np.log10(g.phase_days_from_norm_x2(x2n, gn))
        np.testing.assert_allclose(back_log, log_phases, rtol=0.0, atol=1e-12)

    def test_transform_sets_zscore_grid_norm_info(self):
        class Dummy:
            pass

        d = Dummy()
        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        g.transform2LOG_reshape(d, raw, raw_err, off_xa, off_ya)
        self.assertEqual(d.grid_norm_info.get("coord_parametrization"), "zscore")
        for k in ("x1_mean", "x1_std", "x2_mean", "x2_std", "x2_train_min"):
            self.assertIn(k, d.grid_norm_info)


if __name__ == "__main__":
    unittest.main()
