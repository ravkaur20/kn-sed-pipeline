"""Unit + smoke tests for 2D GP inference (``twodim_gp.run_inference``, ``GP2dim_utils_iter``)."""
import _bootstrap_paths  # noqa: F401
import os
import sys
import tempfile
import unittest

import numpy as np


try:
    from twodim_gp import gp_utils as gu
    from twodim_gp.run_inference import _resolve_sigma_spec_min, run_gp_from_bundle
except ImportError:
    gu = None
    _resolve_sigma_spec_min = None
    run_gp_from_bundle = None

skip_inference = unittest.skipIf(
    run_gp_from_bundle is None, "twodim_gp.run_inference import failed (george/scipy?)"
)


def _tiny_bundle():
    rng = np.random.default_rng(42)
    n = 45
    X = rng.uniform(0.12, 0.93, size=(n, 2)).astype(np.float64)
    y = 0.2 * np.sin(4 * np.pi * X[:, 0]) + 0.15 * X[:, 1] + rng.normal(0, 0.04, size=n)
    yerr = np.full(n, 0.06, dtype=np.float64)
    wg = np.linspace(0.2, 0.82, 6)
    ph = np.linspace(0.18, 0.87, 5)
    x1, x2 = [], []
    for a in wg:
        for b in ph:
            x1.append(a)
            x2.append(b)
    X_fill = np.column_stack([x1, x2]).astype(np.float64)
    return dict(
        X=X,
        y=y,
        yerr=yerr,
        X_fill=X_fill,
        kernel_wls_scale=np.float64(1e-2),
        kernel_time_scale=np.float64(1e-2),
        y_var_scale=np.float64(np.var(y)),
        prior_points=np.zeros((0, 2)),
        prior_values=np.zeros(0),
    )


@skip_inference
class TestTwodimGpGpInference(unittest.TestCase):
    def test_run_mean_none_fixed_hypers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bd = _tiny_bundle()
            out = run_gp_from_bundle(
                bd,
                cache_workdir=tmp,
                mean="none",
                optimize=False,
                predict_chunk=500,
                predict_train=False,
            )
        mu = np.asarray(out["mu"], dtype=float)
        std = np.asarray(out["std"], dtype=float)
        self.assertEqual(mu.shape[0], bd["X_fill"].shape[0])
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(std >= 0.0))
        self.assertIn("mu_raw", out)
        self.assertIn("config_final", out)


@skip_inference
class TestTwodimGpGpOptimizerGuardrails(unittest.TestCase):
    def test_adaptive_sigma_spec_min(self):
        yerr = np.array([0.02, 0.03, 0.04, 0.05], dtype=float)
        pc = np.array([gu.SPEC, gu.SPEC, gu.PHOT, gu.SPEC])
        floor = _resolve_sigma_spec_min({"sigma_spec_adaptive_frac": 0.35}, yerr, pc)
        self.assertAlmostEqual(floor, max(0.005, 0.35 * 0.03))

    def test_default_bounds_sigma_spec_min_override(self):
        cfg = gu.KernelConfig(additive_t=True, additive_w=True)
        bounds = dict(zip(cfg.free_param_names(), cfg.default_bounds(sigma_spec_min=0.012)))
        lo = bounds["log_sigma_spec"][0]
        self.assertAlmostEqual(np.exp(lo), 0.012)


@skip_inference
class TestGp2dimGridIterSmoke(unittest.TestCase):
    def test_fill_rowcount_matches_legacy_pattern(self):
        import GP2dim_utils_iter as gpiter

        class D:
            pass

        d = D()
        d.snname = "TEST"
        d.mode = "extrapolate_spectra"
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
        d.grids = (np.linspace(3.35, 3.55, 8), [])
        d.grid_norm_info = {"norm1": 4.0, "norm2": 1.8, "offset2": -2.0, "offset": 0.0, "scale_factor": 1.0}
        d.pipeline_wl_min_a = None
        d.pipeline_wl_max_a = None
        d.gp_predict_n_wavelength = 12
        d.gp_predict_wl_step = 0.05
        d.gp_predict_dense_log_phase = False
        d.gp_predict_dense_log_phase_n = 32
        d.gp_2d_anchor_t0 = False
        d.gp_predict_progress = False
        d.verbose = False
        d.gp_print_training_size = False

        extrap = np.linspace(-2.9, -0.9, 5)
        y = np.linspace(-0.1, 0.1, 20)
        ye = np.full_like(y, 0.05)
        x1n = np.linspace(0.85, 0.93, y.size)
        x2n = np.linspace(0.1, 0.5, y.size)

        wls_min = float(np.min(d.grids[0]))
        wls_max = float(np.max(d.grids[0]))
        span_wl = float(wls_max - wls_min)
        n_from_step = int(np.ceil(span_wl / float(d.gp_predict_wl_step))) + 1
        n_wl_use = max(2, min(int(d.gp_predict_n_wavelength), n_from_step))
        n_expected = n_wl_use * len(extrap)

        def _stub(bundle, **_):
            xf = bundle["X_fill"]
            nn = xf.shape[0]
            z = np.zeros(nn, dtype=float)
            return {
                "mu": z + 1.0,
                "mu_raw": z + 2.0,
                "std": np.ones(nn, dtype=float) * 0.01,
                "var": np.ones(nn, dtype=float) * 0.0001,
                "X_fill": xf,
                "log_likelihood": 1.0,
                "total_runtime_seconds": 0.0,
                "config_final": {},
            }

        saved_infer = gpiter.run_gp_from_bundle
        saved_export = gpiter.maybe_save_gp_minimal_export
        try:
            gpiter.run_gp_from_bundle = _stub  # type: ignore

            def _no_export(*_args, **_kw):
                return None

            gpiter.maybe_save_gp_minimal_export = _no_export  # type: ignore[attr-defined]

            pts = np.arange(60, dtype=float).reshape(-1, 2)
            vals = np.zeros(pts.shape[0])
            _, _, mf, sf, mrf = gpiter.run_2DGP_GRID_iter(
                d,
                y,
                ye,
                x1n,
                x2n,
                5e-3,
                5e-3,
                extrap,
                prior=True,
                points=pts,
                values=vals,
            )
        finally:
            gpiter.run_gp_from_bundle = saved_infer  # type: ignore
            gpiter.maybe_save_gp_minimal_export = saved_export  # type: ignore[attr-defined]

        self.assertEqual(mf.size, n_expected)
        self.assertEqual(mrf.size, n_expected)
        self.assertEqual(sf.size, n_expected)


class TestTwodimGridPrep(unittest.TestCase):
    def test_mangled_filename_to_mjd(self):
        from twodim_grid_prep import mangled_filename_to_mjd

        self.assertAlmostEqual(
            mangled_filename_to_mjd("57983.9690_mangled_spec.txt"), 57983.9690
        )
        self.assertAlmostEqual(
            mangled_filename_to_mjd("mangled_spec_57983.9690.txt"), 57983.9690
        )

    def test_load_mangled_list_at2017gfo(self):
        import pipeline_config as pc
        from twodim_grid_prep import FullMangledSeries_Class

        out_root = pc.outputs_root()
        sn = pc.SNNAME_DEFAULT
        mangled_dir = os.path.join(out_root, sn, "mangled_spectra")
        if not os.path.isdir(mangled_dir):
            self.skipTest("AT2017gfo mangled_spectra missing (run NB5 first)")

        t0 = pc.SN_EXPLOSION_MJD.get(sn, 57982.52851852)
        with tempfile.TemporaryDirectory() as tmp:
            cls = FullMangledSeries_Class(
                sn,
                t0,
                mode="extrapolate_spectra",
                output_dir=out_root,
                verbose=False,
                prepare_output_dir=False,
            )
            cls.save_plot_path = tmp
            cls.get_mangledspec_list()
            self.assertGreater(len(cls.mangledspec_list), 0)
            one = cls.mangledspec_list[0]
            rec = cls.load_mangledfile(one)
            self.assertIn("wls", rec.dtype.names)
            self.assertIn("flux", rec.dtype.names)


@skip_inference
class TestAt2017gfoGpSmoke(unittest.TestCase):
    """End-to-end smoke: mangled spectra → tiny subsampled 2D GP fit."""

    def test_subsampled_fit_from_mangled_epoch(self):
        import pipeline_config as pc
        from twodim_grid_prep import FullMangledSeries_Class, mangled_filename_to_mjd

        out_root = pc.outputs_root()
        sn = pc.SNNAME_DEFAULT
        mangled_dir = os.path.join(out_root, sn, "mangled_spectra")
        if not os.path.isdir(mangled_dir):
            self.skipTest("AT2017gfo mangled_spectra missing (run NB5 first)")

        t0 = pc.SN_EXPLOSION_MJD[sn]
        with tempfile.TemporaryDirectory() as tmp:
            cls = FullMangledSeries_Class(
                sn, t0, mode="extrapolate_spectra", output_dir=out_root, verbose=False,
                prepare_output_dir=False,
            )
            cls.save_plot_path = tmp
            cls.get_mangledspec_list()
            fname = sorted(cls.mangledspec_list)[0]
            spec = cls.load_mangledfile(fname)
            mjd = mangled_filename_to_mjd(fname)
            phase = max(mjd - t0, 1e-5)
            log_phase = float(np.log10(phase))

            wls_lin = cls._spec_wls_linear(spec)
            log_wls = np.log10(wls_lin)
            flux = np.asarray(spec["flux"], dtype=float)
            ferr = np.asarray(spec["fluxerr"], dtype=float)
            # Subsample pixels for speed
            idx = np.linspace(0, flux.size - 1, min(80, flux.size), dtype=int)
            log_wls = log_wls[idx]
            flux = flux[idx]
            ferr = ferr[idx]

            norm1 = float(np.ptp(log_wls)) or 1.0
            norm2 = 2.0
            offset2 = log_phase - 0.5
            x1 = log_wls / norm1
            x2 = np.full_like(x1, (log_phase - offset2) / norm2)
            X = np.column_stack([x1, x2])
            y = flux
            yerr = np.maximum(ferr, 1e-3)

            wg = np.linspace(float(np.min(x1)), float(np.max(x1)), 8)
            ph = np.linspace(float(np.min(x2)), float(np.max(x2)), 4)
            fill = []
            for a in wg:
                for b in ph:
                    fill.append([a, b])
            X_fill = np.asarray(fill, dtype=float)

            kw = dict(pc.GP_INFERENCE_KWARGS)
            kw.update(optimize=False, mean="none", predict_chunk=500, predict_train=False)

            out = run_gp_from_bundle(
                dict(
                    X=X,
                    y=y,
                    yerr=yerr,
                    X_fill=X_fill,
                    kernel_wls_scale=np.float64(0.01),
                    kernel_time_scale=np.float64(0.01),
                    y_var_scale=np.float64(np.var(y)),
                    prior_points=np.zeros((0, 2)),
                    prior_values=np.zeros(0),
                ),
                cache_workdir=tmp,
                **kw,
            )
        self.assertEqual(out["mu"].shape[0], X_fill.shape[0])
        self.assertTrue(np.all(np.isfinite(out["mu"])))


class TestPipelineConfigGpInference(unittest.TestCase):
    def test_gp_inference_kwargs_additive_v5(self):
        import pipeline_config as pc

        kw = pc.GP_INFERENCE_KWARGS
        self.assertTrue(kw.get("additive_time"))
        self.assertTrue(kw.get("additive_wls"))
        self.assertEqual(kw.get("kernel_time"), "matern52")
        self.assertEqual(kw.get("kernel_wls"), "matern52")
        self.assertIn("optimize", kw)
        self.assertIn("GP_PRIOR_CACHE_SUBDIR", dir(pc))


if __name__ == "__main__":
    unittest.main()
