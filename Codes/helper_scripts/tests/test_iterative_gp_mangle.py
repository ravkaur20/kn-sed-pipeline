import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np


from iterative_gp_mangle import (
    compare_masks,
    load_masks_from_mangled_dir,
    spec_entries_with_mangled_files,
)
from mangle_spectra_log import save_mangled_spectrum
from spectra_pre_scale import SpectrumEntry


class TestIterativeGpMangleHelpers(unittest.TestCase):
    def test_interpolate_mask_apply_mismatch_lengths(self):
        from gp_surface_extract import interpolate_mangling_mask_to_wls

        gp_wls = np.linspace(3500, 12000, 50)
        pre_wls = np.linspace(5500, 9200, 30)
        mask_gp = np.linspace(0.0, 0.05, 50)
        on_pre = interpolate_mangling_mask_to_wls(mask_gp, gp_wls, pre_wls)
        self.assertEqual(on_pre.size, pre_wls.size)

    def test_load_masks_from_mangled_dir(self):
        with tempfile.TemporaryDirectory() as td:
            wls = np.linspace(4000, 8000, 10)
            log_f = np.full(10, -15.0)
            log_e = np.full(10, 0.05)
            mask = np.linspace(0.0, 0.1, 10)
            path = os.path.join(td, "57983.969000_mangled_spec.txt")
            save_mangled_spectrum(path, wls, log_f, log_e, mask)
            loaded = load_masks_from_mangled_dir(td)
            self.assertEqual(len(loaded), 1)
            info = next(iter(loaded.values()))
            np.testing.assert_allclose(info["mask"], mask)
            np.testing.assert_allclose(info["mask_wls"], wls)

    def test_mask_lookup_exact_mjd_suffix(self):
        from iterative_gp_mangle import _mask_info_for_mjd, load_masks_from_mangled_dir

        with tempfile.TemporaryDirectory() as td:
            for mjd, n in [(57987.980001, 5), (57987.980002, 3)]:
                wls = np.linspace(4000, 5000, n)
                save_mangled_spectrum(
                    os.path.join(td, "%.6f_mangled_spec.txt" % mjd),
                    wls,
                    np.full(n, -15.0),
                    np.full(n, 0.05),
                    np.zeros(n),
                )
            by_file = load_masks_from_mangled_dir(td)
            info = _mask_info_for_mjd(by_file, 57987.980002)
            self.assertIsNotNone(info)
            self.assertEqual(info["mask"].size, 3)

    def test_spec_entries_with_mangled_files_filters_missing(self):
        from iterative_gp_mangle import _mask_info_for_mjd, load_masks_from_mangled_dir

        with tempfile.TemporaryDirectory() as td:
            wls = np.linspace(4000, 5000, 5)
            save_mangled_spectrum(
                os.path.join(td, "57983.969000_mangled_spec.txt"),
                wls,
                np.full(5, -15.0),
                np.full(5, 0.05),
                np.zeros(5),
            )
            by_file = load_masks_from_mangled_dir(td)
            entries = [
                SpectrumEntry(57983.969000, 1.0, "/p/a.dat", "a.dat"),
                SpectrumEntry(57983.027880, 0.5, "/p/b.dat", "b.dat"),
            ]
            filtered = spec_entries_with_mangled_files(entries, by_file)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].basename, "a.dat")
            self.assertIsNone(_mask_info_for_mjd(by_file, 57983.027880))

    def test_compare_masks(self):
        old = {"a.dat": np.array([0.0, 0.1, 0.2])}
        new = {"a.dat": np.array([0.0, 0.11, 0.19])}
        m = compare_masks(old, new)
        self.assertEqual(m["n_compared"], 1)
        self.assertGreater(m["delta_mask_rms"], 0.0)


class TestIterativeDriverSmoke(unittest.TestCase):
    def test_photometry_target_gp_fit_enforced(self):
        import pipeline_config as pc
        from iterative_gp_mangle import run_iterative_gp_mangle

        with self.assertRaises(ValueError):
            run_iterative_gp_mangle(
                "SN",
                coco_path=pc.COCO_PATH,
                output_dir=pc.outputs_root(),
                t0_fix=57982.0,
                mode="extrapolate_spectra",
                max_iters=1,
                photometry_target="raw_observed",
            )

    def test_driver_one_iter_stub_gp(self):
        import pipeline_config as pc
        from iterative_gp_mangle import run_iterative_gp_mangle

        out_root = pc.outputs_root()
        sn = pc.SNNAME_DEFAULT
        mangled_dir = os.path.join(out_root, sn, "mangled_spectra")
        if not os.path.isdir(mangled_dir):
            self.skipTest("AT2017gfo mangled_spectra missing (run NB5 first)")

        def _stub(*_args, **_kwargs):
            from spectra_pre_scale import load_spec_list
            from pipeline_config import spec_list_path_for_mangling

            t0 = float(_kwargs.get("t0_fix", pc.SN_EXPLOSION_MJD[sn]))
            entries = load_spec_list(spec_list_path_for_mangling(pc.COCO_PATH, sn))
            phases = []
            for e in entries:
                ph = max(float(e.mjd) - t0, 1e-5)
                phases.append(float(np.log10(ph)))
            if not phases:
                phases = [-2.0, -1.5]
            phases_arr = np.asarray(phases, dtype=float)
            n_phase = len(phases_arr)
            n_wl = 3
            n = n_wl * n_phase
            gn = {
                "offset": -5.0,
                "scale_factor": 1.0,
                "norm1": 4.0,
                "norm2": 2.0,
                "offset2": float(np.min(phases_arr) - 0.1),
            }
            x2 = []
            x1 = []
            for ph in phases_arr:
                x2_norm = (ph - gn["offset2"]) / gn["norm2"]
                for wl in np.linspace(0.85, 0.9, n_wl):
                    x1.append(wl)
                    x2.append(x2_norm)
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            mu = np.zeros(n)
            gp_runs = _kwargs.get("gp_runs_dir", "/tmp")
            full_gp = os.path.join(gp_runs, "full_gp")
            os.makedirs(full_gp, exist_ok=True)
            with open(os.path.join(full_gp, "-1.500000_spec_extended_FL.txt"), "w", encoding="utf-8") as fh:
                fh.write("#wls\tflux\tfluxerr\n4.000000E+03\t1.000000E-15\t1.000000E-16\n")
            return {
                "predictions_path": os.path.join(gp_runs, "predictions.npz"),
                "mu": mu,
                "mu_raw": mu,
                "std": np.ones(n) * 0.01,
                "x1_fill": x1,
                "x2_fill": x2,
                "wls_log_grid": np.linspace(3.5, 3.7, n_wl),
                "phase_log10_columns": phases_arr,
                "grid_norm_info": gn,
                "spec_class": type("D", (), {"grids": (np.linspace(3.5, 3.7, n_wl), [])})(),
                "grid_ext_columns": phases_arr,
                "exported_spectra_paths": [os.path.join(full_gp, "-1.500000_spec_extended_FL.txt")],
                "gp_runs_dir": gp_runs,
            }

        import iterative_gp_mangle as igm

        saved = igm.run_iter_gp_fit
        try:
            igm.run_iter_gp_fit = _stub  # type: ignore
            summary = run_iterative_gp_mangle(
                sn,
                coco_path=pc.COCO_PATH,
                output_dir=out_root,
                t0_fix=pc.SN_EXPLOSION_MJD[sn],
                mode="extrapolate_spectra",
                max_iters=1,
                seed_from_nb5=True,
                nb5_mangled_dir=mangled_dir,
                save_diagnostics=False,
                verbose=False,
            )
            self.assertEqual(len(summary["iterations"]), 1)
            final_gp = os.path.join(summary["final_dir"], "full_gp")
            if os.path.isdir(final_gp):
                has_txt = any(fn.endswith(".txt") for fn in os.listdir(final_gp))
                has_npz = os.path.isfile(os.path.join(final_gp, "predictions.npz"))
                self.assertTrue(has_txt or has_npz)
        finally:
            igm.run_iter_gp_fit = saved


if __name__ == "__main__":
    unittest.main()
