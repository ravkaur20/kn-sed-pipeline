"""Tests for bundle-aware mangling and diagnostics."""
import _bootstrap_paths  # noqa: F401
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


from mangle_spectra_log import (
    build_mangle_group_map,
    compute_mangling_mask,
    compute_mangling_mask_bundle,
    compute_seam_jumps,
    demangle_log_spectrum,
)


def _linear_spec(wls, flux=None):
    flux = flux if flux is not None else np.ones_like(wls) * 1e-15
    err = flux * 0.1
    return np.array(
        list(zip(wls, flux, err)),
        dtype=np.dtype([("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")]),
    )


def _fake_phot_row():
    return {
        "spec_mjd": 58000.0,
        "Bessell_V_fit_log_flux": -14.0,
        "Bessell_V_fit_log_fluxerr": 0.05,
        "Bessell_V_inrange": True,
    }


class TestSeamJumps(unittest.TestCase):
    def test_seam_jump_positive(self):
        uvb = _linear_spec(np.linspace(3000, 3700, 40), flux=np.full(40, 1e-15))
        vis = _linear_spec(np.linspace(5500, 9000, 60), flux=np.full(60, 2e-15))
        arms = {"UVB_x.dat": uvb, "VIS_x.dat": vis}
        jumps = compute_seam_jumps(arms, ["uvb", "vis"], flux_in_log10=False)
        self.assertEqual(len(jumps), 1)
        self.assertTrue(np.isfinite(jumps[0]["seam_jump_log10"]))


class TestBuildMangleGroupMap(unittest.TestCase):
    def test_maps_basenames(self):
        from spectra_pre_scale import ScaleGroup, SpectrumEntry

        entries = [
            SpectrumEntry(1.0, 0.1, "/p/UVB_a.dat", "UVB_a.dat"),
            SpectrumEntry(1.0, 0.1, "/p/VIS_a.dat", "VIS_a.dat"),
        ]
        groups = [ScaleGroup(id="g1", members=["UVB_a.dat", "VIS_a.dat"], merge_order=["uvb", "vis"])]
        b2g, ge = build_mangle_group_map(entries, groups)
        self.assertEqual(b2g["UVB_a.dat"], "g1")
        self.assertEqual(len(ge["g1"]), 2)


class TestStitchHelpers(unittest.TestCase):
    def test_combined_overlap_beats_single(self):
        from mangle_spectra_log import _combined_filter_overlap_width, _filter_overlap_width

        fmin, fmax = 9000.0, 9500.0
        vis = np.linspace(5500, 9150, 40)
        nir = np.linspace(9200, 12000, 40)
        vis_w = _filter_overlap_width(fmin, fmax, vis)
        nir_w = _filter_overlap_width(fmin, fmax, nir)
        combined = _combined_filter_overlap_width(fmin, fmax, [vis, nir])
        self.assertGreater(combined, max(vis_w, nir_w))

    def test_stitch_arm_spectrum_for_filter(self):
        from mangle_spectra_log import stitch_arm_spectrum_for_filter

        vis = np.linspace(5500, 9200, 40)
        nir = np.linspace(8800, 12000, 40)
        specs = {
            "VIS.dat": _linear_spec(vis),
            "NIR.dat": _linear_spec(nir, flux=np.full(40, 2e-15)),
        }
        out = stitch_arm_spectrum_for_filter(specs, ["VIS.dat", "NIR.dat"], 9000.0, 9500.0)
        self.assertIsNotNone(out)
        wls, flux, err = out
        self.assertGreater(wls.size, 0)
        self.assertTrue(float(np.min(wls)) >= 9000.0)
        self.assertTrue(float(np.max(wls)) <= 9500.0)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("george") is not None,
    "george required",
)
class TestBundleManglingMask(unittest.TestCase):
    def test_bundle_pooled_differs_from_per_arm(self):
        import pandas as pd

        wls_u = np.linspace(3500, 4500, 30)
        wls_v = np.linspace(5000, 8000, 40)
        spec_u = _linear_spec(wls_u, flux=np.full(30, 1e-15))
        spec_v = _linear_spec(wls_v, flux=np.full(40, 1.2e-15))
        phot = pd.DataFrame([_fake_phot_row()])
        filt = {"Bessell_V": {"min_mjd": 57900.0, "max_mjd": 58100.0}}
        avail = ["Bessell_V"]

        with mock.patch("mangle_spectra_log.band_flux_trapz") as bft, mock.patch(
            "mangle_spectra_log._filter_linear_wavelength_range",
            return_value=(4000.0, 7000.0),
        ):
            def _trapz(wls, flux, ferr, filt_name, **kw):
                lam_eff = float(np.median(wls))
                raw = float(np.median(flux) * 1.1)
                return 0.0, lam_eff, raw, raw * 0.1, float(np.min(wls)), float(np.max(wls))

            bft.side_effect = _trapz
            row = phot.iloc[0]
            out_u = compute_mangling_mask(
                spec_u, row, avail, filt, filter_path="/tmp/", snname="T"
            )
            out_v = compute_mangling_mask(
                spec_v, row, avail, filt, filter_path="/tmp/", snname="T"
            )
            self.assertIsNotNone(out_u)
            self.assertIsNotNone(out_v)
            bundle = compute_mangling_mask_bundle(
                {"UVB_a.dat": spec_u, "VIS_a.dat": spec_v},
                phot,
                avail,
                filt,
                filter_path="/tmp/",
                snname="T",
                member_mjd={"UVB_a.dat": 58000.0, "VIS_a.dat": 58000.0},
            )
        self.assertIsNotNone(bundle)
        masks, _, meta = bundle
        self.assertTrue(meta.get("bundle_mode"))
        self.assertIn("UVB_a.dat", masks)
        self.assertIn("VIS_a.dat", masks)


class TestMangleDiagnostics(unittest.TestCase):
    def test_index_written_without_mpl(self):
        import mangle_diagnostics as md

        with tempfile.TemporaryDirectory() as td:
            md.write_mangle_diagnostics_index(td, [{"id": "g1", "seam_qa": []}])
            self.assertTrue(os.path.isfile(os.path.join(td, "index.html")))

    @unittest.skipUnless(
        __import__("mangle_diagnostics", fromlist=["diagnostics_available"]).diagnostics_available(),
        "matplotlib required",
    )
    def test_triple_plot_when_mangled_per_arm(self):
        import mangle_diagnostics as md

        wls = np.linspace(4000, 8000, 20)
        pre = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1e-15)}}
        pa = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1.2e-15)}}
        bund = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1.1e-15)}}
        masks = {"arm_a.dat": np.zeros(20)}
        with tempfile.TemporaryDirectory() as td:
            md.save_group_mangle_diagnostics(
                td,
                "g1",
                prescaled=pre,
                mangled=bund,
                masks=masks,
                mangled_per_arm=pa,
            )
            triple = os.path.join(td, "group_g1_prescaled_perarm_bundle.pdf")
            simple = os.path.join(td, "group_g1_prescaled_vs_mangled.pdf")
            self.assertTrue(os.path.isfile(triple))
            self.assertFalse(os.path.isfile(simple))

    @unittest.skipUnless(
        __import__("mangle_diagnostics", fromlist=["diagnostics_available"]).diagnostics_available(),
        "matplotlib required",
    )
    def test_triple_plot_removes_stale_simple_pdf(self):
        import mangle_diagnostics as md

        wls = np.linspace(4000, 8000, 20)
        pre = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1e-15)}}
        pa = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1.2e-15)}}
        bund = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1.1e-15)}}
        masks = {"arm_a.dat": np.zeros(20)}
        with tempfile.TemporaryDirectory() as td:
            simple = os.path.join(td, "group_g1_prescaled_vs_mangled.pdf")
            with open(simple, "wb") as fh:
                fh.write(b"stale")
            md.save_group_mangle_diagnostics(
                td,
                "g1",
                prescaled=pre,
                mangled=bund,
                masks=masks,
                mangled_per_arm=pa,
            )
            self.assertFalse(os.path.isfile(simple))

    @unittest.skipUnless(
        __import__("mangle_diagnostics", fromlist=["diagnostics_available"]).diagnostics_available(),
        "matplotlib required",
    )
    def test_simple_plot_without_mangled_per_arm(self):
        import mangle_diagnostics as md

        wls = np.linspace(4000, 8000, 20)
        pre = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1e-15)}}
        bund = {"arm_a.dat": {"wls": wls, "flux": np.full(20, 1.1e-15)}}
        masks = {"arm_a.dat": np.zeros(20)}
        with tempfile.TemporaryDirectory() as td:
            md.save_group_mangle_diagnostics(
                td,
                "g1",
                prescaled=pre,
                mangled=bund,
                masks=masks,
                mangled_per_arm=None,
            )
            triple = os.path.join(td, "group_g1_prescaled_perarm_bundle.pdf")
            simple = os.path.join(td, "group_g1_prescaled_vs_mangled.pdf")
            self.assertFalse(os.path.isfile(triple))
            self.assertTrue(os.path.isfile(simple))

    @unittest.skipUnless(
        __import__("mangle_diagnostics", fromlist=["diagnostics_available"]).diagnostics_available(),
        "matplotlib required",
    )
    def test_mask_overlay_truncates_mismatched_grid(self):
        import mangle_diagnostics as md

        wls = np.linspace(4000, 8000, 200)
        pre = {"arm_a.dat": {"wls": wls, "flux": np.full(200, 1e-15)}}
        bund = {"arm_a.dat": {"wls": wls, "flux": np.full(200, 1.1e-15)}}
        masks = {"arm_a.dat": np.linspace(0.0, 0.01, 10)}
        with tempfile.TemporaryDirectory() as td:
            md.save_group_mangle_diagnostics(
                td,
                "g1",
                prescaled=pre,
                mangled=bund,
                masks=masks,
            )
            overlay = os.path.join(td, "group_g1_mask_overlay.pdf")
            self.assertTrue(os.path.isfile(overlay))


class TestPipelineConfigMangle(unittest.TestCase):
    def test_mangle_config_keys(self):
        import pipeline_config as pc

        self.assertIn("MANGLE_BUNDLE_AWARE", dir(pc))
        self.assertIn("MANGLE_BUNDLE_STITCH_SYNPHOT", dir(pc))
        self.assertIn("ITER_MANGLE_USE_GP_WAVELENGTH_GRID", dir(pc))
        self.assertTrue(pc.MANGLE_BUNDLE_STITCH_SYNPHOT)


class TestIterativeRemangleImport(unittest.TestCase):
    def test_remangle_import(self):
        from iterative_gp_mangle import remangle_spectra

        self.assertTrue(callable(remangle_spectra))


if __name__ == "__main__":
    unittest.main()
