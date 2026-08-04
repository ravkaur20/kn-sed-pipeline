import _bootstrap_paths  # noqa: F401
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from mangle_spectra_log import (
    apply_mangling_mask_linear,
    demangle_log_spectrum,
    load_mangled_spectrum,
    run_mangle_pipeline,
    save_mangled_spectrum,
)
from spectra_pre_scale import SpectrumEntry, write_spec_list



class TestMangleSpectraLog(unittest.TestCase):
    def test_demangle_roundtrip(self):
        wls = np.linspace(4000, 8000, 50)
        flux = np.exp(-0.5 * ((wls - 6000) / 500) ** 2) * 1e-15
        ferr = flux * 0.1
        mask = 0.05 * np.sin(wls / 1000.0)
        _, log_f, log_e = apply_mangling_mask_linear(wls, flux, ferr, mask)
        back = demangle_log_spectrum(log_f, mask)
        np.testing.assert_allclose(back, np.log10(flux), rtol=1e-10)

    def test_four_column_io(self):
        wls = np.array([4000.0, 5000.0, 6000.0])
        log_f = np.array([-15.0, -14.5, -14.0])
        log_e = np.array([0.05, 0.05, 0.05])
        mask = np.array([0.01, 0.02, 0.01])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "spec.txt")
            save_mangled_spectrum(path, wls, log_f, log_e, mask)
            spec, m = load_mangled_spectrum(path)
            self.assertIsNotNone(m)
            np.testing.assert_allclose(spec["wls"], wls)
            np.testing.assert_allclose(spec["flux"], log_f)
            np.testing.assert_allclose(m, mask)


class TestManglePipelineFailures(unittest.TestCase):
    def test_run_mangle_pipeline_records_failed_entries(self):
        import pipeline_config as pconf

        with tempfile.TemporaryDirectory() as td:
            coco = td + os.sep
            out_root = os.path.join(td, "Outputs") + os.sep
            sn = "TESTSN"
            sn_dir = os.path.join(out_root, sn)
            os.makedirs(sn_dir, exist_ok=True)

            list_dir = os.path.join(coco, "Inputs", "Spectroscopy", "2_spec_lists_prescaled")
            prescaled_dir = os.path.join(coco, "Inputs", "Spectroscopy", "2_spec_prescaled", sn)
            os.makedirs(list_dir, exist_ok=True)
            os.makedirs(prescaled_dir, exist_ok=True)

            spec_path = os.path.join(prescaled_dir, "epoch.dat")
            wls = np.linspace(4000, 8000, 20)
            with open(spec_path, "w", encoding="utf-8") as fh:
                fh.write("# wls flux fluxerr\n")
                for w in wls:
                    fh.write("%E\t%E\t%E\n" % (w, 1e-15, 1e-16))

            list_path = os.path.join(list_dir, "%s.list" % sn)
            write_spec_list(
                list_path,
                [SpectrumEntry(57983.027880, 0.5, spec_path, "epoch.dat")],
            )

            with open(os.path.join(sn_dir, "fitted_phot4mangling_%s.dat" % sn), "w") as fh:
                fh.write("spec_mjd\tBessell_V_fit_log_flux\tBessell_V_fit_log_fluxerr\tBessell_V_inrange\n")
                fh.write("57983.027880\t-14.0\t0.05\t1\n")

            mjd_json = pconf.band_mjd_ranges_json_path(out_root, sn)
            os.makedirs(os.path.dirname(mjd_json), exist_ok=True)
            with open(mjd_json, "w", encoding="utf-8") as fh:
                json.dump({"Bessell_V": {"min_mjd": 57980.0, "max_mjd": 57990.0}}, fh)

            with mock.patch("mangle_spectra_log.compute_mangling_mask", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    summary = run_mangle_pipeline(
                        sn,
                        coco_path=coco,
                        output_dir=out_root,
                        bundle_aware=False,
                        save_diagnostics=False,
                    )

            self.assertEqual(summary["report"]["n_failed"], 1)
            self.assertEqual(len(summary["report"]["failed"]), 1)
            fail = summary["report"]["failed"][0]
            self.assertEqual(fail["basename"], "epoch.dat")
            self.assertAlmostEqual(fail["mjd"], 57983.027880)
            self.assertEqual(fail["reason"], "no_phot_constraints")
            self.assertIn("FAILED", buf.getvalue())
            self.assertIn("57983.027880", buf.getvalue())

    def test_run_mangle_pipeline_writes_band_mjd_json_when_missing(self):
        import pipeline_config as pconf
        import photometry_filter_utils as pfu

        with tempfile.TemporaryDirectory() as td:
            coco = td + os.sep
            out_root = os.path.join(td, "Outputs") + os.sep
            sn = "TESTSN"
            sn_dir = os.path.join(out_root, sn)
            os.makedirs(sn_dir, exist_ok=True)

            raw_dir = os.path.join(coco, "Inputs", "Photometry", "1_LCs_flux_raw")
            os.makedirs(raw_dir, exist_ok=True)
            with open(os.path.join(raw_dir, "%s.dat" % sn), "w", encoding="utf-8") as fh:
                fh.write("MJD,flux,band\n")
                fh.write("57980.0,0,Bessell_V\n")
                fh.write("57990.0,0,Bessell_V\n")

            list_dir = os.path.join(coco, "Inputs", "Spectroscopy", "2_spec_lists_prescaled")
            prescaled_dir = os.path.join(coco, "Inputs", "Spectroscopy", "2_spec_prescaled", sn)
            os.makedirs(list_dir, exist_ok=True)
            os.makedirs(prescaled_dir, exist_ok=True)

            spec_path = os.path.join(prescaled_dir, "epoch.dat")
            wls = np.linspace(4000, 8000, 20)
            with open(spec_path, "w", encoding="utf-8") as fh:
                fh.write("# wls flux fluxerr\n")
                for w in wls:
                    fh.write("%E\t%E\t%E\n" % (w, 1e-15, 1e-16))

            list_path = os.path.join(list_dir, "%s.list" % sn)
            write_spec_list(
                list_path,
                [SpectrumEntry(57983.027880, 0.5, spec_path, "epoch.dat")],
            )

            with open(os.path.join(sn_dir, "fitted_phot4mangling_%s.dat" % sn), "w") as fh:
                fh.write("spec_mjd\tBessell_V_fit_log_flux\tBessell_V_fit_log_fluxerr\tBessell_V_inrange\n")
                fh.write("57983.027880\t-14.0\t0.05\t1\n")

            mjd_json = pconf.band_mjd_ranges_json_path(out_root, sn)
            self.assertFalse(os.path.isfile(mjd_json))

            with mock.patch("mangle_spectra_log.compute_mangling_mask", return_value=None):
                run_mangle_pipeline(
                    sn,
                    coco_path=coco,
                    output_dir=out_root,
                    bundle_aware=False,
                    save_diagnostics=False,
                )

            self.assertTrue(os.path.isfile(mjd_json))
            loaded = pfu.load_band_mjd_ranges_json(mjd_json)
            self.assertAlmostEqual(loaded["Bessell_V"]["min_mjd"], 57980.0)
            self.assertAlmostEqual(loaded["Bessell_V"]["max_mjd"], 57990.0)


if __name__ == "__main__":
    unittest.main()
