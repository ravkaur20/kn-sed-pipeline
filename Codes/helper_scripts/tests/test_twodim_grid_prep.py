import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np


from mangle_spectra_log import _resolve_filter_path, save_mangled_spectrum
from twodim_grid_prep import FullMangledSeries_Class


def _write_mangled_epoch(td, mjd, n=20):
    wls = np.linspace(4000, 8000, n)
    log_f = np.full(n, -15.0)
    log_e = np.full(n, 0.05)
    mask = np.zeros(n)
    path = os.path.join(td, "%s_mangled_spec.txt" % mjd)
    save_mangled_spectrum(path, wls, log_f, log_e, mask)
    return path


class TestGridAllSpectraltimeseries(unittest.TestCase):
    def test_runs_without_plt_when_plot_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            _write_mangled_epoch(td, "57983.969000")
            _write_mangled_epoch(td, "57984.969003")

            t0 = 57982.52851852
            spec_class = FullMangledSeries_Class(
                "TESTSN",
                t0,
                mode="extrapolate_spectra",
                output_dir=td + os.sep,
                mangled_spectra_dir=td,
                prepare_output_dir=False,
            )
            spec_class.plot_grid_rebin = False

            grid_wls, grid_mjd, grid_all, grid_all_err = spec_class.grid_all_spectraltimeseries()

            self.assertIsNotNone(spec_class.grids)
            self.assertEqual(len(spec_class.grids), 4)
            self.assertEqual(len(grid_wls), grid_all.shape[0])
            self.assertEqual(grid_all.shape[1], 2)
            self.assertEqual(grid_all_err.shape, grid_all.shape)
            self.assertEqual(len(grid_mjd), 2)

    def test_plot_flag_writes_pdf(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")

        with tempfile.TemporaryDirectory() as td:
            _write_mangled_epoch(td, "57983.969000")

            t0 = 57982.52851852
            spec_class = FullMangledSeries_Class(
                "TESTSN",
                t0,
                mode="extrapolate_spectra",
                output_dir=td + os.sep,
                mangled_spectra_dir=td,
                prepare_output_dir=False,
            )
            spec_class.plot_grid_rebin = True
            spec_class.save_plot_path = td

            spec_class.grid_all_spectraltimeseries()

            out_pdf = os.path.join(td, "grid_rebin_TESTSN.pdf")
            self.assertTrue(os.path.isfile(out_pdf))
            self.assertGreater(os.path.getsize(out_pdf), 0)


class TestBandFluxModified(unittest.TestCase):
    def test_band_flux_modified_no_csp_global(self):
        import pipeline_config as pc

        filt_dir = os.path.join(pc.COCO_PATH, "Inputs", "Filters")
        filt_path = os.path.join(filt_dir, "GeneralFilters", "EFOSC2_V.dat")
        if not os.path.isfile(filt_path):
            self.skipTest("EFOSC2_V filter missing")

        with tempfile.TemporaryDirectory() as td:
            fname = "57983.969000_mangled_spec.txt"
            _write_mangled_epoch(td, "57983.969000")

            spec_class = FullMangledSeries_Class(
                "AT2017gfo",
                57982.52851852,
                mode="extrapolate_spectra",
                output_dir=td + os.sep,
                mangled_spectra_dir=td,
                filters_dir=filt_dir + os.sep,
                prepare_output_dir=False,
                csp_sne=(),
            )
            log_lam, log_flux = spec_class.band_flux_modified("EFOSC2_V", fname)
            self.assertTrue(np.isfinite(log_lam))
            self.assertTrue(np.isfinite(log_flux))

    def test_csp_sne_selects_site3_path(self):
        with tempfile.TemporaryDirectory() as td:
            site3 = os.path.join(td, "Site3_CSP")
            os.makedirs(site3)
            wls = np.linspace(4000, 8000, 50)
            trans = np.exp(-0.5 * ((wls - 5500) / 500) ** 2)
            np.savetxt(os.path.join(site3, "sdss_g.txt"), np.column_stack([wls, trans]))

            mangled_dir = os.path.join(td, "mangled")
            os.makedirs(mangled_dir)
            fname = "57983.969000_mangled_spec.txt"
            _write_mangled_epoch(mangled_dir, "57983.969000")

            spec_class = FullMangledSeries_Class(
                "SN2004fe",
                57982.52851852,
                mode="extrapolate_spectra",
                output_dir=td + os.sep,
                mangled_spectra_dir=mangled_dir,
                filters_dir=td + os.sep,
                prepare_output_dir=False,
                csp_sne=("SN2004fe",),
            )
            expected = _resolve_filter_path(
                td, "sdss_g", "SN2004fe", ("SN2004fe",)
            )
            self.assertTrue(os.path.isfile(expected))
            filt = spec_class._load_filter_transmission("sdss_g")
            self.assertEqual(len(filt), 50)
            log_lam, log_flux = spec_class.band_flux_modified("sdss_g", fname)
            self.assertTrue(np.isfinite(log_lam))
            self.assertTrue(np.isfinite(log_flux))


if __name__ == "__main__":
    unittest.main()
