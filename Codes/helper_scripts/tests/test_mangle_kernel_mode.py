"""Regression tests for mangle GP kernel lengthscale mode."""
import _bootstrap_paths  # noqa: F401
import os
import unittest
from unittest import mock

import numpy as np

import pipeline_config as pconf
from mangle_spectra_log import (
    _fit_mangling_mask_on_grid,
    _mangle_gp_kernel_length,
    compute_mangling_mask_bundle,
    load_linear_spectrum,
    _avail_filters_from_phot4mangling,
)


def _optical_mask_std(mask: np.ndarray, wls: np.ndarray) -> float:
    opt = (wls > 4000.0) & (wls < 7000.0)
    return float(np.std(mask[opt]))


class TestMangleKernelLengthHelper(unittest.TestCase):
    def test_fixed_5_default(self):
        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "fixed_5"):
            with mock.patch.object(pconf, "MANGLE_GP_KERNEL_FIXED", 5.0):
                self.assertAlmostEqual(_mangle_gp_kernel_length(3.7, 800), 5.0)

    def test_kernel_divide_scaled(self):
        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "kernel_divide_scaled"):
            self.assertAlmostEqual(_mangle_gp_kernel_length(4.0, 800), 200.0)


class TestMangleKernelModeRegression(unittest.TestCase):
    def test_at2017gfo_group003_fixed_5_smoother_than_scaled(self):
        try:
            import george  # noqa: F401
            import pandas as pd
            from photometry_filter_utils import load_band_mjd_ranges_json
        except ImportError:
            self.skipTest("george/pandas not installed")

        sn = "AT2017gfo"
        coco = pconf.COCO_PATH.rstrip(os.sep) + os.sep
        out_root = pconf.outputs_root(coco) + os.sep
        phot_path = os.path.join(out_root, sn, "fitted_phot4mangling_%s.dat" % sn)
        mjd_json = pconf.band_mjd_ranges_json_path(out_root, sn)
        pre_dir = os.path.join(
            coco, "Inputs", "Spectroscopy", "2_spec_prescaled", sn
        )
        names = [
            "NIR_AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974001_Phase+3.41d.dat",
            "UVB_AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974002_Phase+3.41d.dat",
            "VIS_AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974003_Phase+3.41d.dat",
        ]
        if not all(os.path.isfile(os.path.join(pre_dir, n)) for n in names):
            self.skipTest("AT2017gfo prescaled fixture not present")
        if not os.path.isfile(phot_path) or not os.path.isfile(mjd_json):
            self.skipTest("AT2017gfo mangle outputs not present")

        phot = pd.read_csv(phot_path, sep="\t")
        filter_mjd_dict = load_band_mjd_ranges_json(mjd_json)
        avail_filters = _avail_filters_from_phot4mangling(phot)
        filt_root = pconf.filters_root(coco)
        specs = {n: load_linear_spectrum(os.path.join(pre_dir, n)) for n in names}
        mjds = {
            names[0]: 57985.974001,
            names[1]: 57985.974002,
            names[2]: 57985.974003,
        }
        kwargs = dict(
            phot4mangling=phot,
            avail_filters=avail_filters,
            filter_mjd_dict=filter_mjd_dict,
            filter_path=filt_root,
            snname=sn,
            member_mjd=mjds,
            merge_order=["uvb", "vis", "nir"],
            use_gp_wavelength_grid=False,
            kernel_divide=800,
        )

        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "fixed_5"):
            out_fixed = compute_mangling_mask_bundle(specs, **kwargs)
        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "kernel_divide_scaled"):
            out_scaled = compute_mangling_mask_bundle(specs, **kwargs)

        self.assertIsNotNone(out_fixed)
        self.assertIsNotNone(out_scaled)
        vis_name = names[2]
        wls = specs[vis_name]["wls"]
        std_fixed = _optical_mask_std(out_fixed[0][vis_name], wls)
        std_scaled = _optical_mask_std(out_scaled[0][vis_name], wls)
        self.assertLess(std_fixed, 0.05)
        self.assertGreater(std_scaled, 0.08)
        self.assertLess(std_fixed, std_scaled)


if __name__ == "__main__":
    unittest.main()
