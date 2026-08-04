import _bootstrap_paths  # noqa: F401
import unittest

import numpy as np


from gp_surface_extract import (
    demangle_extracted_spectrum,
    extract_gp_spectrum_at_epoch,
    grid_norm_uses_zscore,
    scaled_mu_to_log10_flux,
)


class TestGpSurfaceExtract(unittest.TestCase):
    def test_grid_norm_zscore_detection(self):
        self.assertTrue(
            grid_norm_uses_zscore({"coord_parametrization": "zscore", "x2_mean": 0.0})
        )
        self.assertFalse(grid_norm_uses_zscore({"norm1": 1.0, "norm2": 1.0, "offset2": 0.0}))

    def test_extract_at_epoch_minmax(self):
        gn = {
            "offset": -5.0,
            "scale_factor": 1.0,
            "norm1": 4.0,
            "norm2": 2.0,
            "offset2": -3.0,
        }
        wls_log = np.array([3.5, 3.6, 3.7])
        phases = np.array([-2.0, -1.5])
        x1_list, x2_list, mu_list = [], [], []
        for wl in wls_log / gn["norm1"]:
            for ph in (phases - gn["offset2"]) / gn["norm2"]:
                x1_list.append(wl)
                x2_list.append(ph)
                mu_list.append(0.0)
        x1 = np.asarray(x1_list)
        x2 = np.asarray(x2_list)
        mu = np.asarray(mu_list)
        std = np.full_like(mu, 0.01)
        target_wls = np.power(10.0, wls_log)
        out = extract_gp_spectrum_at_epoch(
            mu,
            std,
            x1,
            x2,
            gn,
            float(phases[0]),
            target_wls,
            wls_log_grid=wls_log,
            phase_log10_columns=phases,
        )
        self.assertEqual(out["log10_flux"].shape, target_wls.shape)
        self.assertTrue(np.all(np.isfinite(out["log10_flux"])))

    def test_demangle_roundtrip(self):
        log_f = np.array([1.0, 1.1, 1.2])
        mask = np.array([0.05, 0.05, 0.05])
        dem = demangle_extracted_spectrum(log_f, mask)
        np.testing.assert_allclose(dem + mask, log_f)

    def test_scaled_mu_to_log10(self):
        gn = {"offset": -10.0, "scale_factor": 2.0}
        out = scaled_mu_to_log10_flux(np.array([0.0, 1.0]), gn)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_interpolate_mangling_mask(self):
        from gp_surface_extract import interpolate_mangling_mask_to_wls

        src_w = np.array([4000.0, 5000.0, 6000.0])
        dst_w = np.array([4500.0, 5500.0])
        mask = np.array([0.0, 0.1, 0.2])
        out = interpolate_mangling_mask_to_wls(mask, src_w, dst_w)
        self.assertEqual(out.shape, dst_w.shape)
        self.assertGreater(out[1], out[0])

    def test_extract_gp_wavelength_grid(self):
        from gp_surface_extract import extract_all_observed_epochs, gp_prediction_wls_linear

        gn = {
            "offset": -5.0,
            "scale_factor": 1.0,
            "norm1": 4.0,
            "norm2": 2.0,
            "offset2": -3.0,
        }
        wls_log = np.array([3.5, 3.6, 3.7])
        phases = np.array([-2.0])
        x1_list, x2_list, mu_list = [], [], []
        for wl in wls_log / gn["norm1"]:
            ph = (phases[0] - gn["offset2"]) / gn["norm2"]
            x1_list.append(wl)
            x2_list.append(ph)
            mu_list.append(0.0)
        predictions = {
            "mu": np.asarray(mu_list),
            "std": np.full(len(mu_list), 0.01),
            "x1_fill": np.asarray(x1_list),
            "x2_fill": np.asarray(x2_list),
            "grid_norm_info": gn,
            "wls_log_grid": wls_log,
            "phase_log10_columns": phases,
        }

        class E:
            def __init__(self, bn, mjd):
                self.basename = bn
                self.mjd = mjd

        prescaled = {"a.dat": np.array([5000.0, 6000.0])}
        out = extract_all_observed_epochs(
            predictions,
            spec_entries=[E("a.dat", 57982.01)],
            t0_fix=57982.0,
            prescaled_wls_by_basename=prescaled,
            wavelength_grid="gp",
        )
        self.assertEqual(len(out), 1)
        gp_wls = gp_prediction_wls_linear(wls_log)
        self.assertEqual(out[0]["extracted"]["log10_flux"].size, gp_wls.size)


if __name__ == "__main__":
    unittest.main()
