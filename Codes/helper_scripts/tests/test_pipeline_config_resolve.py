import os
import unittest

import _bootstrap_paths  # noqa: F401

import comparison_check_log_utils as cc


class TestResolveFinalBranch(unittest.TestCase):
    def test_flat_as_observed_layout(self):
        coco = "/tmp/coco"
        d = cc.resolve_final_directory(coco, "SN", "as_observed")
        self.assertEqual(
            d,
            os.path.join("/tmp/coco", "Outputs", "SN", "FINAL_spectra_2dim", "as_observed"),
        )

    def test_legacy_twodim_branch_still_joins(self):
        coco = "/tmp/coco"
        d = cc.resolve_final_directory(
            coco, "SN", "as_observed", twodim_branch="extrapolate/spliced"
        )
        self.assertIn("extrapolate%s%s" % (os.sep, "spliced"), d)
        self.assertTrue(d.endswith(os.path.join("as_observed")))


class TestTwodimIterPaths(unittest.TestCase):
    def test_iter_path_helpers(self):
        import pipeline_config as pc

        root = pc.twodim_iter_root("/tmp/out", "SN", "extrapolate_spectra")
        self.assertIn("twodim_iter", root)
        self.assertNotIn("extrapolate", root)
        d0 = pc.twodim_iter_dir("/tmp/out", "SN", "extrapolate_spectra", 0)
        self.assertTrue(d0.endswith(os.path.join("iter_00")))
        final = pc.twodim_iter_final_dir("/tmp/out", "SN", "extrapolate_spectra")
        self.assertTrue(final.endswith("final"))
        figs = pc.iter_figs_dir("/tmp/out", "SN", 1, sub="gp_surface")
        self.assertIn(os.path.join("iter_01", "figs", "gp_surface"), figs)
        inf = pc.iter_inference_dir("/tmp/out", "SN", 0)
        self.assertIn(os.path.join("iter_00", "gp_runs", "inference"), inf)
        metrics = pc.twodim_iter_metrics_dir("/tmp/out", "SN")
        self.assertTrue(metrics.endswith(os.path.join("twodim_iter", "metrics")))
        nb5 = pc.mangle_diagnostics_dir("/tmp/out", "SN")
        self.assertIn(os.path.join("figs", "mangle_nb5"), nb5)

    def test_final_spectra_qa_dir(self):
        import pipeline_config as pc

        d = pc.final_spectra_qa_dir("/tmp/out", "SN")
        self.assertTrue(d.endswith(os.path.join("FINAL_spectra_2dim", "as_observed")))


class TestResolveIterGpDirectory(unittest.TestCase):
    def test_resolve_iter_gp_directory(self):
        d = cc.resolve_iter_gp_directory("/tmp/coco", "SN", "extrapolate_spectra")
        self.assertIn("twodim_iter", d)
        self.assertNotIn("extrapolate", d)
        self.assertTrue(d.endswith(os.path.join("final", "full_gp")))

    def test_parse_spec_extended_stem(self):
        self.assertAlmostEqual(
            cc.parse_spectrum_stem("0.842480_spec_extended.txt"), 0.842480
        )


if __name__ == "__main__":
    unittest.main()
