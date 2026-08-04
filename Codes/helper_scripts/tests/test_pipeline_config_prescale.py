import _bootstrap_paths  # noqa: F401
import os
import unittest

import pipeline_config as pconf


class TestPrescalePaths(unittest.TestCase):
    def test_prescaled_list_fallback(self):
        coco = pconf.COCO_PATH
        sn = "AT2017gfo"
        smooth = pconf.smoothed_spec_list_path(coco, sn)
        self.assertTrue(os.path.isfile(smooth))
        # prescaled may not exist yet; helper still returns path
        pre = pconf.prescaled_spec_list_path(coco, sn)
        self.assertIn("2_spec_lists_prescaled", pre)

    def test_spec_scale_paths(self):
        od = os.path.join(pconf.COCO_PATH, "Outputs")
        p = pconf.spec_scale_groups_json_path(od, "SN")
        self.assertIn("spec_scale_groups.json", p)

    def test_gap_max_config(self):
        self.assertEqual(pconf.SPEC_SCALE_GAP_MAX_A, 400.0)
        self.assertFalse(pconf.SPEC_SCALE_CHAIN_MODE)
        self.assertTrue(pconf.SPEC_SCALE_STAR_BRIDGE_FALLBACK)


if __name__ == "__main__":
    unittest.main()
