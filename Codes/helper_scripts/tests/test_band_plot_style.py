"""Tests for shared band plotting styles."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import _bootstrap_paths  # noqa: F401

import band_plot_style as bps
import pipeline_config as pc


class TestBandPlotStyle(unittest.TestCase):
    def test_color_dict_nonempty(self):
        bps.refresh_from_config()
        self.assertGreater(len(bps.COLOR_DICT), 0)
        for k, v in bps.COLOR_DICT.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)
            self.assertTrue(v)

    def test_mark_dict_nonempty(self):
        bps.refresh_from_config()
        self.assertGreater(len(bps.MARK_DICT), 0)

    def test_band_color_known_band(self):
        bps.refresh_from_config()
        self.assertEqual(bps.band_color("Swope_V"), "green")

    def test_band_marker_default(self):
        self.assertEqual(bps.band_marker("UnknownBand_xyz"), "o")

    def test_lowercase_aliases(self):
        self.assertIs(bps.color_dict, bps.COLOR_DICT)
        self.assertIs(bps.mark_dict, bps.MARK_DICT)
        self.assertIs(bps.exclude_filt, bps.EXCLUDE_FILT)

    def test_load_filter_plot_style_from_json(self):
        cfg = {
            "exclude": ["BadBand"],
            "active_filters": ["Swope_V"],
            "bands": {"Swope_V": {"color": "cyan", "marker": "s"}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(cfg, fh)
            path = fh.name
        try:
            style = pc.load_filter_plot_style(path)
            self.assertEqual(style.color_dict["Swope_V"], "cyan")
            self.assertEqual(style.mark_dict["Swope_V"], "s")
            self.assertEqual(style.exclude_filt, ["BadBand"])
            self.assertEqual(style.active_filters, ["Swope_V"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
