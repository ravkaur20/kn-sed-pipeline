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

    def test_unknown_band_falls_back_with_one_warning(self):
        style = pc.load_filter_plot_style()
        with self.assertWarns(UserWarning):
            self.assertEqual(style.color_dict["UnknownBand_xyz"], bps.FALLBACK_COLOR)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # second lookup must not warn again
            self.assertEqual(style.color_dict["UnknownBand_xyz"], bps.FALLBACK_COLOR)
        with self.assertWarns(UserWarning):
            self.assertEqual(style.mark_dict["UnknownBand_xyz"], bps.FALLBACK_MARKER)
        self.assertNotIn("UnknownBand_xyz", style.color_dict)

    def test_bytes_band_key(self):
        style = pc.load_filter_plot_style()
        self.assertEqual(style.color_dict[b"Swope_V"], style.color_dict["Swope_V"])

    def test_runtime_dicts_keep_fallback(self):
        rt = pc.bootstrap_runtime()
        with self.assertWarns(UserWarning):
            self.assertEqual(rt.color_dict["UnknownBand_abc"], bps.FALLBACK_COLOR)
        with self.assertWarns(UserWarning):
            self.assertEqual(bps.COLOR_DICT["UnknownBand_abc"], bps.FALLBACK_COLOR)

    def test_every_shipped_raw_band_has_a_style(self):
        import pandas as pd

        raw = pd.read_csv(pc.raw_photometry_path(None, pc.SNNAME_DEFAULT))
        style = pc.load_filter_plot_style()
        missing = sorted(set(raw["band"]) - set(style.color_dict) | set(raw["band"]) - set(style.mark_dict))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
