"""Tests for raw-photometry MJD span helper."""
import _bootstrap_paths  # noqa: F401
import os
import sys
import tempfile
import unittest


import photometry_filter_utils as pfu


class TestFilterMjdFromRaw(unittest.TestCase):
    def test_min_max_per_band(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False, encoding="utf-8"
        ) as f:
            f.write("MJD,Mag,band,x\n")
            f.write("100.0,1,A,a\n")
            f.write("200.0,1,A,a\n")
            f.write("150.0,1,B,a\n")
            path = f.name
        try:
            d = pfu.filter_mjd_ranges_dict_from_raw_file(path)
            self.assertAlmostEqual(d["A"]["min_mjd"], 100.0)
            self.assertAlmostEqual(d["A"]["max_mjd"], 200.0)
            self.assertEqual(d["A"]["n_obs"], 2)
            self.assertAlmostEqual(d["B"]["min_mjd"], 150.0)
            self.assertAlmostEqual(d["B"]["max_mjd"], 150.0)
        finally:
            os.unlink(path)

    def test_json_roundtrip(self):
        tmpd = tempfile.mkdtemp()
        try:
            raw = os.path.join(tmpd, "sn.dat")
            with open(raw, "w", encoding="utf-8") as f:
                f.write("MJD,flux,band\n")
                f.write("1,0,X\n")
                f.write("3,0,X\n")
            js = os.path.join(tmpd, "out.json")
            pfu.write_band_mjd_ranges_json(raw, js)
            inner = pfu.load_band_mjd_ranges_json(js)
            self.assertAlmostEqual(inner["X"]["min_mjd"], 1.0)
            self.assertAlmostEqual(inner["X"]["max_mjd"], 3.0)
        finally:
            for fn in os.listdir(tmpd):
                os.unlink(os.path.join(tmpd, fn))
            os.rmdir(tmpd)


if __name__ == "__main__":
    unittest.main()
