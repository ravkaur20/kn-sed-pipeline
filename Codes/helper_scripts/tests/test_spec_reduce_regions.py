"""0.2 region editing: regions printed as numbers that paste back unchanged."""
import _bootstrap_paths  # noqa: F401
import ast
import contextlib
import io
import unittest

import spec_reduce as sr  # noqa: E402


class TestRegionText(unittest.TestCase):
    def test_fmt_regions_round_trips(self):
        regs = [(6860, 6890), (7590.04, 7700.26), (9300.5, 9650)]
        self.assertEqual(sr.fmt_regions(regs),
                         "[(6860, 6890), (7590, 7700.3), (9300.5, 9650)]")
        self.assertEqual(ast.literal_eval(sr.fmt_regions([(6860, 6890)])), [(6860, 6890)])
        self.assertEqual(sr.fmt_regions([]), "[]")

    def test_print_proposed_gives_paste_lines(self):
        proposed = {"spike": [(5000.0, 5004.0), (6000.0, 6020.0)],
                    "low_snr": [], "telluric": [(7590, 7700)]}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sr.print_proposed(proposed, (4100.4, 9899.6))
        out = buf.getvalue().splitlines()
        self.assertEqual(out[0], "telluric = [(7590, 7700)]")
        self.assertEqual(out[1], "edges = (4100, 9900)")
        self.assertIn("2 regions", out[3])

class TestRestFrameLoad(unittest.TestCase):
    """Rest-frame inputs move to the observed frame: lambda x (1+z), F_lambda / (1+z)."""

    def setUp(self):
        import os
        import tempfile
        from unittest import mock

        import numpy as np

        self.np = np
        self.tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.tmp.name, "SNX"))
        self.wl = np.array([4000.0, 5000.0, 6000.0])
        self.flux = np.array([1.0, 2.0, 3.0])
        self.err = np.array([0.1, 0.2, 0.3])
        rows4 = np.column_stack([self.wl, self.flux, self.flux, self.err])
        for name, data in (("a_LDSS3.dat", rows4[:, :2]), ("b_LDSS3.dat", rows4),
                           ("c_XS.dat", rows4)):
            np.savetxt(os.path.join(self.tmp.name, "SNX", name), data, header="cols")
        self.z = 0.01
        for p in (mock.patch.object(sr, "RAW_DIR", self.tmp.name),
                  mock.patch.dict(sr.REST_FRAME_FILES, {"SNX": ["LDSS3"]}),
                  mock.patch.dict(sr.pconf.SN_REDSHIFT, {"SNX": self.z})):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def _load(self, name):
        with contextlib.redirect_stdout(io.StringIO()):
            return sr.load_raw("SNX", name)

    def test_rest_frame_without_errors(self):
        s = self._load("a_LDSS3.dat")
        self.assertTrue(s["shifted_to_observed"])
        self.np.testing.assert_allclose(s["wl"], self.wl * (1 + self.z))
        self.np.testing.assert_allclose(s["flux"], self.flux / (1 + self.z))
        self.assertIsNone(s["err"])
        # flux per unit wavelength x wavelength interval is conserved
        trap = lambda y, x: float(self.np.sum(0.5 * (y[1:] + y[:-1]) * self.np.diff(x)))
        self.np.testing.assert_allclose(trap(s["flux"], s["wl"]),
                                        trap(self.flux, self.wl))

    def test_rest_frame_with_errors(self):
        s = self._load("b_LDSS3.dat")
        self.np.testing.assert_allclose(s["flux"], self.flux / (1 + self.z))
        self.np.testing.assert_allclose(s["err"], self.err / (1 + self.z))

    def test_observed_frame_untouched(self):
        s = self._load("c_XS.dat")
        self.assertFalse(s["shifted_to_observed"])
        self.np.testing.assert_array_equal(s["wl"], self.wl)
        self.np.testing.assert_array_equal(s["flux"], self.flux)
        self.np.testing.assert_array_equal(s["err"], self.err)


if __name__ == "__main__":
    unittest.main()
