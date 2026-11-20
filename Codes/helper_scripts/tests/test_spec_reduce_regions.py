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


if __name__ == "__main__":
    unittest.main()
