import _bootstrap_paths  # noqa: F401
import os
import sys
import tempfile
import unittest

_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SCRIPT not in sys.path:
    sys.path.insert(0, _SCRIPT)


class TestClearSnRun(unittest.TestCase):
    def test_dry_run_outputs_only(self):
        import clear_sn_run as csr

        with tempfile.TemporaryDirectory() as td:
            sn = "TESTSN"
            out_sn = os.path.join(td, "Outputs", sn)
            os.makedirs(out_sn)
            with open(os.path.join(out_sn, "dummy.txt"), "w") as fh:
                fh.write("x")
            paths = csr.collect_clear_paths(sn, coco_path=td + os.sep)
            self.assertTrue(any(p.endswith(sn) for p in paths))
            removed = csr.clear_sn_run(
                sn,
                coco_path=td + os.sep,
                dry_run=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(out_sn, "dummy.txt")))
            self.assertGreater(len(removed), 0)

    def test_clears_output_contents(self):
        import clear_sn_run as csr

        with tempfile.TemporaryDirectory() as td:
            sn = "TESTSN"
            out_sn = os.path.join(td, "Outputs", sn)
            os.makedirs(out_sn)
            with open(os.path.join(out_sn, "dummy.txt"), "w") as fh:
                fh.write("x")
            csr.clear_sn_run(sn, coco_path=td + os.sep, yes=True)
            self.assertTrue(os.path.isdir(out_sn))
            self.assertEqual(os.listdir(out_sn), [])


if __name__ == "__main__":
    unittest.main()
