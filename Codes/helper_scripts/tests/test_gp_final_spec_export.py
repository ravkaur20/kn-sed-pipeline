import os
import tempfile
import unittest

import _bootstrap_paths  # noqa: F401

from gp_final_spec_export import export_final_spec_from_full_gp


class TestGpFinalSpecExport(unittest.TestCase):
    def test_spec_extended_to_final_spec(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "full_gp")
            os.makedirs(src)
            stem = "0.842480"
            with open(os.path.join(src, "%s_spec_extended.txt" % stem), "w", encoding="utf-8") as fh:
                fh.write("#wls\tflux\tfluxerr\n4.000000E+03\t1.000000E-15\t1.000000E-16\n")
            out_base = os.path.join(td, "FINAL_spectra_2dim", "as_observed")
            written = export_final_spec_from_full_gp(src, out_base, variant="")
            self.assertEqual(len(written), 1)
            dst = os.path.join(out_base, "%s_FINAL_spec.txt" % stem)
            self.assertTrue(os.path.isfile(dst))


if __name__ == "__main__":
    unittest.main()
