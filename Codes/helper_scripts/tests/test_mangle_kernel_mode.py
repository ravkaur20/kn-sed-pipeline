"""Unit tests for the mangle GP kernel length-scale helper."""
import _bootstrap_paths  # noqa: F401
import unittest
from unittest import mock

import pipeline_config as pconf
from mangle_spectra_log import _mangle_gp_kernel_length


class TestMangleKernelLengthHelper(unittest.TestCase):
    def test_fixed_5_default(self):
        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "fixed_5"):
            with mock.patch.object(pconf, "MANGLE_GP_KERNEL_FIXED", 5.0):
                self.assertAlmostEqual(_mangle_gp_kernel_length(3.7, 800), 5.0)

    def test_kernel_divide_scaled(self):
        with mock.patch.object(pconf, "MANGLE_GP_KERNEL_MODE", "kernel_divide_scaled"):
            self.assertAlmostEqual(_mangle_gp_kernel_length(4.0, 800), 200.0)


if __name__ == "__main__":
    unittest.main()
