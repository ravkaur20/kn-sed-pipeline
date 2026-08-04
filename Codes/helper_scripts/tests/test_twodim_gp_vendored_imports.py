"""Smoke tests for vendored ``helper_scripts/twodim_gp`` import chains used by NB6."""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import HELPER_SCRIPTS, TWODIM_GP


class TestTwodimGpGpVendoredImports(unittest.TestCase):
    def test_spec_bundle_helpers_via_gp2dim_export(self):
        import gp2dim_export as ex

        assign_fn, strings_fn = ex._import_spec_bundle_helpers()
        self.assertTrue(callable(assign_fn))
        self.assertTrue(callable(strings_fn))

    def test_plot_results_package_import(self):
        from twodim_gp.plot_results import _make_heatmap

        self.assertTrue(callable(_make_heatmap))

    def test_plot_bands_overview_script_imports(self):
        """``plot_bands_gp_overview`` uses flat imports; load with cwd on ``twodim_gp/``."""
        path = os.path.join(TWODIM_GP, "plot_bands_gp_overview.py")
        self.assertTrue(os.path.isfile(path), path)
        spec = importlib.util.spec_from_file_location("plot_bands_gp_overview", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        old_cwd = os.getcwd()
        old_path = list(sys.path)
        try:
            os.chdir(TWODIM_GP)
            if TWODIM_GP not in sys.path:
                sys.path.insert(0, TWODIM_GP)
            spec.loader.exec_module(mod)
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
        self.assertTrue(hasattr(mod, "main") or hasattr(mod, "parse_args"))


if __name__ == "__main__":
    unittest.main()
