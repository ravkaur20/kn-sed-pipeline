"""Logic checks for ext_grid / spec_mjd mangle row matching (no george)."""
import os
import unittest

import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import CODES

import numpy as np
import pandas as pd

OUT = os.path.join(
    CODES, "..", "Outputs", "AT2017gfo", "fitted_phot4mangling_AT2017gfo.dat"
)
LSPACE = os.path.join(
    CODES, "..", "Outputs", "AT2017gfo", "fitted_phot_logspace_AT2017gfo.dat"
)


@unittest.skipUnless(
    os.path.isfile(OUT) and os.path.isfile(LSPACE),
    "AT2017gfo mangle and logspace outputs not in tree",
)
class TestMangleKeysAT2017gfo(unittest.TestCase):
    def test_mangle_file_has_ext_grid_phase(self):
        d = pd.read_csv(OUT, sep="\t", nrows=0)
        self.assertIn("ext_grid_phase", d.columns)
        d2 = pd.read_csv(OUT, sep="\t", nrows=1)
        self.assertTrue(np.isfinite(d2["ext_grid_phase"].iloc[0]))

    def test_no_direct_row_for_n29_grid_file(self):
        """Filename stem is GP grid; ext_grid in table is per-spectrum (backfilled from spec_log_phase)."""
        phot = pd.read_csv(OUT, sep="\t")
        file_key = -2.958607
        m = np.zeros(len(phot), dtype=bool)
        if "ext_grid_phase" in phot.columns:
            m |= np.isclose(
                phot["ext_grid_phase"].values, file_key, rtol=0, atol=1e-4, equal_nan=True
            )
        m |= phot["spec_mjd"].values == file_key
        if "spec_log_phase" in phot.columns:
            m |= np.isclose(
                phot["spec_log_phase"].values, file_key, rtol=0, atol=1e-4, equal_nan=True
            )
        self.assertFalse(
            m.any(), "Grid-only key should not match per-spectrum rows before synthetic fix"
        )

    def test_logspace_has_nearest_n29(self):
        lp = pd.read_csv(LSPACE, sep="\t")
        fk = -2.958607
        i = (lp["Log_Phase"] - fk).abs().argmin()
        self.assertLess(abs(float(lp["Log_Phase"].iloc[i]) - fk), 2e-4)

    def test_logspace_nearest_0165_within_one_cell(self):
        """Grid stem 0.165363: nearest Log_Phase can be ~0.004 away; synthetic allows <0.01."""
        lp = pd.read_csv(LSPACE, sep="\t")
        fk = 0.165363
        pv = lp["Log_Phase"].astype(float).to_numpy()
        pos = int(np.argmin(np.abs(pv - fk)))
        self.assertLess(float(np.abs(pv[pos] - fk)), 0.01)


if __name__ == "__main__":
    unittest.main()
