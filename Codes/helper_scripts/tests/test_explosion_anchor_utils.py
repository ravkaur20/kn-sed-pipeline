import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np


import explosion_anchor_utils as ea


class TestExplosionAnchor(unittest.TestCase):
    def test_augment_lc_gp_training(self):
        lp = np.array([-1.0, 0.0])
        fl = np.array([-10.0, -9.0])
        fe = np.array([0.1, 0.1])
        sudo = np.array([False, False])
        norm = float(np.median(fl))
        fn = fl - norm
        en = fe.copy()
        lp2, fl2, fe2, sudo2, fn2, en2 = ea.augment_lc_gp_training_for_t0_anchor(
            lp, fl, fe, sudo, norm, fn, en, log_phase_anchor=-8.0, log_flux_cap=-50.0, log_flux_err=2.0
        )
        self.assertEqual(len(lp2), 3)
        self.assertAlmostEqual(float(lp2[-1]), -8.0)
        self.assertAlmostEqual(float(fl2[-1]), -50.0)
        self.assertTrue(bool(sudo2[-1]))
        self.assertAlmostEqual(float(fn2[-1]), -50.0 - norm)
        self.assertAlmostEqual(float(en2[-1]), 2.0)

    def test_appends_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "f.dat")
            with open(p, "w", encoding="utf-8") as f:
                f.write("Log_Phase\tb_log_flux\tb_log_flux_err\n")
                f.write("-1.0\t-10.0\t0.1\n")
            df = ea.append_explosion_anchor_row(p, log_phase=-8.0, log_flux_cap=-50.0)
            self.assertEqual(len(df), 2)
            self.assertAlmostEqual(float(df.iloc[-1]["Log_Phase"]), -8.0)
            self.assertAlmostEqual(float(df.iloc[-1]["b_log_flux"]), -50.0)


if __name__ == "__main__":
    unittest.main()
