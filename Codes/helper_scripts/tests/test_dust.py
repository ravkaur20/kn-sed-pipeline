"""Tests for dust (final-SED dereddening, info.dat dust parameters)."""
import _bootstrap_paths  # noqa: F401
import json
import os
import tempfile
import unittest

import numpy as np

import dust  # noqa: E402

INFO_HEADER = "## test\n#\nName z RA Dec FullType Type Rich_Type EBV_MW EBV_host RV_MW RV_host\n"


def _write_info(d, body, header=INFO_HEADER):
    p = os.path.join(d, "info.dat")
    with open(p, "w") as fh:
        fh.write(header + body)
    return p


class TestExtinctionMag(unittest.TestCase):
    def test_zero_ebv_is_zero(self):
        np.testing.assert_array_equal(dust.extinction_mag([4000.0, 9000.0], 0.0, 3.1), 0.0)

    def test_a_v_equals_rv_times_ebv(self):
        # A(5495 A) ~ A_V = R_V E(B-V) for every law
        for law in dust.DUST_LAWS:
            a_v = dust.extinction_mag(np.array([5495.0]), 0.1, 3.1, law)[0]
            self.assertAlmostEqual(a_v, 0.31, delta=0.01, msg=law)

    def test_bluer_more_extinction(self):
        a = dust.extinction_mag(np.array([3500.0, 6000.0, 20000.0]), 0.1, 3.1)
        self.assertTrue(a[0] > a[1] > a[2] > 0)

    def test_bad_law_and_fm07_rv(self):
        with self.assertRaises(ValueError):
            dust.extinction_mag([5000.0], 0.1, 3.1, "cardelli")
        with self.assertRaises(ValueError):
            dust.extinction_mag([5000.0], 0.1, 4.0, "fm07")

    def test_host_term_uses_rest_frame(self):
        w = np.array([4000.0, 8000.0])
        z = 0.5
        a = dust.total_extinction_mag(w, ebv_mw=0.0, rv_mw=3.1, ebv_host=0.1, rv_host=3.1,
                                      redshift=z)
        np.testing.assert_allclose(a, dust.extinction_mag(w / (1 + z), 0.1, 3.1))

    def test_mw_and_host_add(self):
        w = np.array([4000.0, 8000.0])
        a = dust.total_extinction_mag(w, ebv_mw=0.1, rv_mw=3.1, ebv_host=0.2, rv_host=2.0,
                                      redshift=0.0)
        np.testing.assert_allclose(
            a, dust.extinction_mag(w, 0.1, 3.1) + dust.extinction_mag(w, 0.2, 2.0)
        )


class TestLoadDustParams(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_columns(self):
        p = _write_info(self.tmp.name, "SNx 0.01 0 0 KN KN KN 0.1 0.05 3.1 2.6\n")
        self.assertEqual(
            dust.load_dust_params(p, "SNx"),
            {"ebv_mw": 0.1, "rv_mw": 3.1, "ebv_host": 0.05, "rv_host": 2.6, "redshift": 0.01},
        )

    def test_missing_object_raises(self):
        p = _write_info(self.tmp.name, "SNx 0.01 0 0 KN KN KN 0.1 0.05 3.1 2.6\n")
        with self.assertRaises(KeyError):
            dust.load_dust_params(p, "SNy")

    def test_missing_rv_column_raises(self):
        header = "#\nName z RA Dec FullType Type Rich_Type EBV_MW EBV_host\n"
        p = _write_info(self.tmp.name, "SNx 0.01 0 0 KN KN KN 0.1 0.05\n", header=header)
        with self.assertRaisesRegex(KeyError, "RV_MW"):
            dust.load_dust_params(p, "SNx")

    def test_nonnumeric_rv_raises(self):
        p = _write_info(self.tmp.name, "SNx 0.01 0 0 KN KN KN 0.1 0.05 3.1 nan\n")
        with self.assertRaisesRegex(KeyError, "RV_host"):
            dust.load_dust_params(p, "SNx")


class TestDereddenFinalSpectra(unittest.TestCase):
    def test_round_trip_and_metadata(self):
        params = {"ebv_mw": 0.1, "rv_mw": 3.1, "ebv_host": 0.05, "rv_host": 2.6,
                  "redshift": 0.01}
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "as_observed"), os.path.join(d, "dereddened")
            os.makedirs(src)
            w = np.geomspace(3300.0, 24000.0, 50)
            f, e = 1e-16 * (w / 5000.0) ** -2, 1e-17 * np.ones_like(w)
            name = "-0.028970_FINAL_spec_FL.txt"
            np.savetxt(os.path.join(src, name), np.column_stack([w, f, e]), fmt="%.6E",
                       delimiter="\t", header="wls\tflux\tfluxerr")
            written = dust.deredden_final_spectra(src, dst, law="ccm89", **params)
            self.assertEqual(written, [os.path.join(dst, name)])

            with open(written[0]) as fh:
                self.assertEqual(fh.readline().strip(), "#wls\tflux\tfluxerr")
            d_out = np.loadtxt(written[0])
            a = dust.total_extinction_mag(w, law="ccm89", **params)
            np.testing.assert_allclose(d_out[:, 0], w, rtol=1e-6)
            np.testing.assert_allclose(d_out[:, 1], f * 10 ** (0.4 * a), rtol=1e-6)
            np.testing.assert_allclose(d_out[:, 2], e * 10 ** (0.4 * a), rtol=1e-6)
            self.assertTrue(np.all(d_out[:, 1] > f))

            with open(os.path.join(dst, dust.DUST_PARAMS_JSON)) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["law"], "ccm89")
            self.assertEqual(meta["n_spectra"], 1)
            self.assertEqual(meta["rv_host"], 2.6)


if __name__ == "__main__":
    unittest.main()
