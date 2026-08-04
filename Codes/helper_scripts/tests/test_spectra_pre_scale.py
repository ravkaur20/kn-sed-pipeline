import _bootstrap_paths  # noqa: F401
import os
import tempfile
import unittest

import numpy as np

from spectra_pre_scale import (
    _bridge_neighbor,
    compute_pair_scale_factor,
    gap_seam_scale_factor,
    load_spectrum_array,
    merge_spectra_concat,
    overlap_scale_factor_wls,
    pair_gap_a,
    save_spectrum_array,
    scale_group_members,
    suggest_scale_groups,
    SpectrumEntry,
)


def _make_spec(wls, flux):
    dt = np.dtype([("wls", "f8"), ("flux", "f8"), ("fluxerr", "f8")])
    out = np.empty(len(wls), dtype=dt)
    out["wls"] = wls
    out["flux"] = flux
    out["fluxerr"] = flux * 0.1
    return out


class TestSpectraPreScale(unittest.TestCase):
    def test_overlap_scale_factor(self):
        w = np.linspace(5000, 7000, 100)
        ref = _make_spec(w, np.ones_like(w) * 2e-15)
        arm = _make_spec(w, np.ones_like(w) * 1e-15)
        m, n = overlap_scale_factor_wls(ref, arm)
        self.assertGreater(n, 10)
        self.assertAlmostEqual(m, 2.0, delta=0.05)

    def test_scale_group_members_star(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = np.linspace(3000, 6000, 80)
            w2 = np.linspace(5500, 9000, 80)
            p1 = os.path.join(td, "uvb.dat")
            p2 = os.path.join(td, "vis.dat")
            save_spectrum_array(p1, _make_spec(w1, np.ones(len(w1)) * 1e-15))
            save_spectrum_array(p2, _make_spec(w2, np.ones(len(w2)) * 0.5e-15))
            scaled, factors, ref, links = scale_group_members(
                [p1, p2], merge_order=["uvb", "vis"], chain_mode=False
            )
            self.assertIn("uvb.dat", scaled)
            self.assertIn("vis.dat", scaled)
            self.assertAlmostEqual(factors["vis.dat"], 2.0, delta=0.15)
            self.assertEqual(links[0]["method"], "overlap_wls")

    def test_gap_seam_small_gap_star_orientation(self):
        w_uvb = np.linspace(5000, 5600, 50)
        w_vis = np.linspace(5637, 7000, 50)
        uvb = _make_spec(w_uvb, np.ones(len(w_uvb)) * 8e-18)
        vis = _make_spec(w_vis, np.ones(len(w_vis)) * 1e-17)
        self.assertGreater(pair_gap_a(uvb, vis), 30.0)
        result = compute_pair_scale_factor(
            vis, uvb, ref_name="vis", arm_name="uvb", gap_max_a=400.0
        )
        self.assertEqual(result.method, "gap_seam")
        self.assertAlmostEqual(result.factor, 1.25, delta=0.05)

    def test_gap_seam_large_gap_vis_nir(self):
        w_vis = np.linspace(6300, 9900, 80)
        w_nir = np.linspace(10243, 12000, 80)
        vis = _make_spec(w_vis, np.ones(len(w_vis)) * 1e-17)
        nir = _make_spec(w_nir, np.ones(len(w_nir)) * 1.03e-17)
        gap = pair_gap_a(vis, nir)
        self.assertGreater(gap, 300.0)
        self.assertLess(gap, 400.0)
        result = compute_pair_scale_factor(
            vis, nir, ref_name="vis", arm_name="nir", gap_max_a=400.0
        )
        self.assertEqual(result.method, "gap_seam")
        self.assertAlmostEqual(result.factor, 1.0, delta=0.05)

    def test_gap_seam_too_large_skipped(self):
        w1 = np.linspace(5000, 5600, 50)
        w2 = np.linspace(6100, 7000, 50)
        ref = _make_spec(w1, np.ones(len(w1)) * 2e-15)
        arm = _make_spec(w2, np.ones(len(w2)) * 1e-15)
        result = compute_pair_scale_factor(ref, arm, ref_name="a", arm_name="b", gap_max_a=100.0)
        self.assertEqual(result.method, "skip")
        self.assertEqual(result.factor, 1.0)

    def test_bridge_neighbor(self):
        order = ["uvb.dat", "vis.dat", "nir.dat"]
        self.assertEqual(_bridge_neighbor(order, "uvb.dat", "nir.dat"), "vis.dat")
        self.assertEqual(_bridge_neighbor(order, "nir.dat", "uvb.dat"), "vis.dat")
        self.assertIsNone(_bridge_neighbor(order, "vis.dat", "nir.dat"))

    def test_star_bridge_fallback_nir_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            w_uvb = np.linspace(3000, 5600, 50)
            w_vis = np.linspace(5637, 9900, 80)
            w_nir = np.linspace(10243, 12000, 80)
            p_uvb = os.path.join(td, "xshooter_uvb.dat")
            p_vis = os.path.join(td, "xshooter_vis.dat")
            p_nir = os.path.join(td, "xshooter_nir.dat")
            save_spectrum_array(p_uvb, _make_spec(w_uvb, np.ones(50) * 1e-18))
            save_spectrum_array(p_vis, _make_spec(w_vis, np.ones(80) * 1e-17))
            nir_spec = _make_spec(w_nir, np.ones(80) * 1e-16)
            nir_spec["fluxerr"] *= 0.01
            save_spectrum_array(p_nir, nir_spec)

            scaled, factors, anchor, links = scale_group_members(
                [p_uvb, p_vis, p_nir],
                merge_order=["uvb", "vis", "nir"],
                chain_mode=False,
                star_bridge_fallback=True,
            )
            self.assertIn("nir", anchor.lower())
            uvb_link = next(lk for lk in links if lk["arm"] == "xshooter_uvb.dat")
            self.assertEqual(uvb_link.get("fallback"), "bridge")
            self.assertEqual(uvb_link.get("direct_method"), "skip")
            self.assertNotAlmostEqual(factors["xshooter_uvb.dat"], 1.0, places=2)
            vis_link = next(lk for lk in links if lk["arm"] == "xshooter_vis.dat")
            self.assertIn(vis_link["method"], ("overlap_wls", "gap_seam"))

    def test_star_bridge_fallback_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            w_uvb = np.linspace(3000, 5600, 50)
            w_vis = np.linspace(5637, 9900, 80)
            w_nir = np.linspace(10243, 12000, 80)
            p_uvb = os.path.join(td, "xshooter_uvb.dat")
            p_vis = os.path.join(td, "xshooter_vis.dat")
            p_nir = os.path.join(td, "xshooter_nir.dat")
            save_spectrum_array(p_uvb, _make_spec(w_uvb, np.ones(50) * 1e-18))
            save_spectrum_array(p_vis, _make_spec(w_vis, np.ones(80) * 1e-17))
            nir_spec = _make_spec(w_nir, np.ones(80) * 1e-16)
            nir_spec["fluxerr"] *= 0.01
            save_spectrum_array(p_nir, nir_spec)

            _, factors, _, links = scale_group_members(
                [p_uvb, p_vis, p_nir],
                merge_order=["uvb", "vis", "nir"],
                chain_mode=False,
                star_bridge_fallback=False,
            )
            uvb_link = next(lk for lk in links if lk["arm"] == "xshooter_uvb.dat")
            self.assertEqual(uvb_link["method"], "skip")
            self.assertEqual(factors["xshooter_uvb.dat"], 1.0)

    def test_merge_join(self):
        w1 = np.linspace(3000, 5500, 50)
        w2 = np.linspace(5600, 9000, 50)
        scaled = {
            "a": _make_spec(w1, np.ones(len(w1))),
            "b": _make_spec(w2, np.ones(len(w2)) * 2),
        }
        merged = merge_spectra_concat(scaled, merge_order=["a", "b"])
        self.assertGreater(len(merged), 90)
        self.assertGreater(float(np.min(np.diff(merged["wls"]))), 0.0)

    def test_suggest_groups(self):
        entries = [
            SpectrumEntry(57983.969, 1.0, "/p/a", "a"),
            SpectrumEntry(57983.969, 1.0, "/p/b", "b"),
            SpectrumEntry(57990.0, 8.0, "/p/c", "c"),
        ]
        groups = suggest_scale_groups(entries, same_time_minutes=5.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].members), 2)

    def test_load_original_format(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "orig.dat")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#wavelength\tflux_tell_corrected\tflux_not_tell_corrected\tvariance/error\n")
                for w in np.linspace(5000, 6000, 20):
                    fh.write("%.4f\t%.4e\t%.4e\t%.4e\n" % (w, 1e-15, 1e-15, 1e-16))
            spec = load_spectrum_array(path)
            self.assertEqual(len(spec), 20)


if __name__ == "__main__":
    unittest.main()
