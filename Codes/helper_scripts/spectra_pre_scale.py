"""Pre-scale spectroscopic arms / same-time epochs before mangling (notebook 4.5).

Default mode ``scale_only`` applies flux multipliers and keeps separate files.
Optional ``merge_join`` concatenates group members after scaling.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

OutputMode = Literal["scale_only", "merge_join"]
MergeGapPolicy = Literal["linear_bridge", "nan_gap"]

_ARM_ORDER_DEFAULT = ("uvb", "vis", "nir", "blue", "red")


@dataclass
class SpectrumEntry:
    mjd: float
    phase: float
    path: str
    basename: str

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class ScaleGroup:
    id: str
    members: list[str]
    output_mode: OutputMode = "scale_only"
    merge_order: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ScaleReport:
    snname: str
    default_output_mode: OutputMode
    groups: list[dict[str, Any]] = field(default_factory=list)
    ungrouped: list[str] = field(default_factory=list)


def resolve_spec_path(path: str, *, coco_path: str | None = None) -> str:
    """Normalize spectrum paths from list files (rewrite stale repo roots)."""
    import pipeline_config as pconf

    p = str(path).strip()
    if os.path.isfile(p):
        return p
    coco = os.path.normpath(coco_path or pconf.COCO_PATH)
    if "PyCoCo_templates" in p:
        tail = p.split("PyCoCo_templates", 1)[-1].lstrip("/\\")
        cand = os.path.join(coco, tail.replace("\\", "/"))
        if os.path.isfile(cand):
            return cand
    if p.startswith("/data/"):
        tail = p[len("/data/") :].lstrip("/")
        cand = os.path.join(pconf.spectroscopy_root(coco), tail.replace("\\", "/"))
        if os.path.isfile(cand):
            return cand
    return p


def load_spec_list(list_path: str, *, coco_path: str | None = None) -> list[SpectrumEntry]:
    rows = np.genfromtxt(list_path, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return []
    if rows.ndim == 0:
        rows = np.array([rows])
    out: list[SpectrumEntry] = []
    for row in rows:
        raw_path = resolve_spec_path(str(row["f2"]).strip(), coco_path=coco_path)
        out.append(
            SpectrumEntry(
                mjd=float(row["f0"]),
                phase=float(row["f1"]),
                path=raw_path,
                basename=os.path.basename(raw_path),
            )
        )
    return out


def write_spec_list(list_path: str, entries: list[SpectrumEntry]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(list_path)), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write("%.6f\t%.6f\t%s\n" % (e.mjd, e.phase, e.path))


def load_scale_groups_json(path: str) -> tuple[OutputMode, list[ScaleGroup]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    default_mode: OutputMode = data.get("default_output_mode", "scale_only")
    groups: list[ScaleGroup] = []
    for g in data.get("groups", []):
        members = [str(m) for m in g.get("members", [])]
        if not members:
            continue
        groups.append(
            ScaleGroup(
                id=str(g.get("id", "group_%i" % len(groups))),
                members=members,
                output_mode=g.get("output_mode", default_mode),
                merge_order=[str(x).lower() for x in g.get("merge_order", [])],
                reason=str(g.get("reason", "")),
            )
        )
    return default_mode, groups


def write_scale_groups_template(path: str, groups: list[ScaleGroup], default_mode: OutputMode = "scale_only") -> None:
    payload = {
        "default_output_mode": default_mode,
        "groups": [
            {
                "id": g.id,
                "members": g.members,
                "output_mode": g.output_mode,
                "merge_order": g.merge_order,
                "reason": g.reason,
            }
            for g in groups
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _basename_key(name: str) -> str:
    return os.path.basename(name)


def _arm_sort_key(filename: str, merge_order: list[str]) -> tuple[int, float]:
    low = filename.lower()
    for i, tag in enumerate(merge_order or list(_ARM_ORDER_DEFAULT)):
        if tag in low:
            return (i, 0.0)
    return (len(_ARM_ORDER_DEFAULT), 0.0)


@dataclass
class PairScaleResult:
    ref_name: str
    arm_name: str
    factor: float
    method: str
    overlap_source: str
    n_overlap: int
    gap_a: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref_name,
            "arm": self.arm_name,
            "factor": self.factor,
            "method": self.method,
            "overlap_source": self.overlap_source,
            "n_overlap": self.n_overlap,
            "gap_a": self.gap_a,
            "reason": self.reason,
        }


def load_spectrum_array(path: str) -> np.ndarray:
    """Load spectrum as structured array (wls, flux, fluxerr).

    Supports smoothed 3-column files and original 4-column XSHooter exports.
    """
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().lower()
    if "tell" in header or "flux_not" in header:
        raw = np.genfromtxt(
            path,
            dtype=None,
            encoding="utf-8",
            names=["wls", "flux", "flux_notell", "fluxerr"],
            comments="#",
        )
    else:
        raw = np.genfromtxt(
            path,
            dtype=None,
            encoding="utf-8",
            names=["wls", "flux", "fluxerr"],
            comments="#",
        )
    if raw.ndim == 0:
        raw = np.array([raw])
    err = raw["fluxerr"]
    if not np.any(np.isfinite(err) & (err > 0)):
        err = np.abs(raw["flux"]) * 0.1
    mask = (
        np.isfinite(raw["wls"])
        & np.isfinite(raw["flux"])
        & np.isfinite(err)
        & (raw["flux"] > 0.0)
    )
    out = np.empty(int(mask.sum()), dtype=[("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")])
    out["wls"] = raw["wls"][mask]
    out["flux"] = raw["flux"][mask]
    out["fluxerr"] = err[mask]
    return out


def save_spectrum_array(path: str, spec: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#wls\tflux\tfluxerr\n")
        for row in spec:
            fh.write("%E\t%E\t%E\n" % (row["wls"], row["flux"], row["fluxerr"]))


def overlap_scale_factor_wls(
    ref: np.ndarray,
    arm: np.ndarray,
    *,
    wl_tol_a: float = 1.0,
) -> tuple[float, int]:
    """WLS linear scale ``m`` so ``m * arm ≈ ref`` on overlapping valid pixels."""
    w_ref = ref["wls"]
    w_arm = arm["wls"]
    lo = max(float(np.min(w_ref)), float(np.min(w_arm)))
    hi = min(float(np.max(w_ref)), float(np.max(w_arm)))
    if hi <= lo:
        return 1.0, 0

    grid = np.linspace(lo, hi, max(50, min(len(ref), len(arm))))
    f_ref = np.interp(grid, w_ref, ref["flux"])
    e_ref = np.interp(grid, w_ref, ref["fluxerr"])
    f_arm = np.interp(grid, w_arm, arm["flux"])
    e_arm = np.interp(grid, w_arm, arm["fluxerr"])

    good = (
        np.isfinite(f_ref)
        & np.isfinite(f_arm)
        & (f_ref > 0)
        & (f_arm > 0)
        & np.isfinite(e_ref)
        & np.isfinite(e_arm)
    )
    if not np.any(good):
        return 1.0, 0

    w = 1.0 / np.maximum(e_arm[good] ** 2, 1e-60)
    num = float(np.sum(w * f_ref[good] * f_arm[good]))
    den = float(np.sum(w * f_arm[good] ** 2))
    if den <= 0.0:
        return 1.0, int(good.sum())
    return num / den, int(good.sum())


def pair_gap_a(spec_a: np.ndarray, spec_b: np.ndarray) -> float:
    """Positive gap (Å) at the join when bluer arm ends before redder arm starts."""
    if float(np.median(spec_a["wls"])) <= float(np.median(spec_b["wls"])):
        bluer, redder = spec_a, spec_b
    else:
        bluer, redder = spec_b, spec_a
    return float(np.min(redder["wls"])) - float(np.max(bluer["wls"]))


def _join_edge_windows(
    ref: np.ndarray,
    arm: np.ndarray,
    *,
    seam_half_width_a: float,
    edge_n_pix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flux windows at the join: ref side and arm side (for ``m * arm ≈ ref``)."""
    if float(np.median(ref["wls"])) <= float(np.median(arm["wls"])):
        ref_max = float(np.max(ref["wls"]))
        arm_min = float(np.min(arm["wls"]))
        ref_win = ref[
            (ref["wls"] >= ref_max - seam_half_width_a)
            & (ref["wls"] <= ref_max)
            & (ref["flux"] > 0)
        ]
        arm_win = arm[
            (arm["wls"] >= arm_min)
            & (arm["wls"] <= arm_min + seam_half_width_a)
            & (arm["flux"] > 0)
        ]
    else:
        ref_min = float(np.min(ref["wls"]))
        arm_max = float(np.max(arm["wls"]))
        ref_win = ref[
            (ref["wls"] >= ref_min)
            & (ref["wls"] <= ref_min + seam_half_width_a)
            & (ref["flux"] > 0)
        ]
        arm_win = arm[
            (arm["wls"] >= arm_max - seam_half_width_a)
            & (arm["wls"] <= arm_max)
            & (arm["flux"] > 0)
        ]
    if ref_win.size == 0 or arm_win.size == 0:
        ref_sorted = ref[np.argsort(ref["wls"])]
        arm_sorted = arm[np.argsort(arm["wls"])]
        if float(np.median(ref["wls"])) <= float(np.median(arm["wls"])):
            ref_win = ref_sorted[-max(1, edge_n_pix) :]
            arm_win = arm_sorted[: max(1, edge_n_pix)]
        else:
            ref_win = ref_sorted[: max(1, edge_n_pix)]
            arm_win = arm_sorted[-max(1, edge_n_pix) :]
    return ref_win, arm_win


def gap_seam_scale_factor(
    ref: np.ndarray,
    arm: np.ndarray,
    *,
    edge_n_pix: int = 10,
    seam_half_width_a: float = 50.0,
    gap_max_a: float = 400.0,
) -> tuple[float, int, float, str]:
    """Median flux ratio at join edges when overlap is absent (``m * arm ≈ ref``)."""
    gap = pair_gap_a(ref, arm)
    if gap <= 0.0:
        return 1.0, 0, gap, "has_overlap"
    if gap > gap_max_a:
        return 1.0, 0, gap, "gap_too_large"

    ref_win, arm_win = _join_edge_windows(
        ref, arm, seam_half_width_a=seam_half_width_a, edge_n_pix=edge_n_pix
    )
    if ref_win.size == 0 or arm_win.size == 0:
        return 1.0, 0, gap, "insufficient_edge_pixels"

    ref_med = float(np.median(ref_win["flux"]))
    arm_med = float(np.median(arm_win["flux"]))
    if arm_med <= 0.0 or ref_med <= 0.0:
        return 1.0, 0, gap, "invalid_flux"
    n_used = min(len(ref_win), len(arm_win))
    return ref_med / arm_med, n_used, gap, "gap_seam"


def compute_pair_scale_factor(
    ref: np.ndarray,
    arm: np.ndarray,
    *,
    ref_name: str = "",
    arm_name: str = "",
    wl_tol_a: float = 1.0,
    gap_max_a: float = 400.0,
    edge_n_pix: int = 10,
    seam_half_width_a: float = 50.0,
) -> PairScaleResult:
    """Smoothed-only: overlap WLS, then gap-seam at join edges."""
    gap = pair_gap_a(ref, arm)

    m, n = overlap_scale_factor_wls(ref, arm, wl_tol_a=wl_tol_a)
    if n > 0:
        return PairScaleResult(
            ref_name=ref_name,
            arm_name=arm_name,
            factor=m,
            method="overlap_wls",
            overlap_source="smoothed",
            n_overlap=n,
            gap_a=gap,
        )

    if 0.0 < gap <= gap_max_a:
        m, n, gap, reason = gap_seam_scale_factor(
            ref,
            arm,
            edge_n_pix=edge_n_pix,
            seam_half_width_a=seam_half_width_a,
            gap_max_a=gap_max_a,
        )
        if n > 0 and reason == "gap_seam":
            return PairScaleResult(
                ref_name=ref_name,
                arm_name=arm_name,
                factor=m,
                method="gap_seam",
                overlap_source="smoothed",
                n_overlap=n,
                gap_a=gap,
            )

    reason = "gap_too_large" if gap > gap_max_a else "no_overlap"
    return PairScaleResult(
        ref_name=ref_name,
        arm_name=arm_name,
        factor=1.0,
        method="skip",
        overlap_source="none",
        n_overlap=0,
        gap_a=gap,
        reason=reason,
    )


def apply_flux_scale(spec: np.ndarray, factor: float) -> np.ndarray:
    out = spec.copy()
    out["flux"] = out["flux"] * factor
    out["fluxerr"] = out["fluxerr"] * abs(factor)
    return out


def median_snr(spec: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = spec["flux"] / np.maximum(spec["fluxerr"], 1e-99)
    snr = snr[np.isfinite(snr) & (snr > 0)]
    if snr.size == 0:
        return 0.0
    return float(np.median(snr))


def _bridge_neighbor(
    order: list[str],
    arm_name: str,
    anchor_name: str,
) -> Optional[str]:
    """Next arm in ``merge_order`` toward ``anchor`` (one hop); None if adjacent or invalid."""
    if arm_name == anchor_name:
        return None
    try:
        arm_idx = order.index(arm_name)
        anchor_idx = order.index(anchor_name)
    except ValueError:
        return None
    if abs(arm_idx - anchor_idx) <= 1:
        return None
    if arm_idx < anchor_idx:
        return order[arm_idx + 1]
    return order[arm_idx - 1]


def scale_group_members(
    member_paths: list[str],
    *,
    merge_order: Optional[list[str]] = None,
    wl_tol_a: float = 1.0,
    chain_mode: bool = False,
    star_bridge_fallback: bool = True,
    gap_max_a: float = 400.0,
    edge_n_pix: int = 10,
    seam_half_width_a: float = 50.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], str, list[dict[str, Any]]]:
    """Return scaled spectra, per-file factors, reference basename, and per-pair link metadata."""
    loaded = {os.path.basename(p): load_spectrum_array(p) for p in member_paths}
    if not loaded:
        return {}, {}, "", []

    order = sorted(
        loaded.keys(),
        key=lambda n: _arm_sort_key(n, merge_order or list(_ARM_ORDER_DEFAULT)),
    )
    if chain_mode:
        anchor = order[0]
    else:
        anchor = max(order, key=lambda n: median_snr(loaded[n]))

    factors: dict[str, float] = {anchor: 1.0}
    scaled: dict[str, np.ndarray] = {anchor: loaded[anchor].copy()}
    pair_links: list[dict[str, Any]] = []

    if chain_mode:
        for arm_name in order[1:]:
            ref_name = order[order.index(arm_name) - 1]
            result = compute_pair_scale_factor(
                scaled[ref_name],
                loaded[arm_name],
                ref_name=ref_name,
                arm_name=arm_name,
                wl_tol_a=wl_tol_a,
                gap_max_a=gap_max_a,
                edge_n_pix=edge_n_pix,
                seam_half_width_a=seam_half_width_a,
            )
            factors[arm_name] = result.factor
            scaled[arm_name] = apply_flux_scale(loaded[arm_name], result.factor)
            pair_links.append(result.to_dict())
    else:
        pair_kw = dict(
            wl_tol_a=wl_tol_a,
            gap_max_a=gap_max_a,
            edge_n_pix=edge_n_pix,
            seam_half_width_a=seam_half_width_a,
        )
        skipped: list[tuple[str, PairScaleResult]] = []

        for name in order:
            if name == anchor:
                continue
            direct = compute_pair_scale_factor(
                loaded[anchor],
                loaded[name],
                ref_name=anchor,
                arm_name=name,
                **pair_kw,
            )
            factors[name] = direct.factor
            scaled[name] = apply_flux_scale(loaded[name], direct.factor)
            pair_links.append(direct.to_dict())
            if direct.method == "skip":
                skipped.append((name, direct))

        if star_bridge_fallback and skipped:
            for arm_name, direct in skipped:
                bridge = _bridge_neighbor(order, arm_name, anchor)
                if bridge is None or bridge not in scaled:
                    continue
                bridge_result = compute_pair_scale_factor(
                    scaled[bridge],
                    loaded[arm_name],
                    ref_name=bridge,
                    arm_name=arm_name,
                    **pair_kw,
                )
                if bridge_result.method == "skip":
                    continue
                factors[arm_name] = bridge_result.factor
                scaled[arm_name] = apply_flux_scale(loaded[arm_name], bridge_result.factor)
                fallback_link = bridge_result.to_dict()
                fallback_link["fallback"] = "bridge"
                fallback_link["direct_ref"] = direct.ref_name
                fallback_link["direct_method"] = direct.method
                fallback_link["direct_gap_a"] = direct.gap_a
                for i, lk in enumerate(pair_links):
                    if lk.get("arm") == arm_name and lk.get("method") == "skip":
                        pair_links[i] = fallback_link
                        break
                else:
                    pair_links.append(fallback_link)

    return scaled, factors, anchor, pair_links


def merge_spectra_concat(
    scaled: dict[str, np.ndarray],
    *,
    merge_order: Optional[list[str]] = None,
    gap_policy: MergeGapPolicy = "linear_bridge",
    gap_log10: float = 0.005,
) -> np.ndarray:
    names = sorted(
        scaled.keys(),
        key=lambda n: _arm_sort_key(n, merge_order or list(_ARM_ORDER_DEFAULT)),
    )
    parts = [scaled[n] for n in names]
    merged = np.concatenate(parts)
    order = np.argsort(merged["wls"])
    merged = merged[order]

    # dedupe near-equal wavelength (keep first)
    if len(merged) > 1:
        dw = np.diff(merged["wls"])
        keep = np.ones(len(merged), dtype=bool)
        keep[1:] = dw > 1e-3
        merged = merged[keep]

    if gap_policy == "nan_gap" or len(merged) < 2:
        return merged

    # optional small-gap linear bridge (in log10 lambda space)
    w = merged["wls"].astype(float)
    logw = np.log10(w)
    for i in range(len(merged) - 1):
        dlog = logw[i + 1] - logw[i]
        if 0.0 < dlog <= gap_log10:
            n_insert = max(2, int(dlog / gap_log10 * 5))
            mid_log = np.linspace(logw[i], logw[i + 1], n_insert + 2)[1:-1]
            mid_w = 10 ** mid_log
            t = (mid_log - logw[i]) / dlog
            f = (1 - t) * merged["flux"][i] + t * merged["flux"][i + 1]
            e = (1 - t) * merged["fluxerr"][i] + t * merged["fluxerr"][i + 1]
            # append — caller may re-sort; for simplicity rebuild once
            extra = np.array(list(zip(mid_w, f, e)), dtype=merged.dtype)
            merged = np.concatenate([merged, extra])
    order = np.argsort(merged["wls"])
    return merged[order]


def suggest_scale_groups(
    entries: list[SpectrumEntry],
    *,
    same_time_minutes: float = 5.0,
) -> list[ScaleGroup]:
    """Cluster list rows by MJD proximity (assist only)."""
    if not entries:
        return []
    dt_days = same_time_minutes / (24.0 * 60.0)
    used = set()
    groups: list[ScaleGroup] = []
    sorted_e = sorted(entries, key=lambda e: e.mjd)
    gid = 0
    for i, e in enumerate(sorted_e):
        if i in used:
            continue
        cluster = [e]
        used.add(i)
        for j in range(i + 1, len(sorted_e)):
            if j in used:
                continue
            if abs(sorted_e[j].mjd - e.mjd) <= dt_days:
                cluster.append(sorted_e[j])
                used.add(j)
        if len(cluster) > 1:
            gid += 1
            groups.append(
                ScaleGroup(
                    id="suggested_%03i_mjd_%.4f" % (gid, e.mjd),
                    members=[c.basename for c in cluster],
                    output_mode="scale_only",
                    merge_order=_infer_merge_order([c.basename for c in cluster]),
                    reason="auto: |ΔMJD| <= %.2f min" % same_time_minutes,
                )
            )
    return groups


def _infer_merge_order(filenames: list[str]) -> list[str]:
    tags = []
    for tag in _ARM_ORDER_DEFAULT:
        if any(tag in f.lower() for f in filenames):
            tags.append(tag)
    return tags


def resolve_member_path(basename: str, entries: list[SpectrumEntry]) -> str:
    for e in entries:
        if e.basename == basename or basename in e.path:
            p = e.path
            if os.path.isfile(p):
                return p
            raise FileNotFoundError("Spectrum path missing on disk: %s" % p)
    raise FileNotFoundError("List has no spectrum matching %r" % basename)


def run_prescale_pipeline(
    *,
    snname: str,
    coco_path: str,
    output_dir: str,
    groups_json: Optional[str] = None,
    default_output_mode: OutputMode = "scale_only",
    same_time_minutes: float = 5.0,
    wl_tol_a: float = 1.0,
    gap_log10: float = 0.005,
    merge_gap_policy: MergeGapPolicy = "linear_bridge",
    chain_mode: Optional[bool] = None,
    star_bridge_fallback: Optional[bool] = None,
    gap_max_a: Optional[float] = None,
    edge_n_pix: Optional[int] = None,
    seam_half_width_a: Optional[float] = None,
    write_diagnostics: bool = True,
    diagnostics_dir: Optional[str] = None,
) -> ScaleReport:
    import pipeline_config as pconf

    if chain_mode is None:
        chain_mode = pconf.SPEC_SCALE_CHAIN_MODE
    if star_bridge_fallback is None:
        star_bridge_fallback = pconf.SPEC_SCALE_STAR_BRIDGE_FALLBACK
    if gap_max_a is None:
        gap_max_a = pconf.SPEC_SCALE_GAP_MAX_A
    if edge_n_pix is None:
        edge_n_pix = pconf.SPEC_SCALE_EDGE_N_PIX
    if seam_half_width_a is None:
        seam_half_width_a = pconf.SPEC_SCALE_SEAM_HALF_WIDTH_A

    list_path = pconf.smoothed_spec_list_path(coco_path, snname)
    entries = load_spec_list(list_path)
    if not entries:
        raise FileNotFoundError("No spectra in %s" % list_path)

    groups_path = groups_json or pconf.spec_scale_groups_json_path(output_dir, snname)
    if os.path.isfile(groups_path):
        default_output_mode, groups = load_scale_groups_json(groups_path)
    else:
        groups = suggest_scale_groups(entries, same_time_minutes=same_time_minutes)
        write_scale_groups_template(groups_path, groups, default_output_mode)

    out_spec_dir = pconf.prescaled_spec_dir(coco_path, snname)
    os.makedirs(out_spec_dir, exist_ok=True)

    grouped_basenames: set[str] = set()
    for g in groups:
        grouped_basenames.update(_basename_key(m) for m in g.members)

    report = ScaleReport(snname=snname, default_output_mode=default_output_mode)
    new_entries: list[SpectrumEntry] = []

    if write_diagnostics and diagnostics_dir:
        from spec_scale_diagnostics import diagnostics_available, save_group_diagnostics

        if not diagnostics_available():
            import warnings

            warnings.warn(
                "matplotlib unavailable; spec scale figures skipped (scaling outputs still written)",
                stacklevel=1,
            )

    for g in groups:
        mode = g.output_mode or default_output_mode
        paths = [resolve_member_path(_basename_key(m), entries) for m in g.members]

        scaled, factors, anchor, pair_links = scale_group_members(
            paths,
            merge_order=g.merge_order or None,
            wl_tol_a=wl_tol_a,
            chain_mode=chain_mode,
            star_bridge_fallback=star_bridge_fallback,
            gap_max_a=gap_max_a,
            edge_n_pix=edge_n_pix,
            seam_half_width_a=seam_half_width_a,
        )
        group_rec: dict[str, Any] = {
            "id": g.id,
            "output_mode": mode,
            "reference": anchor,
            "chain_mode": chain_mode,
            "star_bridge_fallback": star_bridge_fallback,
            "scale_factors": factors,
            "pair_links": pair_links,
            "members": [],
        }
        merged_arr: Optional[np.ndarray] = None

        if mode == "merge_join":
            merged_arr = merge_spectra_concat(
                scaled,
                merge_order=g.merge_order or None,
                gap_policy=merge_gap_policy,
                gap_log10=gap_log10,
            )
            rep_entry = next(e for e in entries if e.basename in scaled)
            out_name = "%.6f_merged_%s.dat" % (rep_entry.mjd, re.sub(r"[^\w]+", "_", g.id))
            out_path = os.path.join(out_spec_dir, out_name)
            save_spectrum_array(out_path, merged_arr)
            new_entries.append(
                SpectrumEntry(mjd=rep_entry.mjd, phase=rep_entry.phase, path=out_path, basename=out_name)
            )
            group_rec["merged_file"] = out_name
            for bn, spec in scaled.items():
                group_rec["members"].append({"file": bn, "factor": factors.get(bn, 1.0)})
        else:
            for bn, spec in scaled.items():
                src_entry = next(e for e in entries if e.basename == bn)
                out_path = os.path.join(out_spec_dir, bn)
                save_spectrum_array(out_path, spec)
                new_entries.append(
                    SpectrumEntry(
                        mjd=src_entry.mjd,
                        phase=src_entry.phase,
                        path=out_path,
                        basename=bn,
                    )
                )
                group_rec["members"].append({"file": bn, "factor": factors.get(bn, 1.0), "output": bn})

        report.groups.append(group_rec)

        if write_diagnostics and diagnostics_dir and diagnostics_available():
            before = {
                bn: load_spectrum_array(resolve_member_path(bn, entries)) for bn in scaled
            }
            save_group_diagnostics(
                diagnostics_dir,
                g.id,
                before=before,
                after=scaled,
                factors=factors,
                merged=merged_arr,
                pair_links=pair_links,
            )

    # ungrouped: copy as-is
    for e in entries:
        if e.basename in grouped_basenames:
            continue
        out_path = os.path.join(out_spec_dir, e.basename)
        if os.path.abspath(e.path) != os.path.abspath(out_path):
            shutil.copy2(e.path, out_path)
        new_entries.append(
            SpectrumEntry(mjd=e.mjd, phase=e.phase, path=out_path, basename=e.basename)
        )
        report.ungrouped.append(e.basename)

    list_out = pconf.prescaled_spec_list_path(coco_path, snname)
    write_spec_list(list_out, new_entries)

    report_path = pconf.spec_scale_report_json_path(output_dir, snname)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "snname": report.snname,
                "default_output_mode": report.default_output_mode,
                "groups": report.groups,
                "ungrouped": report.ungrouped,
                "prescaled_list": list_out,
            },
            fh,
            indent=2,
        )

    if write_diagnostics and diagnostics_dir:
        from spec_scale_diagnostics import write_diagnostics_index

        write_diagnostics_index(diagnostics_dir, report.groups)

    return report
