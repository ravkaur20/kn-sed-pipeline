"""Saved diagnostic figures for NB5 mangling (bundle-aware QA; Agg backend)."""

from __future__ import annotations

import html
import os
from typing import Any, Mapping, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False
    plt = None  # type: ignore


def diagnostics_available() -> bool:
    return _HAS_MPL


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _aligned_wls_y(wls, y):
    """Truncate x/y to common length for matplotlib when grids disagree."""
    w = np.asarray(wls, dtype=float)
    yy = np.asarray(y, dtype=float)
    n = min(w.size, yy.size)
    return w[:n], yy[:n]


def save_group_mangle_diagnostics(
    diagnostics_dir: str,
    group_id: str,
    *,
    prescaled: Mapping[str, Mapping[str, np.ndarray]],
    mangled: Mapping[str, Mapping[str, np.ndarray]],
    masks: Mapping[str, np.ndarray],
    merge_order: Optional[list[str]] = None,
    seam_qa: Optional[list[dict[str, Any]]] = None,
    per_arm_masks: Optional[Mapping[str, np.ndarray]] = None,
    mangled_per_arm: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
) -> None:
    if not _HAS_MPL:
        return
    _ensure_dir(diagnostics_dir)
    safe = group_id.replace("/", "_").replace(" ", "_")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(prescaled), 1)))

    triple_mode = bool(mangled_per_arm)
    simple_pdf = os.path.join(
        diagnostics_dir, "group_%s_prescaled_vs_mangled.pdf" % safe
    )
    triple_pdf = os.path.join(
        diagnostics_dir, "group_%s_prescaled_perarm_bundle.pdf" % safe
    )

    if triple_mode:
        if os.path.isfile(simple_pdf):
            os.remove(simple_pdf)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(sorted(prescaled.keys())):
            c = colors[i % len(colors)]
            pre = prescaled[name]
            pre_f = np.log10(np.clip(np.asarray(pre["flux"], dtype=float), 1e-30, None))
            ax.plot(
                pre["wls"],
                pre_f,
                color=c,
                alpha=0.35,
                lw=0.8,
                label="%s prescaled" % name,
            )
            pa = mangled_per_arm.get(name) if mangled_per_arm else None
            if pa is not None:
                pa_f = np.log10(np.clip(np.asarray(pa["flux"], dtype=float), 1e-30, None))
                ax.plot(
                    pa["wls"],
                    pa_f,
                    color=c,
                    ls="--",
                    lw=1.0,
                    alpha=0.9,
                    label="%s per-arm mangled" % name,
                )
            bund = mangled.get(name)
            if bund is not None:
                bund_f = np.log10(np.clip(np.asarray(bund["flux"], dtype=float), 1e-30, None))
                ax.plot(
                    bund["wls"],
                    bund_f,
                    color=c,
                    ls="-",
                    lw=1.2,
                    label="%s bundle mangled" % name,
                )
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("log10(Fλ)")
        ax.set_title(
            "Mangle group %s: prescaled / per-arm / bundle" % group_id
        )
        ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(triple_pdf, bbox_inches="tight")
        plt.close(fig)
    else:
        if os.path.isfile(triple_pdf):
            os.remove(triple_pdf)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(sorted(prescaled.keys())):
            c = colors[i % len(colors)]
            pre = prescaled[name]
            man = mangled.get(name)
            pre_f = np.log10(np.clip(np.asarray(pre["flux"], dtype=float), 1e-30, None))
            ax.plot(pre["wls"], pre_f, color=c, alpha=0.35, lw=0.8, label="%s prescaled" % name)
            if man is not None:
                man_f = np.log10(np.clip(np.asarray(man["flux"], dtype=float), 1e-30, None))
                ax.plot(man["wls"], man_f, color=c, lw=1.0, label="%s mangled" % name)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("log10(Fλ)")
        ax.set_title("Mangle group %s: prescaled vs mangled" % group_id)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(simple_pdf, bbox_inches="tight")
        plt.close(fig)

    # mask overlay per arm
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(sorted(masks.keys())):
        c = colors[i % len(colors)]
        wls = np.asarray(prescaled[name]["wls"], dtype=float)
        m = np.asarray(masks[name], dtype=float)
        wx, mx = _aligned_wls_y(wls, m)
        ax.plot(wx, mx, color=c, lw=1.0, label=name)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Mangling mask (Δ log10 F)")
    ax.set_title("Group %s mangling masks" % group_id)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        os.path.join(diagnostics_dir, "group_%s_mask_overlay.pdf" % safe),
        bbox_inches="tight",
    )
    plt.close(fig)

    # seam jump bar chart
    if seam_qa:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ["%s→%s" % (s.get("ref", ""), s.get("arm", "")) for s in seam_qa]
        pre_j = [float(s.get("prescale_jump", float("nan"))) for s in seam_qa]
        man_j = [float(s.get("seam_jump_log10", float("nan"))) for s in seam_qa]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, pre_j, w, label="prescaled seam", color="steelblue")
        ax.bar(x + w / 2, man_j, w, label="mangled seam", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("|Δlog10 F| at join")
        ax.set_title("Group %s seam jumps" % group_id)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(diagnostics_dir, "group_%s_seam_jump.png" % safe),
            dpi=120,
        )
        plt.close(fig)

    # per-arm vs bundle mask compare
    if per_arm_masks:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, name in enumerate(sorted(masks.keys())):
            c = colors[i % len(colors)]
            wls = np.asarray(prescaled[name]["wls"], dtype=float)
            bundle_m = np.asarray(masks[name], dtype=float)
            wx, bm = _aligned_wls_y(wls, bundle_m)
            ax.plot(wx, bm, color=c, lw=1.2, label="%s bundle" % name)
            if name in per_arm_masks:
                pa = np.asarray(per_arm_masks[name], dtype=float)
                _, pa_a = _aligned_wls_y(wls, pa)
                if pa_a.size == bm.size:
                    ax.plot(wx, pa_a, "--", color=c, lw=0.9, alpha=0.7, label="%s per-arm" % name)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Mangling mask")
        ax.set_title("Group %s: per-arm vs bundle mask" % group_id)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(
            os.path.join(diagnostics_dir, "group_%s_per_arm_vs_bundle.pdf" % safe),
            bbox_inches="tight",
        )
        plt.close(fig)


def write_mangle_diagnostics_index(
    diagnostics_dir: str, groups: list[dict[str, Any]]
) -> None:
    _ensure_dir(diagnostics_dir)
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Mangle diagnostics</title></head><body>",
        "<h1>Mangling diagnostics</h1><ul>",
    ]
    for g in groups:
        gid = html.escape(str(g.get("id", "")))
        safe = gid.replace("/", "_")
        lines.append("<li><b>%s</b><ul>" % gid)
        triple_fn = "group_%s_prescaled_perarm_bundle.pdf" % safe
        has_triple = os.path.isfile(os.path.join(diagnostics_dir, triple_fn))
        for stem, ext in [
            ("prescaled_perarm_bundle", "pdf"),
            ("prescaled_vs_mangled", "pdf"),
            ("mask_overlay", "pdf"),
            ("seam_jump", "png"),
            ("per_arm_vs_bundle", "pdf"),
            ("phot_closure", "pdf"),
        ]:
            if stem == "prescaled_vs_mangled" and has_triple:
                continue
            fn = "group_%s_%s.%s" % (safe, stem, ext)
            if os.path.isfile(os.path.join(diagnostics_dir, fn)):
                lines.append("<li><a href='%s'>%s</a></li>" % (html.escape(fn), html.escape(fn)))
        seam_qa = g.get("seam_qa") or []
        if seam_qa:
            lines.append("<li>seam QA:<ul>")
            for s in seam_qa:
                flag = " FLAG" if s.get("regression_flag") else ""
                lines.append(
                    "<li>%s→%s prescale=%.4f mangled=%.4f%s</li>"
                    % (
                        html.escape(str(s.get("ref", ""))),
                        html.escape(str(s.get("arm", ""))),
                        float(s.get("prescale_jump", 0)),
                        float(s.get("seam_jump_log10", 0)),
                        flag,
                    )
                )
            lines.append("</ul></li>")
        lines.append("</ul></li>")
    lines.append("</ul></body></html>")
    with open(os.path.join(diagnostics_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run_mangle_diagnostics(
    diagnostics_dir: str,
    diag_groups: list[dict[str, Any]],
    report: Mapping[str, Any],
    *,
    per_arm_compare: bool = False,
) -> None:
    """Write all group diagnostic figures and index.html."""
    if not _HAS_MPL:
        return
    _ensure_dir(diagnostics_dir)
    index_groups: list[dict[str, Any]] = []
    for g in diag_groups:
        gid = str(g.get("id", "group"))
        mangled_per_arm = g.get("mangled_per_arm") if per_arm_compare else None
        if per_arm_compare and not mangled_per_arm:
            import warnings

            warnings.warn(
                "MANGLE_RUN_BOTH_FOR_DIAG=True but group %s has no mangled_per_arm; "
                "restart the notebook kernel and re-run so mangle_spectra_log picks up "
                "the latest code." % gid,
                stacklevel=2,
            )
        save_group_mangle_diagnostics(
            diagnostics_dir,
            gid,
            prescaled=g.get("prescaled", {}),
            mangled=g.get("mangled", {}),
            masks=g.get("masks", {}),
            merge_order=g.get("merge_order"),
            seam_qa=g.get("seam_qa"),
            per_arm_masks=g.get("per_arm_masks") if per_arm_compare else None,
            mangled_per_arm=mangled_per_arm,
        )
        index_groups.append(g)
    write_mangle_diagnostics_index(diagnostics_dir, index_groups)
