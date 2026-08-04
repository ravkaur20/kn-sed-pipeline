"""Per-iteration mangling + GP diagnostics orchestrator (Phase 3)."""

from __future__ import annotations

import html
import json
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False
    plt = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_group_maps(
    output_dir: str,
    snname: str,
    coco_path: str | None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    import pipeline_config as pconf
    from spectra_pre_scale import load_scale_groups_json

    basename_to_group: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}
    group_merge_order: dict[str, list[str]] = {}
    gj = pconf.resolve_mangle_groups_json(output_dir, snname)
    if not os.path.isfile(gj):
        return basename_to_group, group_members, group_merge_order
    _, groups = load_scale_groups_json(gj)
    for g in groups:
        members = [os.path.basename(str(m)) for m in g.members]
        group_members[g.id] = members
        group_merge_order[g.id] = list(g.merge_order or [])
        for m in members:
            basename_to_group[m] = g.id
    return basename_to_group, group_members, group_merge_order


def save_iteration_diagnostics(
    iter_dir: str,
    iteration: int,
    *,
    chain_data: Sequence[Mapping[str, Any]],
    old_masks: Mapping[str, np.ndarray],
    new_masks: Mapping[str, np.ndarray],
    gp_result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    coco_path: str | None = None,
    snname: str | None = None,
    gp_runs_dir: str | None = None,
    spec_class: Any | None = None,
    y_data_nonan: np.ndarray | None = None,
    y_data_nonan_err: np.ndarray | None = None,
    x1_data_norm: np.ndarray | None = None,
    x2_data_norm: np.ndarray | None = None,
    diag_groups: list[dict[str, Any]] | None = None,
    mangle_report: Mapping[str, Any] | None = None,
    t0_fix: float | None = None,
    mangled_dir: str | None = None,
    phot4mangling_path: str | None = None,
    filter_mjd_dict: Mapping[str, Mapping[str, float]] | None = None,
    avail_filters: Sequence[str] | None = None,
    filter_path: str | None = None,
    csp_sne: Sequence[str] = (),
    output_dir: str | None = None,
) -> None:
    del diag_groups, mangle_report, spec_class, y_data_nonan, y_data_nonan_err
    del x1_data_norm, x2_data_norm, gp_result, iteration
    if not _HAS_MPL:
        return
    import pipeline_config as pconf

    figs_root = os.path.join(iter_dir, "figs")
    _ensure_dir(figs_root)
    out_root = (output_dir or pconf.outputs_root(coco_path or pconf.COCO_PATH)).rstrip(os.sep) + os.sep
    sn = snname or pconf.SNNAME_DEFAULT

    use_ryan = bool(getattr(pconf, "ITER_GP_DIAG_FIGS", True))
    if use_ryan and gp_runs_dir:
        try:
            from iter_gp_style_diagnostics import run_iter_gp_style_diagnostics

            run_iter_gp_style_diagnostics(
                iter_dir,
                gp_runs_dir,
                figs_root,
                coco_path=coco_path or pconf.COCO_PATH,
                snname=sn,
                t0_fix=t0_fix,
                phot4mangling_path=phot4mangling_path,
                avail_filters=list(avail_filters or []),
            )
        except Exception:
            pass

    basename_to_group, group_members, group_merge_order = _load_group_maps(
        out_root, sn, coco_path
    )
    try:
        from iter_plot_suite import run_iter_plot_suite

        run_iter_plot_suite(
            iter_dir,
            chain_data=chain_data,
            old_masks=old_masks,
            new_masks=new_masks,
            mangled_dir=mangled_dir,
            phot4mangling_path=phot4mangling_path,
            filter_mjd_dict=filter_mjd_dict,
            avail_filters=avail_filters,
            filter_path=filter_path,
            snname=sn,
            csp_sne=csp_sne,
            basename_to_group=basename_to_group,
            group_members=group_members,
            group_merge_order=group_merge_order,
            metrics=metrics,
        )
    except Exception:
        pass


def write_diagnostics_summary(
    summary_dir: str,
    iterations: Sequence[Mapping[str, Any]],
    iter_root: str,
) -> None:
    _ensure_dir(summary_dir)
    iters = list(iterations)
    if _HAS_MPL and iters:
        ks = [int(m.get("iteration", i)) for i, m in enumerate(iters)]
        max_phot = [float(m.get("max_rel_phot_err", np.nan)) for m in iters]
        med_phot = [float(m.get("median_rel_phot_err", np.nan)) for m in iters]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, max_phot, "o-", label="max rel phot err")
        ax.plot(ks, med_phot, "s--", label="median rel phot err")
        ax.set_xlabel("iteration")
        ax.set_ylabel("relative error")
        ax.legend(fontsize=8)
        ax.set_title("Photometry closure vs iteration")
        fig.tight_layout()
        fig.savefig(os.path.join(summary_dir, "convergence_phot.png"), dpi=120)
        plt.close(fig)

        dmask = [float(m.get("delta_mask_rms", np.nan)) for m in iters]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, dmask, "o-", color="coral")
        ax.set_xlabel("iteration")
        ax.set_ylabel("max RMS Δmask")
        ax.set_title("Mask stability vs iteration")
        fig.tight_layout()
        fig.savefig(os.path.join(summary_dir, "convergence_mask_rms.png"), dpi=120)
        plt.close(fig)

    metrics_dir = os.path.join(iter_root, "metrics")
    subdirs = (
        "gp_surface",
        "gp_vs_mangled",
        "mangle_delta",
        "residuals",
        "phot_lc",
    )
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Iter GP+mangle diagnostics</title></head><body>",
        "<h1>Iterative GP+mangle summary</h1>",
    ]
    if os.path.isdir(metrics_dir):
        rel = os.path.relpath(metrics_dir, summary_dir)
        lines.append("<p><a href='%s'>Run metrics (chi2, lengthscales)</a></p>" % html.escape(rel))
    lines.append("<ul>")
    for m in iters:
        k = int(m.get("iteration", 0))
        figs = os.path.join(iter_root, "iter_%02d" % k, "figs")
        full_gp = os.path.join(iter_root, "iter_%02d" % k, "gp_runs", "full_gp")
        lines.append("<li><b>iter_%02d</b><ul>" % k)
        for sub in subdirs:
            subpath = os.path.join(figs, sub)
            if os.path.isdir(subpath):
                rel = os.path.relpath(subpath, summary_dir)
                lines.append(
                    "<li><a href='%s'>figs/%s/</a></li>" % (html.escape(rel), html.escape(sub))
                )
        summary_json = os.path.join(figs, "iteration_summary.json")
        if os.path.isfile(summary_json):
            rel = os.path.relpath(summary_json, summary_dir)
            lines.append(
                "<li><a href='%s'>iteration_summary.json</a></li>" % html.escape(rel)
            )
        if os.path.isdir(full_gp):
            rel = os.path.relpath(full_gp, summary_dir)
            lines.append(
                "<li><a href='%s'>full_gp spectra/</a></li>" % html.escape(rel)
            )
        lines.append("</ul></li>")
    lines.append("</ul></body></html>")
    with open(os.path.join(summary_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
