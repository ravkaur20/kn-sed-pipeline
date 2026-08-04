"""GP diagnostic GP diagnostic figures for one outer-loop iteration (subprocess, Agg)."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from typing import Optional


def run_iter_gp_style_diagnostics(
    iter_dir: str,
    gp_runs_dir: str,
    figs_root: str,
    *,
    coco_path: str,
    snname: str,
    t0_fix: float | None = None,
    phot4mangling_path: str | None = None,
    avail_filters: list[str] | None = None,
    twodim_gp_dir: Optional[str] = None,
    python_exe: Optional[str] = None,
) -> Optional[str]:
    """Run ``plot_results.py`` + ``plot_bands_gp_overview.py``; write under ``figs_root/``."""
    pred = os.path.join(gp_runs_dir, "predictions.npz")
    export_glob = os.path.join(gp_runs_dir, "gp_minimal_export", "*_meta.json")
    meta_files = glob.glob(export_glob)
    bundle_glob = os.path.join(gp_runs_dir, "gp_minimal_export", "gp_minimal_bundle.npz")
    if not os.path.isfile(pred) or not meta_files or not os.path.isfile(bundle_glob):
        return None

    import pipeline_config as pconf

    here = os.path.dirname(os.path.abspath(__file__))
    _ry = twodim_gp_dir or os.path.join(here, "twodim_gp")
    py = python_exe or sys.executable
    meta = meta_files[0]
    bundle = bundle_glob
    tag = "gp_runs"
    parent = os.path.abspath(iter_dir)
    cfg_json = os.path.join(gp_runs_dir, "config.json")
    gp_surface = os.path.join(figs_root, "gp_surface")
    phot_lc = os.path.join(figs_root, "phot_lc")
    os.makedirs(gp_surface, exist_ok=True)
    os.makedirs(phot_lc, exist_ok=True)
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"

    pr_cmd = [
        py,
        os.path.join(_ry, "plot_results.py"),
        "--tag",
        tag,
        "--bundle",
        bundle,
        "--meta",
        meta,
        "--output-dir",
        parent,
        "--heatmap-raw",
        "--heatmap-normalized",
    ]
    subprocess.run(pr_cmd, cwd=_ry, check=False, env=env)

    src_figs = os.path.join(gp_runs_dir, "figs")
    if os.path.isdir(src_figs):
        for fn in os.listdir(src_figs):
            sp = os.path.join(src_figs, fn)
            dp = os.path.join(gp_surface, fn)
            if os.path.isdir(sp):
                if os.path.isdir(dp):
                    shutil.rmtree(dp)
                shutil.copytree(sp, dp)
            else:
                shutil.copy2(sp, dp)
        shutil.rmtree(src_figs)

    enrich_path = os.path.join(gp_runs_dir, "gp_phot_enrich.npz")
    if phot4mangling_path and os.path.isfile(phot4mangling_path) and t0_fix is not None:
        try:
            import json

            import iter_gp_phot_enrich as ipe

            gn: dict = {}
            with open(meta, encoding="utf-8") as mf:
                meta_j = json.load(mf)
            if isinstance(meta_j.get("grid_norm_info"), dict):
                gn = meta_j["grid_norm_info"]
            ipe.build_phot_enrich_npz(
                bundle,
                enrich_path,
                phot4mangling_path=phot4mangling_path,
                grid_norm_info=gn,
                t0_fix=float(t0_fix),
                avail_filters=avail_filters or [],
            )
        except Exception:
            enrich_path = ""

    ob_cmd = [
        py,
        os.path.join(_ry, "plot_bands_gp_overview.py"),
        "--bundle",
        bundle,
        "--meta",
        meta,
        "--predictions",
        pred,
        "--output-dir",
        phot_lc,
        "--plot-residuals-vs-gp",
    ]
    if enrich_path and os.path.isfile(enrich_path):
        ob_cmd.extend(["--enrich", enrich_path])
    if os.path.isfile(cfg_json):
        ob_cmd.extend(["--run-config", cfg_json])
    subprocess.run(ob_cmd, cwd=_ry, check=False, env=env)

    plot_bundle_qa = bool(getattr(pconf, "PLOT_BUNDLE_SCALING_QA", False))
    for fn in list(os.listdir(phot_lc)):
        if "_pairs.png" not in fn:
            continue
        src = os.path.join(phot_lc, fn)
        if plot_bundle_qa:
            debug_dir = os.path.join(figs_root, "debug", "bundle_scaling")
            os.makedirs(debug_dir, exist_ok=True)
            shutil.move(src, os.path.join(debug_dir, fn))
        else:
            os.remove(src)

    return figs_root
