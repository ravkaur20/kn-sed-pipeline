#!/usr/bin/env python3
"""Pipeline step 3: log-space LC GP fit."""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER = os.path.join(_SCRIPT_DIR, "helper_scripts")
for _p in (_SCRIPT_DIR, _HELPER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_config as pconf
from lc_gp_diagnostics import (
    save_per_filter_gp_plots,
    save_plot_gpfit,
    save_plot_gpfit_spec,
)
from lc_gp_fit import run_lc_gp_fit


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GP-fit light curves (pipeline step 3).")
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument(
        "--coco-path",
        default=None,
        help="Repo root (default: COCO_PATH env or auto-detect).",
    )
    p.add_argument(
        "--kernel-settings",
        default=None,
        help="Optional JSON file with per-band GP kernel overrides.",
    )
    p.add_argument(
        "--save-general-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write fittedGP_*.pdf summary plots (default: on).",
    )
    p.add_argument(
        "--save-per-filter-plots",
        action="store_true",
        help="Write one PDF per filter under lc_gp_per_filter/.",
    )
    p.add_argument(
        "--anchor-t0-in-lc-gp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pre-fit explosion anchor in GP training (default: pipeline_config).",
    )
    p.add_argument(
        "--append-t0-row",
        action="store_true",
        help="Append explosion anchor row to fitted_phot_logspace after fit.",
    )
    p.add_argument(
        "--exclude-filt",
        nargs="*",
        default=None,
        help="Override filter_plot_config.json exclude list.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.coco_path:
        pconf.COCO_PATH = args.coco_path.rstrip("/") + os.sep

    sn, result = run_lc_gp_fit(
        args.snname,
        coco_path=pconf.COCO_PATH,
        kernel_settings_path=args.kernel_settings,
        exclude_filt=args.exclude_filt,
        anchor_t0_in_lc_gp=args.anchor_t0_in_lc_gp,
        append_t0_row=args.append_t0_row,
        verbose=args.verbose,
    )

    rt = pconf.bootstrap_runtime(photometry_stage="extrapolated", snname=args.snname)
    if args.save_general_plots:
        save_plot_gpfit(sn, color_dict=rt.color_dict, mark_dict=rt.mark_dict)
        save_plot_gpfit_spec(sn, color_dict=rt.color_dict, mark_dict=rt.mark_dict)
    if args.save_per_filter_plots:
        save_per_filter_gp_plots(sn, color_dict=rt.color_dict, mark_dict=rt.mark_dict)

    print("Wrote:", result.fitted_phot_logspace_path)
    print("Wrote:", result.fitted_phot4mangling_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
