#!/usr/bin/env python3
"""Pipeline step 5: log-space spectrum mangling."""

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
from mangle_spectra_log import run_mangle_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mangle spectra (pipeline step 5).")
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument("--coco-path", default=None)
    p.add_argument(
        "--bundle-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--save-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--save-epoch-plots",
        action="store_true",
        help="Save per-epoch mangling PDFs with photometry overlay.",
    )
    p.add_argument(
        "--kernel-divide",
        type=int,
        default=None,
        help="Matern32 length scale numerator when --mangle-kernel-mode=kernel_divide_scaled.",
    )
    p.add_argument(
        "--mangle-kernel-mode",
        choices=("fixed_5", "kernel_divide_scaled"),
        default=None,
        help="GP kernel lengthscale mode (default: pipeline_config MANGLE_GP_KERNEL_MODE).",
    )
    p.add_argument("--groups-json", default=None)
    p.add_argument(
        "--run-both-for-diag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run per-arm and bundle-aware compare diagnostics.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.coco_path:
        pconf.COCO_PATH = args.coco_path.rstrip("/") + os.sep
    if args.mangle_kernel_mode is not None:
        pconf.MANGLE_GP_KERNEL_MODE = args.mangle_kernel_mode
    if args.kernel_divide is not None:
        pconf.MANGLE_KERNEL_DIVIDE = int(args.kernel_divide)

    bundle = (
        bool(args.bundle_aware)
        if args.bundle_aware is not None
        else bool(pconf.MANGLE_BUNDLE_AWARE)
    )
    save_diag = (
        bool(args.save_diagnostics)
        if args.save_diagnostics is not None
        else bool(pconf.MANGLE_SAVE_DIAGNOSTICS)
    )
    run_both = (
        bool(args.run_both_for_diag)
        if args.run_both_for_diag is not None
        else bool(pconf.MANGLE_RUN_BOTH_FOR_DIAG)
    )

    summary = run_mangle_pipeline(
        args.snname,
        coco_path=pconf.COCO_PATH,
        output_dir=pconf.outputs_root(pconf.COCO_PATH) + os.sep,
        bundle_aware=bundle,
        groups_json=args.groups_json,
        save_diagnostics=save_diag,
        save_epoch_plots=args.save_epoch_plots,
        run_both_for_diag=run_both,
        kernel_divide=args.kernel_divide,
        verbose=args.verbose,
    )
    print("Mangle report:", summary.get("report_path"))
    print("Mangled dir:", summary.get("mangled_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
