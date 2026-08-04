#!/usr/bin/env python3
"""Pipeline step 6: iterative GP + re-mangle loop."""

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
from iterative_gp_mangle import run_iterative_gp_mangle


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Iterative GP + mangling (pipeline step 6).")
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument("--coco-path", default=None)
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override ITER_GP_MANGLE_MAX_ITERS from pipeline_config.",
    )
    p.add_argument("--phot-convergence-frac", type=float, default=None)
    p.add_argument(
        "--bundle-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--seed-from-nb5",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--save-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--t0-fix",
        type=float,
        default=None,
        help="Explosion MJD anchor (default: SN_EXPLOSION_MJD).",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.coco_path:
        pconf.COCO_PATH = args.coco_path.rstrip("/") + os.sep

    sn = args.snname
    t0 = args.t0_fix if args.t0_fix is not None else pconf.explosion_date_mjd(sn)

    summary = run_iterative_gp_mangle(
        sn,
        coco_path=pconf.COCO_PATH,
        output_dir=pconf.outputs_root(pconf.COCO_PATH) + os.sep,
        t0_fix=t0,
        max_iters=args.max_iters,
        phot_convergence_frac=args.phot_convergence_frac,
        bundle_aware=args.bundle_aware,
        seed_from_nb5=args.seed_from_nb5,
        save_diagnostics=args.save_diagnostics,
        warm_start=args.warm_start,
        verbose=args.verbose,
    )
    print("Iter GP done:", summary.get("final_dir", summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
