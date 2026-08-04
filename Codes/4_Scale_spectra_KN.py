#!/usr/bin/env python3
"""Pipeline step 4: prescale / group spectroscopic arms."""

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
from spectra_pre_scale import (
    run_prescale_pipeline,
    suggest_scale_groups,
    write_scale_groups_template,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prescale spectroscopy (pipeline step 4).")
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument("--coco-path", default=None)
    p.add_argument(
        "--write-groups-only",
        action="store_true",
        help="Only write/update spec_scale_groups JSON template; do not scale.",
    )
    p.add_argument("--groups-json", default=None, help="Override groups JSON path.")
    p.add_argument(
        "--output-mode",
        choices=("scale_only", "merge_join"),
        default=None,
    )
    p.add_argument("--same-time-minutes", type=float, default=None)
    p.add_argument(
        "--chain-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--gap-max-a", type=float, default=None)
    p.add_argument(
        "--write-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.coco_path:
        pconf.COCO_PATH = args.coco_path.rstrip("/") + os.sep

    coco = pconf.COCO_PATH
    out_root = pconf.outputs_root(coco)
    groups_path = args.groups_json or pconf.spec_scale_groups_json_path(out_root, args.snname)

    if args.write_groups_only:
        list_path = pconf.smoothed_spec_list_path(coco, args.snname)
        from spectra_pre_scale import load_spec_list

        entries = load_spec_list(list_path)
        if not entries:
            raise SystemExit("No spectra in %s" % list_path)
        groups = suggest_scale_groups(
            entries,
            same_time_minutes=args.same_time_minutes or pconf.SPEC_SCALE_SAME_TIME_MINUTES,
        )
        mode = args.output_mode or pconf.SPEC_SCALE_OUTPUT_MODE
        write_scale_groups_template(groups_path, groups, default_mode=mode)
        print("Wrote groups template:", groups_path)
        return 0

    kwargs: dict = {}
    if args.output_mode is not None:
        kwargs["default_output_mode"] = args.output_mode
    if args.same_time_minutes is not None:
        kwargs["same_time_minutes"] = args.same_time_minutes
    if args.chain_mode is not None:
        kwargs["chain_mode"] = args.chain_mode
    if args.gap_max_a is not None:
        kwargs["gap_max_a"] = args.gap_max_a
    if args.write_diagnostics is not None:
        kwargs["write_diagnostics"] = args.write_diagnostics

    report = run_prescale_pipeline(
        snname=args.snname,
        coco_path=coco,
        output_dir=out_root + os.sep,
        groups_json=args.groups_json,
        **kwargs,
    )
    print("Prescale done for", args.snname, "groups=", len(report.groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
