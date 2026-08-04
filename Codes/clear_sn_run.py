#!/usr/bin/env python3
"""Clear pipeline outputs (and optionally prescale inputs) for a fresh SN rerun."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER = os.path.join(_SCRIPT_DIR, "helper_scripts")
for _p in (_SCRIPT_DIR, _HELPER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_config as pconf


def _validate_snname(snname: str) -> None:
    if not snname or snname in (".", ".."):
        raise ValueError("Invalid --snname: %r" % snname)


def collect_clear_paths(
    snname: str,
    *,
    coco_path: str,
    include_input_intermediates: bool = False,
) -> list[str]:
    _validate_snname(snname)
    coco = coco_path.rstrip("/") + os.sep
    out_sn = os.path.join(pconf.outputs_root(coco), snname)
    paths = [out_sn]
    if include_input_intermediates:
        paths.append(pconf.prescaled_spec_dir(coco, snname))
        paths.append(pconf.prescaled_spec_list_path(coco, snname))
    return paths


def clear_sn_run(
    snname: str,
    *,
    coco_path: str | None = None,
    include_input_intermediates: bool = False,
    dry_run: bool = False,
    yes: bool = False,
) -> list[str]:
    """Remove ``Outputs/<SN>/`` contents; optionally prescale intermediates."""
    coco = (coco_path or pconf.COCO_PATH).rstrip("/") + os.sep
    paths = collect_clear_paths(
        snname,
        coco_path=coco,
        include_input_intermediates=include_input_intermediates,
    )
    removed: list[str] = []
    out_sn = paths[0]

    if include_input_intermediates and not yes and not dry_run:
        print(
            "\n*** WARNING: --include-input-intermediates will DELETE prescaled "
            "spectra and list files under Inputs/.\n"
            "    Re-run step 4 (prescale) after clearing.\n"
            "    Pass --yes to confirm.\n",
            flush=True,
        )
        raise SystemExit(2)

    for path in paths:
        if not os.path.exists(path):
            continue
        if dry_run:
            print("[dry-run] would remove:", path, flush=True)
            removed.append(path)
            continue
        if os.path.isdir(path) and path == out_sn:
            for name in os.listdir(path):
                target = os.path.join(path, name)
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
                removed.append(target)
            os.makedirs(path, exist_ok=True)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
        else:
            os.remove(path)
            removed.append(path)

    return removed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clear Outputs/<SN>/ for a fresh pipeline run.",
    )
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument("--coco-path", default=None)
    p.add_argument(
        "--include-input-intermediates",
        action="store_true",
        help="Also delete prescaled spec dir and list (requires --yes).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths that would be removed.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation for --include-input-intermediates.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.coco_path:
        pconf.COCO_PATH = args.coco_path.rstrip("/") + os.sep
    removed = clear_sn_run(
        args.snname,
        coco_path=pconf.COCO_PATH,
        include_input_intermediates=bool(args.include_input_intermediates),
        dry_run=bool(args.dry_run),
        yes=bool(args.yes),
    )
    print("Cleared %d path(s) for %s" % (len(removed), args.snname))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
