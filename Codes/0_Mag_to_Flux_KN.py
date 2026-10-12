#!/usr/bin/env python3
"""Pipeline step 0: AB magnitudes -> band-mean F_lambda in the raw photometry file.

Recomputes the ``Flux`` / ``Flux_err`` columns of
``Inputs/Photometry/1_LCs_flux_raw/<SN>.dat`` from ``Mag`` / ``Mag_err`` (AB, in each
instrument's own filter system) at each filter's **pivot wavelength**::

    Flux     = 3631 Jy * 10**(-0.4 Mag) * c / lam_piv**2
    Flux_err = Flux * 0.4 ln(10) * Mag_err

This is the photon-weighted band-mean F_lambda that the mangling step and the synthetic
photometry (``true_ab``) use, so it must be run (or ``--check``-ed) before notebook 1 every
time the raw photometry or the filter curves change. All other columns are left unchanged.

Examples::

    python 0_Mag_to_Flux_KN.py --snname AT2017gfo --check   # verify only; exit 1 if stale
    python 0_Mag_to_Flux_KN.py --snname AT2017gfo           # rewrite Flux/Flux_err in place
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER = os.path.join(_SCRIPT_DIR, "helper_scripts")
for _p in (_SCRIPT_DIR, _HELPER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

import pipeline_config as pconf
import photometry_mag_to_flux as m2f


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recompute raw-photometry Flux/Flux_err from AB Mag at the filter pivot "
        "wavelength (pipeline step 0)."
    )
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument(
        "--coco-path",
        default=None,
        help="Repo root (default: COCO_PATH env or auto-detect).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit with status 1 if any Flux differs from the pivot conversion.",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for --check (default: 1e-6).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write to this path instead of overwriting the raw photometry file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    coco = (args.coco_path or pconf.COCO_PATH).rstrip("/") + os.sep
    raw_path = pconf.raw_photometry_path(coco, args.snname)
    filters_root = pconf.filters_root(coco)

    df = m2f.read_raw_photometry(raw_path)
    new_df, report = m2f.convert_raw_photometry(df, filters_root)
    table = m2f.bands_report_table(report)
    stale = table[~(table["max_rel_change"] <= args.rtol)]

    print("Raw photometry:", raw_path)
    print("%d rows, %d bands" % (len(df), len(table)))
    with pd.option_context("display.max_rows", 500, "display.width", 120):
        print(table.to_string(index=False))

    if args.check:
        if len(stale):
            print(
                "\nFAIL: %d band(s) have Flux inconsistent with the AB pivot-wavelength "
                "conversion (max rel. change > %g): %s\nRun without --check to fix."
                % (len(stale), args.rtol, ", ".join(stale["band"]))
            )
            return 1
        print("\nOK: Flux/Flux_err match the AB pivot-wavelength conversion.")
        return 0

    out_path = args.output or raw_path
    m2f.write_raw_photometry(new_df, out_path)
    print("\nWrote %s (%d band(s) changed)." % (out_path, len(stale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
