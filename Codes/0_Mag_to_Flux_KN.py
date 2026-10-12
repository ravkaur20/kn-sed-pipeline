#!/usr/bin/env python3
"""Pipeline step 0: AB magnitudes -> band-mean F_lambda raw photometry file.

Produces / maintains ``Inputs/Photometry/1_LCs_flux_raw/<SN>.dat``
(``MJD,Mag,Mag_err,Flux,Flux_err,band,Instr,FilterSet,MagSys``) with band-mean fluxes
computed from the magnitudes at each filter's **pivot wavelength**::

    Flux     = 3631 Jy * 10**(-0.4 m_AB) * c / lam_piv**2      (MagSys = AB, or no column)
    m_AB     = Mag + (m_AB - m_Vega)_band                          (MagSys = Vega)
    Flux_err = Flux * 0.4 ln(10) * Mag_err

This is the photon-weighted band-mean F_lambda that the mangling step and the synthetic
photometry (``true_ab``) use. Exactly one mode is required:

``--from-json PATH``
    Build the raw file from a compiled-photometry JSON (a list of records with keys such as
    ``MJD``, ``Telescope``, ``Instrument``, ``Filter``, ``Villar Mag``, ``Villar Mag_err``,
    ``AB/VEGA/Etc``). Magnitudes are used exactly as given, so any dust correction must be
    removed (or not) in the JSON beforehand. Each row's magnitude system is written to a
    ``MagSys`` column (``AB`` or ``Vega``); Vega magnitudes stay in Vega and their fluxes use
    the Vega zero point from the spectrum in ``Inputs/Standards/`` (``--vega-spectrum``).
    ``--drop-repeated`` drops rows marked ``R`` in ``Repeated obs``; ``--villar-used-only``
    keeps only rows Villar et al. used. A per-row report of kept/dropped rows is written to
    ``Outputs/<SN>/<SN>_step0_json_rows.csv``.
``--redo-fluxes``
    Keep an existing raw-format file as is and recompute only its ``Flux``/``Flux_err``
    columns from ``Mag``/``Mag_err``.
``--check``
    Verify an existing raw-format file without writing; exit 1 if any flux is stale.

Writing never replaces an existing file unless ``--overwrite`` is given. **Rename or back up
the old file first** (e.g. ``mv AT2017gfo.dat AT2017gfo_old.dat``).

Examples::

    python 0_Mag_to_Flux_KN.py --snname AT2017gfo --from-json KN_SED_DATA_dust_corrected.json
    python 0_Mag_to_Flux_KN.py --snname AT2017gfo --redo-fluxes --overwrite
    python 0_Mag_to_Flux_KN.py --snname AT2017gfo --check
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
        description="Pipeline step 0: build or update the raw photometry file with fluxes "
        "computed from AB magnitudes at each filter's pivot wavelength."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-json",
        metavar="PATH",
        help="Build the raw photometry file from a compiled-photometry JSON.",
    )
    mode.add_argument(
        "--redo-fluxes",
        action="store_true",
        help="Recompute only Flux/Flux_err of an existing raw-format .dat file (no format change).",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing raw-format .dat file; exit 1 if any Flux is stale. Never writes.",
    )
    p.add_argument("--snname", default=pconf.SNNAME_DEFAULT)
    p.add_argument(
        "--coco-path",
        default=None,
        help="Repo root (default: COCO_PATH env or auto-detect).",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Raw-format .dat to read for --redo-fluxes/--check "
        "(default: Inputs/Photometry/1_LCs_flux_raw/<SN>.dat).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Where to write (default: Inputs/Photometry/1_LCs_flux_raw/<SN>.dat).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file. Rename/back up the old file first!",
    )
    p.add_argument(
        "--filter-set",
        default="GeneralFilters",
        help="Filter folder under Inputs/Filters for --from-json (default: GeneralFilters).",
    )
    p.add_argument(
        "--villar-used-only",
        action="store_true",
        help="--from-json: keep only rows marked '*' in 'Used in Villar model?'.",
    )
    p.add_argument(
        "--drop-repeated",
        action="store_true",
        help="--from-json: drop rows marked 'R' in 'Repeated obs' (keeps everything else).",
    )
    p.add_argument(
        "--vega-spectrum",
        default=None,
        help="Vega spectrum giving the zero point for MagSys=Vega rows "
        "(default: Inputs/Standards/alpha_lyr_stis_006.fits).",
    )
    p.add_argument(
        "--mag-key",
        default=m2f.JsonColumns.mag,
        help="--from-json: JSON key holding the AB magnitude (default: %(default)r).",
    )
    p.add_argument(
        "--mag-err-key",
        default=m2f.JsonColumns.mag_err,
        help="--from-json: JSON key holding the magnitude error (default: %(default)r).",
    )
    p.add_argument(
        "--report",
        default=None,
        help="--from-json: per-row CSV report path "
        "(default: Outputs/<SN>/<SN>_step0_json_rows.csv).",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for --check (default: 1e-6).",
    )
    return p


def _refuse_overwrite(path: str) -> int:
    print(
        "\nRefusing to overwrite existing file:\n  %s\n"
        "Rename or back up the old file first so it is not lost, e.g.\n"
        "  mv '%s' '%s'\n"
        "then re-run, or pass --overwrite to replace it."
        % (path, path, os.path.splitext(path)[0] + "_old.dat"),
        file=sys.stderr,
    )
    return 2


def _vega_path(args, coco: str) -> str:
    return args.vega_spectrum or os.path.join(coco, m2f.DEFAULT_VEGA_SPECTRUM)


def _print_band_table(report, columns=None) -> pd.DataFrame:
    table = m2f.bands_report_table(report)
    with pd.option_context("display.max_rows", 500, "display.width", 120):
        print((table if columns is None else table[columns]).to_string(index=False))
    return table


def _from_json(args, coco: str, out_path: str) -> int:
    filters_root = pconf.filters_root(coco)
    cols = m2f.JsonColumns(mag=args.mag_key, mag_err=args.mag_err_key)
    records = m2f.load_photometry_json(args.from_json, cols)
    vega_spectrum = _vega_path(args, coco)
    raw, rows, bands = m2f.json_to_raw_photometry(
        records, filters_root, cols=cols, filter_set=args.filter_set,
        villar_used_only=args.villar_used_only, drop_repeated=args.drop_repeated,
        vega_spectrum=vega_spectrum,
    )
    print("JSON:", args.from_json)
    print("%d records -> %d rows kept (%d Vega, %d assumed AB), %d dropped"
          % (len(rows), len(raw), int((rows[m2f.MAG_SYS_COLUMN] == "Vega").sum()),
             int((rows["status"] == "kept_assumed_AB").sum()),
             int((rows["status"] == "dropped").sum())))
    dropped = rows[rows["status"] == "dropped"]
    if len(dropped):
        print("\nDropped rows by reason:")
        print(dropped["reason"].value_counts().to_string())
    kept = rows[rows["status"] != "dropped"]
    flagged = kept[kept["reason"].str.contains("assumed AB|case-insensitive|Vega magnitude")]
    if len(flagged):
        print("\nKept with warnings (check these):")
        cols_show = [cols.telescope, cols.instrument, cols.filter, "band", "reason"]
        reasons = flagged.assign(reason=flagged["reason"].str.replace(
            r";? ?same MJD and band as another kept row", "", regex=True))
        print(reasons.groupby(cols_show).size().rename("n_rows").to_string())
    n_dup = int(kept["reason"].str.contains("same MJD and band").sum())
    if n_dup:
        print("\nWARNING: %d kept rows share an MJD and band with another kept row (the same "
              "epoch reported by more than one source?). They are all written; see 'reason' "
              "in the row report to review them." % n_dup)
    if raw.empty:
        print("\nNo rows kept; nothing written.", file=sys.stderr)
        return 1
    print()
    _print_band_table(bands, columns=["band", "MagSys", "lam_piv_A", "n_rows"])

    report_path = args.report or os.path.join(
        pconf.outputs_root(coco), args.snname, "%s_step0_json_rows.csv" % args.snname
    )
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    rows.to_csv(report_path, index=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    m2f.write_raw_photometry(raw, out_path)
    print("\nWrote %s (%d rows)\nRow report: %s" % (out_path, len(raw), report_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    coco = (args.coco_path or pconf.COCO_PATH).rstrip("/") + os.sep
    raw_path = pconf.raw_photometry_path(coco, args.snname)
    out_path = args.output or raw_path

    if not args.check and os.path.exists(out_path) and not args.overwrite:
        return _refuse_overwrite(out_path)

    if args.from_json:
        return _from_json(args, coco, out_path)

    in_path = args.input or raw_path
    df = m2f.read_raw_photometry(in_path)
    new_df, report = m2f.convert_raw_photometry(df, pconf.filters_root(coco),
                                                vega_spectrum=_vega_path(args, coco))
    print("Raw photometry:", in_path)
    print("%d rows, %d bands" % (len(df), len(report)))
    table = _print_band_table(report)
    stale = table[~(table["max_rel_change"] <= args.rtol)]

    if args.check:
        if len(stale):
            print(
                "\nFAIL: %d band(s) have Flux inconsistent with the pivot-wavelength "
                "conversion (max rel. change > %g): %s\nRun with --redo-fluxes to fix."
                % (len(stale), args.rtol, ", ".join(stale["band"]))
            )
            return 1
        print("\nOK: Flux/Flux_err match the pivot-wavelength conversion of Mag (MagSys).")
        return 0

    m2f.write_raw_photometry(new_df, out_path)
    print("\nWrote %s (%d band(s) changed)." % (out_path, len(stale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
