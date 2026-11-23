"""Export pure GP ``full_gp`` extended spectra (NB6 convention, no spliced products)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np

import GP2dim_utils as GP2dim
from gp_surface_extract import _log10_wls_from_x1_norm, _x2_mask_for_phase


def _write_spec_txt(out_dir: str, phase_log: float, fname_suffix: str, wls_a, flx, flxerr) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pth = os.path.join(out_dir, "%.6f%s" % (float(phase_log), fname_suffix))
    with open(pth, "w", encoding="utf-8") as fout:
        fout.write("#wls\tflux\tfluxerr\n")
        for w, f, ferr in zip(wls_a, flx, flxerr):
            fout.write("%E\t%E\t%E\n" % (float(w), float(f), float(ferr)))
    return pth


def linearize_gp_surface(
    spec_class,
    mu_fill: np.ndarray,
    std_fill: np.ndarray,
    y_data_nonan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gn = spec_class.grid_norm_info
    offset = float(gn["offset"])
    scale_factor = float(gn["scale_factor"])
    mu_conv = GP2dim.scaled_ln_to_linear(mu_fill, offset, scale_factor)
    std_conv = np.abs(scale_factor * mu_conv * std_fill)
    y_conv = GP2dim.scaled_ln_to_linear(y_data_nonan, offset, scale_factor)
    return mu_conv, std_conv, y_conv


def export_full_gp_spectra(
    spec_class,
    *,
    x1_fill: np.ndarray,
    x2_fill: np.ndarray,
    mu_fill: np.ndarray,
    std_fill: np.ndarray,
    grid_ext_columns: Sequence[float],
    y_data_nonan: np.ndarray,
    out_dir: str,
    mu_key: str | None = None,
) -> list[str]:
    """Write ``full_gp/{phase}_spec_extended*.txt`` (linear Å, linear flux). Returns paths."""
    full_gp_dir = os.path.join(out_dir, "full_gp")
    os.makedirs(full_gp_dir, exist_ok=True)

    mu_conv, std_conv, _y_conv = linearize_gp_surface(
        spec_class, np.asarray(mu_fill, dtype=float), np.asarray(std_fill, dtype=float), y_data_nonan
    )

    gn = spec_class.grid_norm_info
    written: list[str] = []

    # Every prediction column is written, including the epochs of observed spectra.
    # Near-identical copies (e.g. X-shooter arms of one night) become one file.
    for mj in distinct_phase_columns(grid_ext_columns):
        mask = _x2_mask_for_phase(x2_fill, mj, gn)
        x1_sel = np.asarray(x1_fill[mask], dtype=float)
        # copies ~1e-12 apart share the mask; keep one row per wavelength
        _, keep = np.unique(np.round(x1_sel, 9), return_index=True)
        wls_log = _log10_wls_from_x1_norm(x1_sel[keep], gn)
        smooth_ext_spec = np.asarray(mu_conv[mask], dtype=float)[keep]
        smooth_ext_spec_err = np.asarray(std_conv[mask], dtype=float)[keep]
        if wls_log.size == 0:
            continue
        written.append(
            _write_spec_txt(
                full_gp_dir,
                mj,
                "_spec_extended_FL.txt",
                np.power(10.0, wls_log),
                smooth_ext_spec,
                smooth_ext_spec_err,
            )
        )

    return written


# Columns closer than this in log10(phase days) are the same epoch (1e-5 dex ~ 0.0002 d at 8 d).
PHASE_COLUMN_MERGE_ATOL_LOG10 = 1e-5


def distinct_phase_columns(columns, atol: float = PHASE_COLUMN_MERGE_ATOL_LOG10) -> np.ndarray:
    """Sorted log10-phase columns with near-identical copies collapsed to the first one."""
    c = np.sort(np.asarray(columns, dtype=float).ravel())
    c = c[np.isfinite(c)]
    out: list[float] = []
    for v in c:
        if not out or v - out[-1] > float(atol):
            out.append(float(v))
    return np.asarray(out, dtype=float)


def copy_full_gp_tree(src_dir: str, dst_dir: str) -> None:
    """Copy ``*.txt`` from ``src_dir/full_gp`` (or ``src_dir`` if flat) into ``dst_dir``."""
    src = src_dir
    if os.path.isdir(os.path.join(src_dir, "full_gp")):
        src = os.path.join(src_dir, "full_gp")
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src):
        return
    for fn in os.listdir(src):
        if fn.endswith(".txt"):
            src_p = os.path.join(src, fn)
            dst_p = os.path.join(dst_dir, fn)
            with open(src_p, "rb") as rf, open(dst_p, "wb") as wf:
                wf.write(rf.read())
