"""Helpers for 7.5_comparison_check_log: phase-stem FINAL spectra, log10 flux on disk, MJD indexing.

TwoDim FINAL / extended filenames use a numeric stem (often GP ``Log_Phase``, not calendar MJD).
Calendar-MJD stems (legacy linear notebooks) are detected when ``abs(stem)`` is very large.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Tuple

import numpy as np

import rimangle_log_spectrum as rml

__all__ = [
    "FINAL_SUFFIXES",
    "ITER_GP_SPEC_SUFFIXES",
    "SPECTRUM_STEM_SUFFIXES",
    "parse_final_stem",
    "parse_spectrum_stem",
    "stem_looks_like_calendar_mjd",
    "t0_fix_from_late_lc",
    "stem_to_spec_mjd",
    "resolve_final_directory",
    "resolve_final_directory_legacy",
    "resolve_iter_gp_directory",
    "twodim_final_branch",
    "read_final_spectrum_linear",
    "deduplicate_wavelength_flux",
    "create_lookup_table",
    "nearest_final_spectrum_native",
    "list_final_spectra_native_rows",
    "index_final_native_files",
    "default_list_file_for_mode",
    "default_spec_dir_for_mode",
    "collect_input_spectra_for_mode",
    "resolve_comparison_spectrum_path",
    "load_comparison_spectrum_xy",
    "lookup_index_is_mjd",
    "lookup_spec_mjd_axis",
    "times_for_lookup_plot",
    "augment_spectra_list_explosion_mjd",
    "dense_plot_axis_log_days",
]

FINAL_SUFFIXES: Tuple[str, ...] = (
    "_FINAL_spec_FL.txt",
    "_FINAL_spec_SMOOTH.txt",
    "_FINAL_spec_SNF.txt",
    "_FINAL_spec.txt",
)

ITER_GP_SPEC_SUFFIXES: Tuple[str, ...] = (
    "_spec_extended_FL.txt",
    "_spec_extended.txt",
)

SPECTRUM_STEM_SUFFIXES: Tuple[str, ...] = FINAL_SUFFIXES + ITER_GP_SPEC_SUFFIXES

FluxOnDisk = Literal["auto", "linear", "log10"]


def deduplicate_wavelength_flux(
    wavelength: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge duplicate / near-duplicate wavelength bins (mean flux, mean fluxerr). Synphot needs unique λ."""
    w = np.asarray(wavelength, dtype=float).ravel()
    f = np.asarray(flux, dtype=float).ravel()
    if fluxerr is None:
        e = np.zeros_like(f)
    else:
        e = np.asarray(fluxerr, dtype=float).ravel()
    m = np.isfinite(w) & np.isfinite(f)
    w, f, e = w[m], f[m], e[m]
    if w.size == 0:
        return w, f, e
    order = np.argsort(w)
    w, f, e = w[order], f[order], e[order]
    out_w: list[float] = []
    out_f: list[float] = []
    out_e: list[float] = []
    cw = float(w[0])
    acc_f: list[float] = [float(f[0])]
    acc_e: list[float] = [float(e[0])]
    for k in range(1, w.size):
        wk = float(w[k])
        if np.isclose(wk, cw, rtol=rtol, atol=atol):
            acc_f.append(float(f[k]))
            acc_e.append(float(e[k]))
        else:
            out_w.append(cw)
            out_f.append(float(np.mean(acc_f)))
            out_e.append(float(np.mean(acc_e)))
            cw = wk
            acc_f = [float(f[k])]
            acc_e = [float(e[k])]
    out_w.append(cw)
    out_f.append(float(np.mean(acc_f)))
    out_e.append(float(np.mean(acc_e)))
    return (
        np.asarray(out_w, dtype=float),
        np.asarray(out_f, dtype=float),
        np.asarray(out_e, dtype=float),
    )


def parse_spectrum_stem(filename: str) -> float:
    """Parse phase/MJD stem from FINAL or iter ``_spec_extended`` filenames."""
    name = os.path.basename(filename)
    for suf in SPECTRUM_STEM_SUFFIXES:
        if name.endswith(suf):
            return float(name[: -len(suf)])
    raise ValueError("Unrecognized spectrum filename: %r" % name)


def parse_final_stem(filename: str) -> float:
    return parse_spectrum_stem(filename)


def stem_looks_like_calendar_mjd(stem: float, threshold: float = 40_000.0) -> bool:
    """True if the stem is almost certainly an absolute MJD (legacy naming)."""
    return abs(float(stem)) >= float(threshold)


def t0_fix_from_late_lc(datalc_path: str, snname: str) -> float:
    """Same as 4_LCfit_KN_log / rimangle: ``t0_fix = MJD[0] - Phase[0]`` on late-extrap LC."""
    p = os.path.join(datalc_path, "%s.dat" % snname)
    d = np.genfromtxt(p, dtype=None, encoding="utf-8", names=True)
    return float(d["MJD"][0] - d["Phase"][0])


def stem_to_spec_mjd(
    stem: float,
    coco_path: str,
    snname: str,
    *,
    datalc_path: str | None = None,
    voronoi_half: float = 0.5,
) -> float:
    """Map a TwoD filename stem to spectrum MJD via ``fitted_phot_logspace`` (full-precision ``Log_Phase``).

    The stem must be **unambiguously** closest to one grid row: not midway between two ``Log_Phase``
    values, and within ``voronoi_half * (min grid step)`` of that row. Table phases are **not** rounded;
    MJD uses ``t0_fix + 10**Log_Phase`` with the exact ``Log_Phase`` float from the matched row.
    """
    stem = float(stem)
    if stem_looks_like_calendar_mjd(stem):
        return stem
    if datalc_path is None:
        datalc_path = os.path.join(
            coco_path, "Inputs", "Photometry", "3_LCs_extrapolated"
        )
    t0f = t0_fix_from_late_lc(datalc_path, snname)
    lpath = os.path.join(
        coco_path, "Outputs", snname, "fitted_phot_logspace_%s.dat" % snname
    )
    if not os.path.isfile(lpath):
        raise FileNotFoundError("Missing logspace photometry table: %s" % lpath)
    lp = np.genfromtxt(lpath, names=True, delimiter="\t", encoding="utf-8")
    if lp.size == 0:
        raise ValueError("empty fitted_phot_logspace table: %s" % lpath)
    names = lp.dtype.names
    if names is None or "Log_Phase" not in names:
        raise ValueError("fitted_phot_logspace file has no Log_Phase column: %s" % lpath)
    pv = np.atleast_1d(np.asarray(lp["Log_Phase"], dtype=np.float64))
    pos = _unambiguous_nearest_log_phase_index(float(stem), pv, voronoi_half=float(voronoi_half))
    lpv = float(pv[pos])
    return float(t0f + 10.0**lpv)


def _unambiguous_nearest_log_phase_index(
    stem: float,
    pv: np.ndarray,
    *,
    voronoi_half: float = 0.5,
) -> int:
    """Index of the unique nearest ``Log_Phase`` to ``stem``; raise if ties or stem too far from grid."""
    pv = np.asarray(pv, dtype=np.float64).ravel()
    stem = float(np.float64(stem))
    if pv.size == 0:
        raise ValueError("empty Log_Phase column")
    distances = np.abs(pv - stem)
    pos = int(np.argmin(distances))
    d_nearest = float(distances[pos])
    if pv.size >= 2:
        if pos == 0:
            alt = distances[1:]
        elif pos == len(distances) - 1:
            alt = distances[:-1]
        else:
            alt = np.concatenate([distances[:pos], distances[pos + 1 :]])
        d_second = float(np.min(alt))
        eps = 1e-12 * (1.0 + abs(stem))
        if d_second <= d_nearest + eps:
            raise ValueError(
                "Ambiguous Log_Phase for stem=%r: distance to two grid rows is nearly equal "
                "(|Δ|≈%s vs %s). Use the stem that matches one row of fitted_phot_logspace."
                % (stem, d_nearest, d_second)
            )
    u = np.unique(pv)
    if u.size >= 2:
        min_step = float(np.min(np.diff(np.sort(u))))
        limit = float(voronoi_half) * min_step + 1e-9
        if d_nearest > limit:
            raise ValueError(
                "stem=%r is too far from the nearest Log_Phase row (|stem−Log_Phase|=%.6g; "
                "for this grid require < %.6g). Nearest Log_Phase=%.17g"
                % (stem, d_nearest, limit, float(pv[pos]))
            )
    return pos


def resolve_final_directory(
    coco_path: str,
    snname: str,
    final_variant: str,
    *,
    twodim_branch: str | None = None,
    extension_type: str = "2dim",
) -> str:
    """
    ``final_variant``:
      "" or "default" -> base FINAL directory (optionally under ``twodim_branch``)
      else -> that subdirectory (e.g. ``as_observed``).

    Production layout: ``FINAL_spectra_2dim/as_observed/`` (``twodim_branch`` ignored).
    Legacy nested branches (``twodim_iter/extrapolate/full_gp``, etc.) supported only
    when explicitly passed via ``twodim_branch`` for old Outputs trees.
    """
    import pipeline_config as pconf

    if twodim_branch is None:
        return pconf.final_spectra_qa_dir(
            pconf.outputs_root(coco_path), snname, variant=final_variant
        )
    base = os.path.join(coco_path, "Outputs", snname, "FINAL_spectra_%s" % extension_type)
    base = os.path.join(base, *str(twodim_branch).replace("\\", "/").split("/"))
    if not final_variant or final_variant in ("default",):
        return base
    return os.path.join(base, final_variant)


def twodim_final_branch(mode_short: str, product: str) -> str:
    """Legacy path segment for old nested FINAL layouts."""
    return "%s/%s" % (mode_short.replace("\\", "/").strip("/"), product.replace("\\", "/").strip("/"))


def resolve_final_directory_legacy(
    coco_path: str,
    snname: str,
    final_variant: str,
) -> str | None:
    """Find FINAL spectra dir across flat and legacy nested layouts."""
    import pipeline_config as pconf

    candidates = [
        pconf.final_spectra_qa_dir(pconf.outputs_root(coco_path), snname, variant=final_variant),
        os.path.join(
            coco_path, "Outputs", snname, "FINAL_spectra_2dim", final_variant
        ),
        resolve_final_directory(
            coco_path, snname, final_variant,
            twodim_branch="twodim_iter/extrapolate/full_gp",
        ),
        resolve_final_directory(
            coco_path, snname, final_variant,
            twodim_branch="twodim_iter/extrapolate/spliced",
        ),
    ]
    for d in candidates:
        if os.path.isdir(d) and any(fn.endswith(".txt") for fn in os.listdir(d)):
            return d
    return None


def resolve_iter_gp_directory(
    coco_path: str,
    snname: str,
    mode: str,
    *,
    product: str = "full_gp",
    output_dir: str | None = None,
) -> str:
    """Runtime iter final spectra: ``Outputs/<SN>/twodim_iter/<mode>/final/<product>/``."""
    import pipeline_config as pconf

    out = output_dir or os.path.join(coco_path, "Outputs")
    return pconf.twodim_iter_final_spectra_dir(out, snname, mode, product=product)


def read_final_spectrum_linear(
    path: str,
    flux_on_disk: FluxOnDisk = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    wl = np.asarray(raw[:, 0], dtype=float)
    fl = np.asarray(raw[:, 1], dtype=float)
    fe = np.asarray(raw[:, 2], dtype=float) if raw.shape[1] > 2 else np.zeros_like(fl)

    if flux_on_disk == "auto":
        flux_in_log10 = rml.auto_extended_flux_is_log10(wl, fl)
    elif flux_on_disk == "log10":
        flux_in_log10 = True
    elif flux_on_disk == "linear":
        flux_in_log10 = False
    else:
        raise ValueError("flux_on_disk must be 'auto', 'linear', or 'log10'")

    dt: Any = [("wls", "f8"), ("flux", "f8"), ("fluxerr", "f8")]
    ext = np.empty(len(wl), dtype=dt)
    ext["wls"] = wl
    ext["flux"] = fl
    ext["fluxerr"] = fe
    lin = rml.extended_to_linear_recarray(ext, flux_in_log10)
    wl_out, fl_out, fe_out = deduplicate_wavelength_flux(
        lin["wls"], lin["flux"], lin["fluxerr"]
    )
    return wl_out, fl_out, fe_out


def _enumerate_final_spectra_native_rows(
    data_dir: str,
    coco_path: str,
    snname: str,
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
) -> list[tuple[float, float, np.ndarray, np.ndarray, str]]:
    """Sorted list of ``(spec_mjd, stem_float, wl_Å, fl_linear, fname)`` for FINAL .txt files."""
    spectra_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if final_suffixes:
        spectra_files = [f for f in spectra_files if any(f.endswith(s) for s in final_suffixes)]
    if not spectra_files:
        raise FileNotFoundError(
            "No matching .txt spectra in %s (final_suffixes=%r)" % (data_dir, final_suffixes)
        )

    rows: list[tuple[float, float, np.ndarray, np.ndarray, str]] = []
    for fname in spectra_files:
        stem_float = parse_final_stem(fname)
        smjd = stem_to_spec_mjd(
            stem_float, coco_path, snname, datalc_path=datalc_path
        )
        path = os.path.join(data_dir, fname)
        wl, fl, _fe = read_final_spectrum_linear(path, flux_on_disk=flux_on_disk)
        m = np.isfinite(wl) & np.isfinite(fl)
        wl, fl = wl[m], fl[m]
        if wl.size == 0:
            continue
        order = np.argsort(wl)
        wl, fl = wl[order], fl[order]
        rows.append((float(smjd), float(stem_float), wl, fl, fname))

    if not rows:
        raise ValueError("No valid spectra loaded from %s" % data_dir)
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def nearest_final_spectrum_native(
    data_dir: str,
    mjd_query: float,
    coco_path: str,
    snname: str,
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """
    Load the FINAL spectrum whose ``spec_mjd`` is closest to ``mjd_query``, on its **native**
    wavelength grid (no ``linspace`` / lookup-table interpolation).

    Returns
    -------
    spec_mjd, wl, fl, fname
        Observer-frame MJD, wavelength Å, linear F_lambda, basename of the file.
    """
    rows = _enumerate_final_spectra_native_rows(
        data_dir,
        coco_path,
        snname,
        flux_on_disk=flux_on_disk,
        datalc_path=datalc_path,
        final_suffixes=final_suffixes,
    )
    spec_mjds = np.asarray([r[0] for r in rows], dtype=float)
    pos = int(np.argmin(np.abs(spec_mjds - float(mjd_query))))
    smjd, _stem, wl, fl, fname = rows[pos]
    return float(smjd), wl.copy(), fl.copy(), fname


def list_final_spectra_native_rows(
    data_dir: str,
    coco_path: str,
    snname: str,
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
) -> list[tuple[float, float, np.ndarray, np.ndarray, str]]:
    """All FINAL spectra in ``data_dir``: ``(spec_mjd, stem_float, wl_Å, fl_linear, fname)``, sorted."""
    return _enumerate_final_spectra_native_rows(
        data_dir,
        coco_path,
        snname,
        flux_on_disk=flux_on_disk,
        datalc_path=datalc_path,
        final_suffixes=final_suffixes,
    )


def index_final_native_files(
    data_dir: str,
    coco_path: str,
    snname: str,
    *,
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
) -> list[tuple[float, float, str]]:
    """Lightweight index: ``(spec_mjd, stem_float, fname)`` sorted, without loading flux."""
    try:
        spectra_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    except FileNotFoundError:
        raise FileNotFoundError("No such FINAL directory: %s" % data_dir) from None
    if final_suffixes:
        spectra_files = [f for f in spectra_files if any(f.endswith(s) for s in final_suffixes)]
    if not spectra_files:
        raise FileNotFoundError(
            "No matching .txt spectra in %s (final_suffixes=%r)" % (data_dir, final_suffixes)
        )
    rows: list[tuple[float, float, str]] = []
    for fname in spectra_files:
        stem_float = parse_final_stem(fname)
        smjd = stem_to_spec_mjd(
            stem_float, coco_path, snname, datalc_path=datalc_path
        )
        rows.append((float(smjd), float(stem_float), fname))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def default_list_file_for_mode(mode: str, snname: str, coco_path: str) -> str | None:
    """Spectrum list path under Inputs/Spectroscopy for original/smoothed."""
    mode = str(mode).lower()
    if mode == "smoothed":
        return os.path.join(
            coco_path, "Inputs", "Spectroscopy", "2_spec_lists_smoothed", "%s.list" % snname
        )
    if mode == "original":
        return os.path.join(
            coco_path, "Inputs", "Spectroscopy", "1_spec_lists_original", "%s.list" % snname
        )
    return None


def default_spec_dir_for_mode(mode: str, snname: str, coco_path: str) -> str:
    mode = str(mode).lower()
    if mode == "smoothed":
        return os.path.join(coco_path, "Inputs", "Spectroscopy", "2_spec_smoothed")
    if mode == "original":
        return os.path.join(coco_path, "Inputs", "Spectroscopy", "1_spec_original", snname)
    if mode == "mangled":
        return os.path.join(coco_path, "Outputs", snname, "mangled_spectra")
    raise ValueError("mode must be 'original', 'smoothed', or 'mangled'")


def collect_input_spectra_for_mode(
    mode: str,
    list_file: str | None,
    original_spec_dir: str | None,
    snname: str,
    coco_path: str,
) -> tuple[list[str], np.ndarray, str, str | None]:
    """
    Build parallel lists of spectrum paths (relative or absolute) and MJDs.

    For mode='mangled' with list_file missing or not a file, scans original_spec_dir for *.txt.
    Returns ``(orig_paths, orig_mjds, original_spec_dir, list_ref)``.
    """
    mode = str(mode).lower()
    if mode not in ("original", "smoothed", "mangled"):
        raise ValueError("mode must be 'original', 'smoothed', or 'mangled'")

    if original_spec_dir is None:
        original_spec_dir = default_spec_dir_for_mode(mode, snname, coco_path)

    orig_paths: list[str] = []
    orig_mjds_list: list[float] = []
    list_ref: str | None = list_file

    if mode == "mangled":
        lf = list_file
        use_scan = (lf is None) or (not os.path.isfile(os.path.expanduser(str(lf))))
        if use_scan:
            list_ref = None
            if not os.path.isdir(original_spec_dir):
                return orig_paths, np.asarray([], dtype=float), original_spec_dir, list_ref
            basenames = sorted(f for f in os.listdir(original_spec_dir) if f.endswith(".txt"))
            orig_paths = basenames
            for fname in basenames:
                try:
                    orig_mjds_list.append(float(str(fname).split("_")[0]))
                except Exception:
                    orig_mjds_list.append(float("nan"))
        else:
            list_ref = lf
            with open(lf) as fp:
                for line in fp:
                    if line.strip() == "" or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    try:
                        orig_paths.append(parts[0])
                        orig_mjds_list.append(float(parts[-1]))
                    except Exception:
                        continue
    elif mode == "smoothed":
        if list_file is None:
            list_file = default_list_file_for_mode("smoothed", snname, coco_path)
        list_ref = list_file
        list_file = str(list_file).replace("1_spec_lists_original", "2_spec_lists_smoothed")
        with open(list_file) as fp:
            for line in fp:
                if line.strip() == "" or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    orig_mjds_list.append(float(parts[0]))
                    orig_paths.append(parts[2])
                except Exception:
                    continue
    else:
        if list_file is None:
            list_file = default_list_file_for_mode("original", snname, coco_path)
        list_ref = list_file
        with open(list_file) as fp:
            for line in fp:
                if line.strip() == "" or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    orig_paths.append(parts[0])
                    orig_mjds_list.append(float(parts[-1]))
                except Exception:
                    continue

    orig_mjds = np.asarray(orig_mjds_list, dtype=float)
    return orig_paths, orig_mjds, original_spec_dir, list_ref


def resolve_comparison_spectrum_path(
    orig_path: str,
    mode: str,
    original_spec_dir: str,
    coco_path: str,
    snname: str,
) -> str:
    """Map list path entries to a local on-disk path (handles legacy /data/ prefixes)."""
    p = str(orig_path)
    if p.startswith("/data/2_spec_smoothed/"):
        local_base = os.path.join(coco_path, "Inputs", "Spectroscopy", "2_spec_smoothed")
        return os.path.join(local_base, p.replace("/data/2_spec_smoothed/", "").lstrip("/"))
    if p.startswith("/data/1_spec_original/"):
        local_base = os.path.join(coco_path, "Inputs", "Spectroscopy", "1_spec_original", snname)
        return os.path.join(local_base, p.replace("/data/1_spec_original/", "").lstrip("/"))
    if os.path.isabs(p):
        return os.path.expanduser(p)
    return os.path.join(os.path.expanduser(original_spec_dir), p)


def load_comparison_spectrum_xy(
    path: str,
    mode: str,
    flux_on_disk: FluxOnDisk,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear F_lambda and wavelength (Å); mangled uses ``read_final_spectrum_linear``."""
    if str(mode).lower() == "mangled":
        wl, fl, _ = read_final_spectrum_linear(path, flux_on_disk=flux_on_disk)
        return np.asarray(wl, dtype=float), np.asarray(fl, dtype=float)
    d = np.loadtxt(path)
    return np.asarray(d[:, 0], dtype=float), np.asarray(d[:, 1], dtype=float)


def create_lookup_table(
    data_dir: str,
    coco_path: str,
    snname: str,
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    wavelength_range: tuple[float, float] | None = None,
    wavelength_bins: int = 10_000,
    final_suffixes: tuple[str, ...] | None = None,
    prepend_explosion_mjd: float | None = None,
    prepend_flux_floor_linear: float = 1e-99,
) -> tuple[Any, np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    """
    Load FINAL spectra from ``data_dir``, convert flux to linear F_lambda, interpolate to a common
    grid.

    **Index:** default integer row index (one row per file). Physical MJDs are
    ``lookup_table.attrs['spec_mjd']``; filename phase keys are ``attrs['phase_stem']``.
    Several files can share the same ``spec_mjd``; they are **not** averaged.

    ``final_suffixes``: if set (e.g. ``("_FINAL_spec_FL.txt",)``), only those products are loaded.

    ``prepend_explosion_mjd``: if set (MJD), prepend one synthetic row at that time with constant
    ``prepend_flux_floor_linear`` F_λ on ``common_wavelengths``, **only if** the earliest loaded
    spectrum is strictly later. Use ``t0_fix`` for explosion time. Intended for synphot / 7.5 LC
    plots (not real data).

    Returns
    -------
    lookup_table, spec_mjds, common_wavelengths, spectra_list
        ``spec_mjds`` aligns with rows (physical MJD per spectrum). ``spectra_list`` entries are
        ``(spec_mjd, wavelength_Å, linear_flux)`` on ``common_wavelengths``.
    """
    import pandas as pd

    rows = _enumerate_final_spectra_native_rows(
        data_dir,
        coco_path,
        snname,
        flux_on_disk=flux_on_disk,
        datalc_path=datalc_path,
        final_suffixes=final_suffixes,
    )
    spec_mjds_list = [float(r[0]) for r in rows]
    stems_list = [float(r[1]) for r in rows]
    wavelengths = [r[2] for r in rows]
    fluxes_linear = [r[3] for r in rows]

    all_wavelengths = np.concatenate(wavelengths)
    wl_min = (
        float(wavelength_range[0])
        if wavelength_range
        else float(np.nanmin(all_wavelengths))
    )
    wl_max = (
        float(wavelength_range[1])
        if wavelength_range
        else float(np.nanmax(all_wavelengths))
    )
    common_wavelengths = np.linspace(wl_min, wl_max, int(wavelength_bins))

    fluxes_interpolated = []
    for wl, fl in zip(wavelengths, fluxes_linear):
        flux_interp = np.interp(
            common_wavelengths, wl, fl, left=0.0, right=0.0
        )
        fluxes_interpolated.append(flux_interp)

    if prepend_explosion_mjd is not None and spec_mjds_list:
        pe = float(prepend_explosion_mjd)
        if float(min(spec_mjds_list)) > pe + 1e-9:
            zero_row = np.full(int(wavelength_bins), float(prepend_flux_floor_linear), dtype=float)
            spec_mjds_list = [pe] + spec_mjds_list
            stems_list = [float("nan")] + stems_list
            fluxes_interpolated = [zero_row] + fluxes_interpolated

    fluxes_interpolated = np.asarray(fluxes_interpolated, dtype=float)
    lookup_table = pd.DataFrame(fluxes_interpolated, columns=common_wavelengths)
    spec_mjd_arr = np.asarray(spec_mjds_list, dtype=float)
    lookup_table.attrs["spec_mjd"] = spec_mjd_arr
    lookup_table.attrs["phase_stem"] = np.asarray(stems_list, dtype=float)
    lookup_table.attrs["snname"] = snname

    spectra_list = [
        (
            float(spec_mjd_arr[i]),
            common_wavelengths.copy(),
            fluxes_interpolated[i].astype(float),
        )
        for i in range(len(spec_mjd_arr))
    ]

    return lookup_table, spec_mjd_arr, common_wavelengths, spectra_list


def augment_spectra_list_explosion_mjd(
    spectra_list: list,
    explosion_mjd: float,
    common_wavelengths: np.ndarray,
    *,
    flux_floor_linear: float = 1e-99,
    only_if_before_first: bool = True,
) -> list:
    """Prepend one epoch at ``explosion_mjd`` with constant F_lambda = ``flux_floor_linear``.

    If ``only_if_before_first`` and the earliest spectrum MJD is already at or before
    ``explosion_mjd``, return ``spectra_list`` unchanged (shallow copied list).
    """
    if not spectra_list:
        return list(spectra_list)
    times = [float(t) for (t, _, _) in spectra_list]
    t_min = min(times)
    if only_if_before_first and t_min <= float(explosion_mjd) + 1e-9:
        return list(spectra_list)
    wl = np.asarray(common_wavelengths, dtype=float)
    fl = np.full_like(wl, float(flux_floor_linear), dtype=float)
    head: tuple[float, np.ndarray, np.ndarray] = (float(explosion_mjd), wl.copy(), fl)
    return [head] + list(spectra_list)


def dense_plot_axis_log_days(
    syn_mjd: np.ndarray,
    syn_values: np.ndarray,
    mjd0: float,
    *,
    n_points: int = 256,
    min_pos_days: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ``syn_values`` vs log10(days since ``mjd0``) for smoother **plot** lines only.

    Days axis is clamped to ``min_pos_days`` so log10 is finite. Returns
    ``(days_since_mjd0, values_interp)``; edge values outside the data range are NaN from ``np.interp``.
    """
    t_rel = np.asarray(syn_mjd, dtype=float) - float(mjd0)
    v = np.asarray(syn_values, dtype=float)
    m = np.isfinite(t_rel) & np.isfinite(v)
    t_rel, v = t_rel[m], v[m]
    if t_rel.size < 2:
        return t_rel, v
    order = np.argsort(t_rel)
    t_rel, v = t_rel[order], v[order]
    pos = t_rel[t_rel > 0]
    lo = float(min_pos_days)
    if pos.size:
        lo = max(lo, float(np.nanmin(pos)))
    hi = float(np.nanmax(t_rel))
    if hi <= lo:
        return t_rel, v
    u = np.logspace(np.log10(lo), np.log10(hi), int(max(2, n_points)))
    out = np.interp(u, t_rel, v, left=np.nan, right=np.nan)
    return u, out


def times_for_lookup_plot(lookup_table: Any) -> np.ndarray:
    """1D time coordinate for plotting (MJD if available, else index)."""
    return lookup_spec_mjd_axis(lookup_table)


def lookup_spec_mjd_axis(lookup_table: Any) -> np.ndarray:
    """Observer MJD per row (for synphot / light curves). Falls back to index if missing."""
    if hasattr(lookup_table, "attrs") and "spec_mjd" in lookup_table.attrs:
        return np.asarray(lookup_table.attrs["spec_mjd"], dtype=float)
    return np.asarray(lookup_table.index, dtype=float)


def lookup_index_is_mjd(lookup_table: Any, threshold: float = 40_000.0) -> bool:
    if hasattr(lookup_table, "attrs") and "spec_mjd" in lookup_table.attrs:
        sm = np.asarray(lookup_table.attrs["spec_mjd"], dtype=float)
        return float(np.nanmax(np.abs(sm))) >= float(threshold)
    idx = np.asarray(lookup_table.index, dtype=float)
    return float(np.nanmax(np.abs(idx))) >= float(threshold)
