"""AB magnitude -> F_lambda band flux for the raw photometry file (pipeline step 0).

The pipeline works with **band-mean fluxes**: the mangling step fits the photon-weighted
mean ``<f_lam> = int f_lam S lam dlam / int S lam dlam`` (``mangle_spectra_log.band_flux_trapz``)
and synthetic photometry uses the photon-counting AB definition. For an AB magnitude the
photon-weighted band mean is exact and independent of the source SED::

    <f_lam> = 3631 Jy * 10**(-0.4 m_AB) * c / lam_piv**2
    lam_piv = sqrt( int S lam dlam / int S / lam dlam )

so ``Flux`` must be computed at the filter **pivot** wavelength (not an effective or median
wavelength, which give fluxes inconsistent with the rest of the pipeline).

``S`` is the photon-counting throughput in ``Inputs/Filters/<FilterSet>/<band>.dat``; the
overall normalization of the curve (fraction vs percent) does not matter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import integrate

C_AA_PER_S = 2.99792458e18
AB_FNU_CGS = 3631e-23  # AB zero point: f_nu = 3631 Jy (m_AB = -2.5 log10 f_nu - 48.5999)
FLUX_FLOAT_FORMAT = "%.12g"
MAG_SYSTEM_COLUMNS = ("MagSys", "mag_sys", "system")

__all__ = [
    "pivot_wavelength",
    "abmag_to_flambda",
    "abmag_err_to_flambda_err",
    "flambda_to_abmag",
    "resolve_band_filter_path",
    "BandConversion",
    "convert_raw_photometry",
    "check_raw_photometry",
    "read_raw_photometry",
    "write_raw_photometry",
    "bands_report_table",
]


def pivot_wavelength(filt_w: np.ndarray, filt_t: np.ndarray) -> float:
    """Pivot wavelength [Angstrom] of a photon-counting throughput curve."""
    w = np.asarray(filt_w, dtype=float)
    t = np.clip(np.asarray(filt_t, dtype=float), 0.0, None)
    order = np.argsort(w)
    w, t = w[order], t[order]
    den = integrate.trapezoid(t / w, w)
    if not den > 0:
        raise ValueError("filter throughput integrates to zero")
    return float(np.sqrt(integrate.trapezoid(t * w, w) / den))


def abmag_to_flambda(mag, lam_piv: float):
    """AB mag -> photon-weighted band-mean F_lambda [erg s^-1 cm^-2 A^-1]."""
    return AB_FNU_CGS * 10.0 ** (-0.4 * np.asarray(mag, dtype=float)) * C_AA_PER_S / lam_piv**2


def abmag_err_to_flambda_err(flux, mag_err):
    """Linear propagation sigma_F = F * 0.4 ln(10) * sigma_m (adequate for S/N >~ 10)."""
    return np.abs(np.asarray(flux, dtype=float)) * 0.4 * np.log(10.0) * np.asarray(mag_err, dtype=float)


def flambda_to_abmag(flux, lam_piv: float):
    """Inverse of :func:`abmag_to_flambda`."""
    return -2.5 * np.log10(np.asarray(flux, dtype=float) * lam_piv**2 / (C_AA_PER_S * AB_FNU_CGS))


def resolve_band_filter_path(band: str, filter_set: str, filters_root: str) -> str:
    """``Inputs/Filters/<FilterSet>/<band>.dat`` (``.txt`` accepted)."""
    for ext in (".dat", ".txt"):
        p = os.path.join(filters_root, str(filter_set), "%s%s" % (band, ext))
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No filter curve for band %r in %s (expected %s.dat). The `band` value must match "
        "the filter file name." % (band, os.path.join(filters_root, str(filter_set)), band)
    )


@dataclass
class BandConversion:
    band: str
    filter_path: str
    lam_piv: float
    n_rows: int
    max_rel_change: float  # max |Flux_new / Flux_old - 1| over the band's rows
    median_dmag_old_minus_new: float  # -2.5 log10(Flux_old / Flux_new)


def read_raw_photometry(path: str) -> pd.DataFrame:
    """Read the raw photometry CSV with every column as text (lossless round-trip)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"Mag", "Mag_err", "Flux", "Flux_err", "band"} - set(df.columns)
    if missing:
        raise ValueError("%s is missing required columns %s" % (path, sorted(missing)))
    return df


def write_raw_photometry(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def _check_ab_only(df: pd.DataFrame) -> None:
    for col in MAG_SYSTEM_COLUMNS:
        if col in df.columns:
            sys_vals = {s.strip().upper() for s in df[col] if s.strip()}
            if sys_vals - {"AB"}:
                raise ValueError(
                    "Column %r contains non-AB magnitude systems %s; only AB is supported."
                    % (col, sorted(sys_vals - {"AB"}))
                )


def _to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce").to_numpy(dtype=float)


def convert_raw_photometry(
    df: pd.DataFrame,
    filters_root: str,
    *,
    default_filter_set: str = "GeneralFilters",
) -> tuple[pd.DataFrame, list[BandConversion]]:
    """Recompute ``Flux``/``Flux_err`` from ``Mag``/``Mag_err`` at each band's pivot wavelength.

    All other columns are left byte-for-byte unchanged. Rows with a missing ``Mag`` keep
    their existing ``Flux``/``Flux_err``. Returns ``(new_df, per_band_report)``.
    """
    _check_ab_only(df)
    out = df.copy()
    mag = _to_float(df["Mag"])
    mag_err = _to_float(df["Mag_err"])
    flux_old = _to_float(df["Flux"])
    fsets = df["FilterSet"] if "FilterSet" in df.columns else pd.Series(
        [default_filter_set] * len(df), index=df.index
    )

    report: list[BandConversion] = []
    new_flux = flux_old.copy()
    new_err = _to_float(df["Flux_err"])
    for (band, fset), idx in df.groupby([df["band"], fsets.replace("", default_filter_set)]).groups.items():
        pos = df.index.get_indexer(idx)
        path = resolve_band_filter_path(band, fset, filters_root)
        d = np.loadtxt(path, usecols=(0, 1), dtype=float)
        lam_piv = pivot_wavelength(d[:, 0], d[:, 1])
        ok = np.isfinite(mag[pos])
        p_ok = pos[ok]
        new_flux[p_ok] = abmag_to_flambda(mag[p_ok], lam_piv)
        new_err[p_ok] = abmag_err_to_flambda_err(new_flux[p_ok], mag_err[p_ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux_old[p_ok] / new_flux[p_ok]
        fin = np.isfinite(ratio) & (ratio > 0)
        report.append(
            BandConversion(
                band=str(band),
                filter_path=path,
                lam_piv=lam_piv,
                n_rows=int(len(pos)),
                max_rel_change=float(np.max(np.abs(ratio[fin] - 1.0))) if fin.any() else float("nan"),
                median_dmag_old_minus_new=float(np.median(-2.5 * np.log10(ratio[fin])))
                if fin.any() else float("nan"),
            )
        )

    def _fmt(vals: np.ndarray, old: pd.Series) -> list[str]:
        return [o if not np.isfinite(v) else FLUX_FLOAT_FORMAT % v for v, o in zip(vals, old)]

    has_mag = np.isfinite(mag)
    out["Flux"] = np.where(has_mag, _fmt(new_flux, df["Flux"]), df["Flux"])
    out["Flux_err"] = np.where(has_mag, _fmt(new_err, df["Flux_err"]), df["Flux_err"])
    report.sort(key=lambda r: r.lam_piv)
    return out, report


def check_raw_photometry(
    df: pd.DataFrame,
    filters_root: str,
    *,
    rtol: float = 1e-6,
    default_filter_set: str = "GeneralFilters",
) -> list[BandConversion]:
    """Bands whose existing ``Flux`` deviates from the pivot-wavelength AB conversion by > ``rtol``."""
    _, report = convert_raw_photometry(df, filters_root, default_filter_set=default_filter_set)
    return [r for r in report if not (r.max_rel_change <= rtol)]


def bands_report_table(report: Iterable[BandConversion]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "band": r.band,
                "lam_piv_A": round(r.lam_piv, 1),
                "n_rows": r.n_rows,
                "dmag_old_minus_new": round(r.median_dmag_old_minus_new, 4),
                "max_rel_change": r.max_rel_change,
            }
            for r in report
        ]
    )
