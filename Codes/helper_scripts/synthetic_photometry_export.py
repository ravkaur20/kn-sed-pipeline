"""Synthetic AB photometry from the FINAL GP spectra (``FINAL_spectra_2dim/as_observed``).

Produces a per-band light curve on the FINAL-spectrum epochs, plus the same curve
interpolated to every observed photometry row, for an external chi^2 comparison.

Two integration conventions (``method``):

``"true_ab"``
    Photon-counting AB magnitude (same as synphot ``effstim('abmag')`` and the
    lambda-weighted band mean used by the mangling step)::

        m = -2.5 log10[ int F_lam T lam dlam / int F_lam^AB T lam dlam ]

``"wtf_band_ab"``
    Energy-weighted convention of ``what_the_flux.Band_AB`` (``zpFlux`` /
    ``Spectrum.bandflux``), i.e. the inverse of how notebook 1 turned catalog mags
    into the ``Flux`` column::

        m = -2.5 log10[ int F_lam T dlam / int F_lam^AB T dlam ]

with F_lam^AB = 3631 Jy * c / lam^2 in both cases. Integrals use the trapezoid rule on
the union of the filter and spectrum wavelength grids (spectrum linearly interpolated).

Magnitude errors propagate the spectrum ``fluxerr`` assuming errors are fully
correlated across wavelength within a band (conservative for a GP surface).
"""

from __future__ import annotations

import os
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import integrate

C_AA_PER_S = 2.99792458e18
AB_FNU_CGS = 3631e-23  # erg / s / cm^2 / Hz
MAG_PER_LN = 2.5 / np.log(10.0)

SynthMethod = Literal["true_ab", "wtf_band_ab"]
SYNTH_METHODS: tuple[str, ...] = ("true_ab", "wtf_band_ab")

FINAL_SPEC_SUFFIXES: tuple[str, ...] = ("_FINAL_spec_FL.txt", "_FINAL_spec.txt")

__all__ = [
    "SYNTH_METHODS",
    "FINAL_SPEC_SUFFIXES",
    "load_filter",
    "list_final_spectra",
    "read_final_spectrum",
    "nb1_band_extinction_mag",
    "band_coverage_fraction",
    "synth_ab_mag",
    "synth_lightcurves_on_grid",
    "interpolate_to_observations",
    "load_sn_info",
]


def _weight(wls: np.ndarray, method: str) -> np.ndarray:
    if method == "true_ab":
        return wls
    if method == "wtf_band_ab":
        return np.ones_like(wls)
    raise ValueError("method must be one of %s, got %r" % (SYNTH_METHODS, method))


def load_filter(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-column filter curve (Angstrom, throughput); sorted, negatives clipped to 0."""
    d = np.loadtxt(path, usecols=(0, 1), dtype=float)
    w, t = d[:, 0], d[:, 1]
    order = np.argsort(w)
    w, t = w[order], np.clip(t[order], 0.0, None)
    w, idx = np.unique(w, return_index=True)
    return w, t[idx]


def list_final_spectra(
    final_dir: str, suffixes: Iterable[str] = FINAL_SPEC_SUFFIXES
) -> list[tuple[float, str]]:
    """``[(log10_phase_days, filename), ...]`` sorted by phase."""
    out = []
    for fn in os.listdir(final_dir):
        for suf in suffixes:
            if fn.endswith(suf):
                out.append((float(fn[: -len(suf)]), fn))
                break
    out.sort()
    return out


def read_final_spectrum(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FINAL spectrum on disk: linear Angstrom, linear F_lam, F_lam error."""
    d = np.atleast_2d(np.loadtxt(path, dtype=float))
    w, f = d[:, 0], d[:, 1]
    e = d[:, 2] if d.shape[1] > 2 else np.zeros_like(f)
    m = np.isfinite(w) & np.isfinite(f)
    w, f, e = w[m], f[m], np.nan_to_num(e[m])
    order = np.argsort(w)
    return w[order], f[order], e[order]


def nb1_band_extinction_mag(
    filt_w: np.ndarray,
    filt_t: np.ndarray,
    *,
    ebv_mw: float,
    rv_mw: float = 3.1,
    ebv_host: float = 0.0,
    rv_host: float = 3.1,
    redshift: float = 0.0,
) -> float:
    """Band extinction A_band [mag] exactly as notebook 1 removed it from the photometry.

    NB1 (``what_the_flux.Band_AB.extinction`` with sncosmo ``CCM89Dust``) uses a
    flat-spectrum band average, dropping the first/last filter samples::

        ext = int T 10^(-0.4 A_lam) dlam / int T dlam

    host term on the rest-frame filter (lam / (1+z)), MW term on the observed filter.
    NB1 multiplied fluxes by 1 / (ext_host * ext_mw); re-reddening adds
    A_band = -2.5 log10(ext_host * ext_mw) to the magnitude.
    """
    import extinction

    def _ext(w, ebv, rv):
        if not ebv:
            return 1.0
        trans = 10.0 ** (-0.4 * extinction.ccm89(w, rv * ebv, rv))
        return integrate.trapezoid((filt_t * trans)[1:-1], w[1:-1]) / integrate.trapezoid(
            filt_t[1:-1], w[1:-1]
        )

    ext = _ext(filt_w / (1.0 + redshift), ebv_host, rv_host) * _ext(filt_w, ebv_mw, rv_mw)
    return float(-2.5 * np.log10(ext))


def band_coverage_fraction(
    spec_wls: np.ndarray, filt_w: np.ndarray, filt_t: np.ndarray, method: str = "true_ab"
) -> float:
    """Fraction of the (method-weighted) filter response inside the spectrum range."""
    wt = filt_t * _weight(filt_w, method)
    tot = integrate.trapezoid(wt, filt_w)
    if not np.isfinite(tot) or tot <= 0:
        return 0.0
    lo, hi = float(np.min(spec_wls)), float(np.max(spec_wls))
    inside = (filt_w >= lo) & (filt_w <= hi)
    if inside.sum() < 2:
        return 0.0
    return float(integrate.trapezoid(wt[inside], filt_w[inside]) / tot)


def synth_ab_mag(
    spec_wls: np.ndarray,
    spec_flux: np.ndarray,
    spec_err: np.ndarray,
    filt_w: np.ndarray,
    filt_t: np.ndarray,
    *,
    method: SynthMethod = "true_ab",
    min_coverage: float = 0.99,
) -> tuple[float, float, float]:
    """``(mag_AB, mag_err, coverage_frac)``; NaN mags when coverage < ``min_coverage``."""
    cov = band_coverage_fraction(spec_wls, filt_w, filt_t, method)
    if cov < min_coverage:
        return float("nan"), float("nan"), cov

    lo = max(float(spec_wls[0]), float(filt_w[0]))
    hi = min(float(spec_wls[-1]), float(filt_w[-1]))
    grid = np.union1d(filt_w[(filt_w >= lo) & (filt_w <= hi)],
                      spec_wls[(spec_wls >= lo) & (spec_wls <= hi)])
    t = np.interp(grid, filt_w, filt_t, left=0.0, right=0.0)
    f = np.interp(grid, spec_wls, spec_flux)
    e = np.interp(grid, spec_wls, spec_err)
    wt = t * _weight(grid, method)

    num = integrate.trapezoid(wt * f, grid)
    num_err = integrate.trapezoid(wt * e, grid)  # fully correlated within the band
    # AB reference over the full filter curve (spectrum covers >= min_coverage of it)
    f_ab = AB_FNU_CGS * C_AA_PER_S / filt_w**2
    den = integrate.trapezoid(filt_t * _weight(filt_w, method) * f_ab, filt_w)
    if not (num > 0 and den > 0):
        return float("nan"), float("nan"), cov
    mag = -2.5 * np.log10(num / den)
    return float(mag), float(MAG_PER_LN * num_err / num), cov


def synth_lightcurves_on_grid(
    final_dir: str,
    bands: Iterable[str],
    filter_paths: dict[str, str],
    *,
    t0_mjd: float,
    method: SynthMethod = "true_ab",
    apply_extinction: bool = True,
    ext_kwargs: dict | None = None,
    min_coverage: float = 0.99,
    excluded_bands: Iterable[str] = (),
) -> pd.DataFrame:
    """Long-format synthetic light curves (one row per band x FINAL spectrum).

    ``apply_extinction=True`` -> columns ``mag_AB_obs`` (NB1 dust correction undone, see
    :func:`nb1_band_extinction_mag`; compare to raw catalog mags), ``mag_AB_dustcorr``
    (spectra as-is) and ``A_ext_band``. ``False`` -> ``mag_AB_model`` (spectra as-is only).
    """
    bands = list(bands)
    excluded = set(excluded_bands)
    filters = {b: load_filter(filter_paths[b]) for b in bands}
    a_band = (
        {b: nb1_band_extinction_mag(*filters[b], **(ext_kwargs or {})) for b in bands}
        if apply_extinction else {}
    )
    rows = []
    for log_phase, fn in list_final_spectra(final_dir):
        w, f, e = read_final_spectrum(os.path.join(final_dir, fn))
        phase = 10.0**log_phase
        for b in bands:
            fw, ft = filters[b]
            mag, err, cov = synth_ab_mag(w, f, e, fw, ft, method=method, min_coverage=min_coverage)
            row = {
                "band": b,
                "MJD": t0_mjd + phase,
                "phase_days": phase,
                "log10_phase": log_phase,
                "spec_file": fn,
                "coverage_frac": round(cov, 5),
                "excluded_from_fit": b in excluded,
            }
            if apply_extinction:
                row.update(mag_AB_obs=mag + a_band[b], mag_AB_obs_err=err,
                           mag_AB_dustcorr=mag, mag_AB_dustcorr_err=err,
                           A_ext_band=a_band[b])
            else:
                row.update(mag_AB_model=mag, mag_AB_model_err=err)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["band", "MJD"], kind="stable").reset_index(drop=True)


def _mag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("mag_AB_")]


def interpolate_to_observations(
    grid: pd.DataFrame, obs: pd.DataFrame, *, t0_mjd: float
) -> pd.DataFrame:
    """Interpolate each band's synthetic LC to the observed ``MJD`` rows of ``obs``.

    Linear in log10(phase) and in magnitude. Rows outside a band's synthetic phase range
    (or bands without valid synthetic mags) get NaN. ``input_row`` is the 0-based row
    index of ``obs`` (the raw photometry file) for joining.
    """
    mag_cols = _mag_columns(grid)
    out = pd.DataFrame({
        "input_row": obs.index.to_numpy(),
        "MJD": obs["MJD"].to_numpy(dtype=float),
        "band": obs["band"].astype(str).to_numpy(),
    })
    phase = out["MJD"].to_numpy() - t0_mjd
    out["phase_days"] = phase
    with np.errstate(invalid="ignore", divide="ignore"):
        out["log10_phase"] = np.where(phase > 0, np.log10(phase), np.nan)
    for c in mag_cols + ["coverage_frac", "excluded_from_fit", "nearest_gp_epoch_dt_days"]:
        out[c] = np.nan
    out["excluded_from_fit"] = out["excluded_from_fit"].astype(object)

    for b, g in grid.groupby("band", sort=False):
        sel = (out["band"] == b).to_numpy()
        if not sel.any():
            continue
        x = out.loc[sel, "log10_phase"].to_numpy()
        gx = g["log10_phase"].to_numpy()
        gmjd = g["MJD"].to_numpy()
        for c in mag_cols:
            ok = np.isfinite(g[c].to_numpy())
            vals = np.full(x.shape, np.nan)
            if ok.sum() >= 2:
                inside = np.isfinite(x) & (x >= gx[ok].min()) & (x <= gx[ok].max())
                vals[inside] = np.interp(x[inside], gx[ok], g[c].to_numpy()[ok])
            out.loc[sel, c] = vals
        out.loc[sel, "coverage_frac"] = float(g["coverage_frac"].min())
        out.loc[sel, "excluded_from_fit"] = bool(g["excluded_from_fit"].iloc[0])
        mjd_obs = out.loc[sel, "MJD"].to_numpy()
        out.loc[sel, "nearest_gp_epoch_dt_days"] = np.min(
            np.abs(mjd_obs[:, None] - gmjd[None, :]), axis=1
        )
    return out


def load_sn_info(info_path: str, snname: str) -> dict:
    """``z``, ``EBV_MW``, ``EBV_host``, ``Type`` for ``snname`` from ``Inputs/SNe_Info/info.dat``."""
    info = pd.read_csv(info_path, comment="#", sep=r"\s+")
    row = info[info["Name"] == snname]
    if row.empty:
        raise KeyError("%s not in %s" % (snname, info_path))
    r = row.iloc[0]
    return {"z": float(r["z"]), "EBV_MW": float(r["EBV_MW"]),
            "EBV_host": float(r["EBV_host"]), "Type": str(r["Type"])}
