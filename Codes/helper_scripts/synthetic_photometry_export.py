"""Synthetic AB photometry from the FINAL GP spectra (``FINAL_spectra_2dim/as_observed``).

Produces a per-band light curve on the FINAL-spectrum epochs, plus the same curve
interpolated to every observed photometry row, for an external chi^2 comparison.

The GP is fit to the photometry as observed, so ``as_observed`` synthesizes directly to
magnitudes comparable with the raw catalog (``mag_AB_obs``). The ``dereddened`` variant
(``dust.py``, written at the end of step 6) gives ``mag_AB_dustcorr``; their difference
is the SED-weighted band extinction ``A_ext_band`` per epoch (photon-counting with
``method="true_ab"``).

Two integration conventions (``method``):

``"true_ab"``
    Photon-counting AB magnitude (same as synphot ``effstim('abmag')`` and the
    lambda-weighted band mean used by the mangling step)::

        m = -2.5 log10[ int F_lam T lam dlam / int F_lam^AB T lam dlam ]

``"wtf_band_ab"``
    Energy-weighted convention of ``what_the_flux.Band_AB`` (``zpFlux`` /
    ``Spectrum.bandflux``)::

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
    "band_coverage_fraction",
    "synth_ab_mag",
    "synth_lightcurves_on_grid",
    "interpolate_to_observations",
    "load_sn_info",
    "add_mag_system_columns",
]


# FINAL spectrum filenames store log10(phase) to 6 decimals, so the first/last GP epoch
# can sit up to 5e-7 dex inside the data epoch it was placed on.
_FINAL_STEM_LOG_PHASE_TOL = 5e-7


def _snap_to_grid_ends(x, gx):
    """Move ``x`` within ``_FINAL_STEM_LOG_PHASE_TOL`` of the grid's ends onto them."""
    lo, hi = float(np.min(gx)), float(np.max(gx))
    x = np.asarray(x, dtype=float).copy()
    x[(x < lo) & (lo - x <= _FINAL_STEM_LOG_PHASE_TOL)] = lo
    x[(x > hi) & (x - hi <= _FINAL_STEM_LOG_PHASE_TOL)] = hi
    return x


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
    dereddened_dir: str | None = None,
    min_coverage: float = 0.99,
    excluded_bands: Iterable[str] = (),
) -> pd.DataFrame:
    """Long-format synthetic light curves (one row per band x FINAL spectrum).

    ``final_dir`` (``as_observed``) -> ``mag_AB_obs``, compared directly to raw catalog
    mags. With ``dereddened_dir`` (same file names, ``FINAL_spectra_2dim/dereddened``)
    also ``mag_AB_dustcorr`` and ``A_ext_band = mag_AB_obs - mag_AB_dustcorr``.
    """
    bands = list(bands)
    excluded = set(excluded_bands)
    filters = {b: load_filter(filter_paths[b]) for b in bands}
    rows = []
    for log_phase, fn in list_final_spectra(final_dir):
        w, f, e = read_final_spectrum(os.path.join(final_dir, fn))
        if dereddened_dir is not None:
            dpath = os.path.join(dereddened_dir, fn)
            if not os.path.isfile(dpath):
                raise FileNotFoundError("No dereddened spectrum %s for %s" % (dpath, fn))
            wd, fd, ed = read_final_spectrum(dpath)
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
                "mag_AB_obs": mag,
                "mag_AB_obs_err": err,
            }
            if dereddened_dir is not None:
                mag_d, err_d, _ = synth_ab_mag(wd, fd, ed, fw, ft, method=method,
                                               min_coverage=min_coverage)
                row.update(mag_AB_dustcorr=mag_d, mag_AB_dustcorr_err=err_d,
                           A_ext_band=mag - mag_d)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["band", "MJD"], kind="stable").reset_index(drop=True)


def _mag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("mag_AB_")]


def interpolate_to_observations(
    grid: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    t0_mjd: float,
    min_coverage: float = 0.99,
    mjd_text: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Interpolate each band's synthetic LC to the observed ``MJD`` rows of ``obs``.

    Linear in log10(phase) and in magnitude between the two GP epochs that bracket each
    observation (no rounding or nearest-epoch matching). Rows outside a band's synthetic
    phase range, or bands without valid synthetic mags, get NaN, and ``nan_reason`` says
    why (empty when the first synthetic magnitude column is finite). ``input_row`` is the
    0-based row index of ``obs`` (the raw photometry file) for joining.

    ``mjd_text``: the input file's MJD strings; when given, the output ``MJD`` column is
    written exactly as in the input (the interpolation still uses the numeric values).
    ``min_coverage`` must match the value used for the grid (only used in ``nan_reason``).
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
    reason = np.full(len(out), "band not synthesized (no filter curve)", dtype=object)

    main = mag_cols[0] if mag_cols else None
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
                xc = _snap_to_grid_ends(x, gx[ok])
                inside = np.isfinite(xc) & (xc >= gx[ok].min()) & (xc <= gx[ok].max())
                vals[inside] = np.interp(xc[inside], gx[ok], g[c].to_numpy()[ok])
            out.loc[sel, c] = vals
        cov = float(g["coverage_frac"].min())
        out.loc[sel, "coverage_frac"] = cov
        out.loc[sel, "excluded_from_fit"] = bool(g["excluded_from_fit"].iloc[0])
        mjd_obs = out.loc[sel, "MJD"].to_numpy()
        out.loc[sel, "nearest_gp_epoch_dt_days"] = np.min(
            np.abs(mjd_obs[:, None] - gmjd[None, :]), axis=1
        )
        # why a value is missing
        ok = np.isfinite(g[main].to_numpy())
        r = np.full(x.shape, "", dtype=object)
        if not ok.any():
            r[:] = ("filter only %.0f%% inside the GP wavelength range (needs >= %.0f%%)"
                    % (100 * cov, 100 * min_coverage)) if cov < min_coverage else "no valid synthetic magnitude"
        else:
            ph = 10.0 ** gx[ok]
            p = phase[sel]
            x = _snap_to_grid_ends(x, gx[ok])
            r[~np.isfinite(x)] = "MJD before t0"
            r[np.isfinite(x) & (x < gx[ok].min())] = "MJD before first GP epoch (%.4f d)" % ph.min()
            r[np.isfinite(x) & (x > gx[ok].max())] = [
                "MJD after last GP epoch (phase %.3f d > %.3f d)" % (pi, ph.max())
                for pi in p[np.isfinite(x) & (x > gx[ok].max())]
            ]
            vals = out.loc[sel, main].to_numpy()
            r[(r == "") & ~np.isfinite(vals)] = "no valid synthetic magnitude at bracketing epochs"
        reason[sel] = r
    out["nan_reason"] = reason
    if mjd_text is not None:
        txt = [str(s).strip() for s in mjd_text]
        if len(txt) != len(out):
            raise ValueError("mjd_text has %d entries for %d rows" % (len(txt), len(out)))
        if not np.allclose(pd.to_numeric(pd.Series(txt)).to_numpy(), out["MJD"].to_numpy(), rtol=0, atol=1e-9):
            raise ValueError("mjd_text does not match obs['MJD']")
        out["MJD"] = txt
    return out


def add_mag_system_columns(
    at_data: pd.DataFrame, obs: pd.DataFrame, ab_minus_vega: dict[str, float]
) -> pd.DataFrame:
    """Express each synthetic magnitude in the system of the matching input row.

    ``obs`` is the raw photometry table (its optional ``MagSys`` column is ``AB`` or
    ``Vega``; missing -> AB) and ``ab_minus_vega`` maps band -> ``m_AB - m_Vega``
    (``photometry_mag_to_flux.vega_to_ab_offset``) for every band with Vega rows.
    Adds ``MagSys``, ``ab_minus_insys`` (0 for AB rows) and, for each ``mag_AB_<x>``
    column, ``mag_<x>_insys`` (+ ``_err``) to compare directly with the input ``Mag``.
    """
    out = at_data.copy()
    msys = (obs["MagSys"].fillna("").astype(str).str.strip() if "MagSys" in obs.columns
            else pd.Series("AB", index=obs.index))
    msys = msys.replace({"": "AB"}).str.upper().map({"AB": "AB", "VEGA": "Vega"})
    out["MagSys"] = msys.loc[out["input_row"]].to_numpy()
    vega = (out["MagSys"] == "Vega").to_numpy()
    missing = sorted(set(out.loc[vega, "band"]) - set(ab_minus_vega))
    if missing:
        raise KeyError("No AB-Vega offset for Vega band(s) %s" % missing)
    offset = np.where(vega, out["band"].map(ab_minus_vega).astype(float).fillna(0.0), 0.0)
    out["ab_minus_insys"] = offset
    for c in [c for c in _mag_columns(at_data) if not c.endswith("_err")]:
        tag = c[len("mag_AB_"):]
        out["mag_%s_insys" % tag] = out[c] - offset
        if c + "_err" in out.columns:
            out["mag_%s_insys_err" % tag] = out[c + "_err"]
    return out


def load_sn_info(info_path: str, snname: str) -> dict:
    """``z``, ``EBV_MW``, ``EBV_host``, ``RV_MW``, ``RV_host``, ``Type`` from ``info.dat``."""
    from dust import load_dust_params

    d = load_dust_params(info_path, snname)
    info = pd.read_csv(info_path, comment="#", sep=r"\s+")
    r = info[info["Name"] == snname].iloc[0]
    return {"z": d["redshift"], "EBV_MW": d["ebv_mw"], "EBV_host": d["ebv_host"],
            "RV_MW": d["rv_mw"], "RV_host": d["rv_host"], "Type": str(r["Type"])}
