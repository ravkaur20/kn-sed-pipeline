"""Trapezoid synthetic photometry on native FINAL grids + 1D GP time smoothing vs observed LC.

Mirrors ``FinalLC.band_flux`` in ``8_Final_Template.ipynb`` (λ-weighted integrals, no synphot).
Uses the same FINAL filename → MJD mapping as ``comparison_check_log_utils.create_lookup_table``.
"""

from __future__ import annotations

import difflib
import os
import warnings
from collections.abc import Collection
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import interpolate, integrate, optimize

import pipeline_config as pconf

from comparison_check_log_utils import (
    FluxOnDisk,
    deduplicate_wavelength_flux,
    parse_final_stem,
    read_final_spectrum_linear,
    stem_to_spec_mjd,
)


def _resolve_phase_reference_mjd(
    snname: str,
    df: pd.DataFrame,
    mjd0: float | None,
) -> float:
    """
    Phase origin in MJD: user ``mjd0``, else ``pconf.SN_EXPLOSION_MJD[snname]``, else min photometry.
    """
    if mjd0 is not None:
        return float(mjd0)
    mmap = getattr(pconf, "SN_EXPLOSION_MJD", None) or {}
    if snname in mmap:
        return float(mmap[snname])
    ref = float(np.nanmin(df["MJD"].values))
    warnings.warn(
        "Phase reference MJD not set and SN_EXPLOSION_MJD has no entry for %r; "
        "using minimum photometry MJD (%.6f)." % (snname, ref),
        UserWarning,
        stacklevel=3,
    )
    return ref

__all__ = [
    "CSP_SN_NAMES",
    "trapz_band_flux",
    "load_filter_curve",
    "resolve_filter_path",
    "collect_trapz_lightcurves",
    "smooth_lightcurve_gp",
    "fit_empirical_mag_zp",
    "warn_bands_photometry",
    "prepare_trapz_gp_comparison",
    "compare_trapz_gp_to_observed_photometry",
    "synphot_abmag_native",
    "collect_synphot_abmag_per_epoch",
]

# Site3 CSP natural system objects (same list as notebook 8 / rimangle).
CSP_SN_NAMES: frozenset[str] = frozenset(
    [
        "SN2004fe",
        "SN2005bf",
        "SN2006V",
        "SN2007C",
        "SN2007Y",
        "SN2009bb",
        "SN2008aq",
        "SN2006T",
        "SN2004gq",
        "SN2004gt",
        "SN2004gv",
        "SN2006ep",
        "SN2008fq",
        "SN2006aa",
    ]
)


def warn_bands_photometry(
    bands: list[str],
    df: pd.DataFrame,
    *,
    csv_path: str | None = None,
    max_uniq_show: int = 40,
    skip_empty_bands: Collection[str] | None = None,
) -> None:
    """
    Emit a ``UserWarning`` for each requested ``band`` that has no rows in ``df['band']``.

    Includes a sample of existing ``band`` strings and ``difflib`` near-miss suggestions.

    ``skip_empty_bands``
        Band names allowed to have no photometry (e.g. synthetic-only filters); no warning.
    """
    if "band" not in df.columns:
        return
    skip = frozenset(skip_empty_bands) if skip_empty_bands else frozenset()
    uniq = sorted(df["band"].astype(str).unique().tolist())
    loc = csv_path or "photometry CSV"
    for band in bands:
        sub = df[df["band"] == band]
        if not sub.empty:
            continue
        if band in skip:
            continue
        sugg = difflib.get_close_matches(str(band), uniq, n=3, cutoff=0.5)
        samp = uniq[:max_uniq_show]
        tail = (
            ""
            if len(uniq) <= max_uniq_show
            else " (+%d more)" % (len(uniq) - max_uniq_show)
        )
        msg = (
            "No photometry rows for band %r in %s. Sample `band` values: %s%s."
            % (band, loc, ", ".join(samp) if samp else "(none)", tail)
        )
        if sugg:
            msg += " Did you mean: %s?" % ", ".join(sugg)
        warnings.warn(msg, UserWarning, stacklevel=2)


def load_filter_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column filter file: wavelength [Å], throughput [0–1]."""
    d = np.loadtxt(path)
    return d[:, 0].astype(float), d[:, 1].astype(float)


def trapz_band_flux(
    spec_wls: np.ndarray,
    spec_flux: np.ndarray,
    spec_flux_err: np.ndarray,
    filt_wls: np.ndarray,
    filt_thr: np.ndarray,
) -> tuple[float, float]:
    """
    Notebook-8 style band mean: ∫(λ T F) dλ / ∫(λ T) dλ with trapezoid rule.
    ``spec_*`` must be observer-frame linear F_λ consistent with filter curve.
    """
    spec_wls = np.asarray(spec_wls, dtype=float).ravel()
    spec_flux = np.asarray(spec_flux, dtype=float).ravel()
    spec_flux_err = np.asarray(spec_flux_err, dtype=float).ravel()
    filt_wls = np.asarray(filt_wls, dtype=float).ravel()
    filt_thr = np.asarray(filt_thr, dtype=float).ravel()

    m = np.isfinite(spec_wls) & np.isfinite(spec_flux) & np.isfinite(spec_flux_err)
    if m.sum() < 2:
        return float("nan"), float("nan")
    spec_wls = spec_wls[m]
    spec_flux = spec_flux[m]
    spec_flux_err = spec_flux_err[m]
    order = np.argsort(spec_wls)
    spec_wls = spec_wls[order]
    spec_flux = spec_flux[order]
    spec_flux_err = spec_flux_err[order]

    lo, hi = float(np.min(filt_wls)), float(np.max(filt_wls))
    cut = (spec_wls > lo) & (spec_wls < hi)
    if cut.sum() < 2:
        return float("nan"), float("nan")
    cut_w = spec_wls[cut]
    cut_f = spec_flux[cut]
    cut_e = spec_flux_err[cut]

    tf = interpolate.interp1d(
        filt_wls, filt_thr, kind="linear", bounds_error=False, fill_value=0.0
    )
    T = tf(cut_w).astype(float)
    TxL = T * cut_w
    den = integrate.trapezoid(TxL, cut_w)
    if not np.isfinite(den) or abs(den) < 1e-300:
        return float("nan"), float("nan")
    num = integrate.trapezoid(TxL * cut_f, cut_w)
    raw = num / den
    raw_err = np.sqrt(integrate.trapezoid((TxL * cut_e) ** 2, cut_w)) / den
    return float(raw), float(raw_err)


def resolve_filter_path(
    band: str,
    *,
    filters_parent: str,
    snname: str = "",
) -> str:
    """
    Find filter curve file for band name matching photometry ``band`` column.

    ``filters_parent`` is ``.../Inputs/Filters`` (not ``GeneralFilters`` alone).
    """
    band = str(band).strip()
    if os.path.isfile(band):
        return band

    base = os.path.normpath(filters_parent)
    cands = [
        os.path.join(base, "GeneralFilters", "%s.dat" % band),
        os.path.join(base, "%s.dat" % band),
        os.path.join(base, "SWIFT", "%s.dat" % band),
        os.path.join(base, "Swift", "%s.dat" % band),
    ]
    if snname in CSP_SN_NAMES:
        cands.append(os.path.join(base, "Site3_CSP", "%s.txt" % band))
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No filter curve file for band %r under %s (tried GeneralFilters/%s.dat, "
        "SWIFT/Swift, Site3_CSP). Spelling must match the throughput filename stem; "
        "compare with the `band` column in your photometry CSV if those differ."
        % (band, base, band)
    )


def collect_trapz_lightcurves(
    data_dir: str,
    coco_path: str,
    snname: str,
    bands: list[str],
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
    filters_parent: str | None = None,
    synphot_abmag_bands: Collection[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    For each FINAL spectrum in ``data_dir``, compute trapz flux in each ``bands`` filter.

    Parameters
    ----------
    synphot_abmag_bands
        If set, also compute per-epoch **synphot** ``effstim('abmag')`` (standard AB band
        mag) for those bands using the same native spectrum grid—no extra file IO.
        Requires ``synphot`` / ``astropy``. Approximate ``abmag_err`` from fractional flux
        error (rough; for GP weights only).

    Returns
    -------
    dict[band, dict]
        ``mjd``, ``flux``, ``flux_err``. Bands in ``synphot_abmag_bands`` also have
        ``abmag``, ``abmag_err`` aligned with ``mjd``.
    """
    if filters_parent is None:
        filters_parent = os.path.join(
            os.path.normpath(coco_path), "Inputs", "Filters"
        )
    spectra_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if final_suffixes:
        spectra_files = [f for f in spectra_files if any(f.endswith(s) for s in final_suffixes)]
    if not spectra_files:
        raise FileNotFoundError("No matching .txt in %s" % data_dir)

    rows: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for fname in spectra_files:
        stem_f = parse_final_stem(fname)
        smjd = stem_to_spec_mjd(
            stem_f, coco_path, snname, datalc_path=datalc_path
        )
        path = os.path.join(data_dir, fname)
        wl, fl, fe = read_final_spectrum_linear(path, flux_on_disk=flux_on_disk)
        m = np.isfinite(wl) & np.isfinite(fl) & np.isfinite(fe)
        wl, fl, fe = wl[m], fl[m], fe[m]
        if wl.size < 2:
            continue
        order = np.argsort(wl)
        wl, fl, fe = wl[order], fl[order], fe[order]
        wl, fl, fe = deduplicate_wavelength_flux(wl, fl, fe)
        rows.append((float(smjd), fname, wl, fl, fe))

    rows.sort(key=lambda r: (r[0], r[1]))
    if not rows:
        raise ValueError("No valid spectra in %s" % data_dir)

    ab_set = frozenset(synphot_abmag_bands) if synphot_abmag_bands else frozenset()
    out: dict[str, dict[str, list]] = {}
    for b in bands:
        od: dict[str, list] = {"mjd": [], "flux": [], "flux_err": []}
        if b in ab_set:
            od["abmag"] = []
            od["abmag_err"] = []
        out[b] = od

    filt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for smjd, _fn, wl, fl, fe in rows:
        for band in bands:
            fpath = resolve_filter_path(band, filters_parent=filters_parent, snname=snname)
            if fpath not in filt_cache:
                filt_cache[fpath] = load_filter_curve(fpath)
            fw, ft = filt_cache[fpath]
            fval, ferr = trapz_band_flux(wl, fl, fe, fw, ft)
            out[band]["mjd"].append(smjd)
            out[band]["flux"].append(fval)
            out[band]["flux_err"].append(ferr)
            if band in ab_set:
                try:
                    mab = float(synphot_abmag_native(wl, fl, fpath))
                except Exception:
                    mab = float("nan")
                out[band]["abmag"].append(mab)
                if np.isfinite(fval) and np.isfinite(ferr) and fval > 0:
                    me = max(
                        0.02,
                        (2.5 / np.log(10.0)) * (float(ferr) / float(fval)),
                    )
                else:
                    me = float("nan")
                out[band]["abmag_err"].append(me)

    result: dict[str, dict[str, np.ndarray]] = {}
    for b in bands:
        dout = {
            "mjd": np.asarray(out[b]["mjd"], dtype=float),
            "flux": np.asarray(out[b]["flux"], dtype=float),
            "flux_err": np.asarray(out[b]["flux_err"], dtype=float),
        }
        if b in ab_set:
            dout["abmag"] = np.asarray(out[b]["abmag"], dtype=float)
            dout["abmag_err"] = np.asarray(out[b]["abmag_err"], dtype=float)
        result[b] = dout
    return result


def smooth_lightcurve_gp(
    time_mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    time_predict_mjd: np.ndarray,
    *,
    optimize: bool = True,
    length_scale_days: float | None = None,
    length_scale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D GP (Matern 3/2) smoothing in flux space. Requires ``george``.

    Length scale (time units = same as ``time_mjd``, typically MJD days):
    - Default: ``max(ptp(time) / 4, 1e-6) * length_scale_factor``.
    - If ``length_scale_days`` is set, it replaces that auto scale (``length_scale_factor``
      is ignored for the initial scale in that case).

    Larger length scale ⇒ smoother curve. If ``optimize`` is True, hyperparameters
    are refined by maximum likelihood (starting from the initial scale above).

    Returns predicted mean and std dev on ``time_predict_mjd``.
    """
    import george
    from george.kernels import Matern32Kernel

    t = np.asarray(time_mjd, dtype=float).ravel()
    y = np.asarray(flux, dtype=float).ravel()
    ye = np.asarray(flux_err, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(ye) & (ye > 0)
    t, y, ye = t[m], y[m], ye[m]
    if t.size == 0:
        tp = np.asarray(time_predict_mjd, dtype=float)
        return np.full_like(tp, np.nan), np.full_like(tp, np.nan)
    if t.size < 3:
        mu = np.interp(np.asarray(time_predict_mjd, dtype=float), t, y)
        return mu, np.full_like(mu, np.nan)

    norm = float(np.nanmedian(np.abs(y[np.isfinite(y) & (np.abs(y) > 0)])))
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0
    yn = y / norm
    yerrn = ye / norm
    var = float(np.nanvar(yn))
    if var <= 0 or not np.isfinite(var):
        var = 1e-12
    span = float(np.ptp(t)) or 1.0
    auto_lscale = max(span / 4.0, 1e-6) * float(length_scale_factor)
    if length_scale_days is not None:
        lscale = max(float(length_scale_days), 1e-6)
    else:
        lscale = auto_lscale
    # George 1D: metric is the squared length scale for Matern32Kernel.
    kernel = var * Matern32Kernel(lscale**2)
    gp = george.GP(kernel)
    gp.compute(t, yerrn)

    if optimize and t.size >= 4:

        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(yn, quiet=True)
            return -ll if np.isfinite(ll) else 1e10

        def grad_nll(p):
            gp.set_parameter_vector(p)
            g = gp.grad_log_likelihood(yn, quiet=True)
            return -g

        p0 = gp.get_parameter_vector().copy()
        try:
            res = optimize.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
            if res.success or res.fun < nll(p0):
                gp.set_parameter_vector(res.x)
        except Exception:
            gp.set_parameter_vector(p0)

    tp = np.asarray(time_predict_mjd, dtype=float).ravel()
    mu_n, var_n = gp.predict(yn, tp, return_var=True)
    std_n = np.sqrt(np.maximum(var_n, 0.0))
    return mu_n * norm, std_n * norm


def fit_empirical_mag_zp(
    mjd_syn: np.ndarray,
    flux_syn: np.ndarray,
    mjd_obs: np.ndarray,
    mag_obs: np.ndarray,
    *,
    dt_match_days: float = 2.0,
) -> tuple[float, int]:
    """
    One additive constant ``zp`` such that ``mag ≈ -2.5 log10(flux_syn) + zp`` on matched epochs.

    Matches each observation to the nearest synthetic epoch if within ``dt_match_days``.

    Returns
    -------
    zp : float
        Median offset, or ``0.0`` if no valid matches (uncalibrated pseudo-mags).
    n_matched : int
        Number of observation epochs used in the median (synthetic epoch within
        ``dt_match_days`` and positive finite flux).
    """
    mjd_syn = np.asarray(mjd_syn, dtype=float)
    flux_syn = np.asarray(flux_syn, dtype=float)
    mjd_obs = np.asarray(mjd_obs, dtype=float)
    mag_obs = np.asarray(mag_obs, dtype=float)
    deltas: list[float] = []
    for i, (to, mo) in enumerate(zip(mjd_obs, mag_obs)):
        if not np.isfinite(to) or not np.isfinite(mo):
            continue
        j = int(np.argmin(np.abs(mjd_syn - to)))
        if abs(mjd_syn[j] - to) <= float(dt_match_days):
            fs = flux_syn[j]
            if not np.isfinite(fs) or fs <= 0:
                continue
            pmag = -2.5 * np.log10(fs)
            deltas.append(float(mo - pmag))
    if not deltas:
        return 0.0, 0
    return float(np.median(deltas)), int(len(deltas))


def prepare_trapz_gp_comparison(
    data_dir: str,
    coco_path: str,
    snname: str,
    photometry_csv: str,
    bands: list[str],
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
    filters_parent: str | None = None,
    mjd0: float | None = None,
    dt_match_days: float = 2.0,
    synthetic_only_bands: Collection[str] | None = None,
) -> dict[str, Any]:
    """
    Load photometry and trapz light curves per band; fit empirical magnitude zeropoint per band.

    Does not plot. Use this from a notebook and plot locally. Emits warnings for missing
    photometry bands (see :func:`warn_bands_photometry`) and for ``n_zp_matches == 0``.

    Parameters
    ----------
    mjd0
        Phase origin for ``days_since = MJD - mjd0`` (typically **explosion / merger MJD**).
        If ``None``, uses ``pipeline_config.SN_EXPLOSION_MJD[snname]`` when defined, otherwise
        minimum photometry MJD (with a warning).
    synthetic_only_bands
        Band names with **no photometry** that should still be included: trapz light curve,
        optional per-epoch **synphot AB mags** (``abmag`` / ``abmag_err``) when listed here,
        ``zp = 0``, ``n_zp_matches = 0``, and an empty ``obs`` frame.
        Plotting uses ``abmag`` when present so magnitudes match standard AB band definitions;
        trapz ``flux`` remains for band-integrated means used in notebook 8.

    Returns
    -------
    dict
        ``mjd0``, ``photometry_csv``, ``snname``, ``bands`` mapping each successful band to
        ``mjd``, ``flux``, ``flux_err``, ``obs`` (DataFrame), ``zp``, ``n_zp_matches``.
        Synthetic-only bands may include ``abmag``, ``abmag_err`` (synphot AB).
        Bands with no photometry rows are omitted unless listed in ``synthetic_only_bands``.
    """
    df = pd.read_csv(photometry_csv)
    for col in ("MJD", "Mag", "Mag_err", "band"):
        if col not in df.columns:
            raise ValueError("Photometry CSV missing column %r" % col)

    mjd0 = _resolve_phase_reference_mjd(snname, df, mjd0)

    syn_only = frozenset(synthetic_only_bands) if synthetic_only_bands else frozenset()
    warn_bands_photometry(bands, df, csv_path=photometry_csv, skip_empty_bands=syn_only)

    lcs = collect_trapz_lightcurves(
        data_dir,
        coco_path,
        snname,
        bands,
        flux_on_disk=flux_on_disk,
        datalc_path=datalc_path,
        final_suffixes=final_suffixes,
        filters_parent=filters_parent,
        synphot_abmag_bands=syn_only,
    )

    bands_out: dict[str, Any] = {}
    for band in bands:
        obs = df[df["band"] == band].copy()
        lc = lcs[band]
        mjd = lc["mjd"]
        flx = lc["flux"]
        fe = lc["flux_err"]
        if obs.empty:
            if band not in syn_only:
                continue
            warnings.warn(
                "Band %r: synthetic-only (no photometry). Using synphot AB mags (abmag) plus "
                "zp=0 on trapz flux only."
                % (band,),
                UserWarning,
                stacklevel=2,
            )
            obs_empty = df.iloc[0:0].copy()
            row: dict[str, Any] = {
                "mjd": mjd,
                "flux": flx,
                "flux_err": fe,
                "obs": obs_empty,
                "zp": 0.0,
                "n_zp_matches": 0,
            }
            if "abmag" in lc:
                row["abmag"] = lc["abmag"]
                row["abmag_err"] = lc["abmag_err"]
            bands_out[band] = row
            continue
        zp, n_zp = fit_empirical_mag_zp(
            mjd,
            flx,
            obs["MJD"].values,
            obs["Mag"].values,
            dt_match_days=dt_match_days,
        )
        if n_zp == 0:
            warnings.warn(
                "Band %r: empirical magnitude zeropoint is uncalibrated (no synthetic epochs "
                "within dt_match_days=%s of any observation). `zp` is 0.0; trapz pseudo-mags "
                "are not on the catalog AB scale."
                % (band, dt_match_days),
                UserWarning,
                stacklevel=2,
            )
        bands_out[band] = {
            "mjd": mjd,
            "flux": flx,
            "flux_err": fe,
            "obs": obs,
            "zp": zp,
            "n_zp_matches": n_zp,
        }

    return {
        "mjd0": mjd0,
        "photometry_csv": photometry_csv,
        "snname": snname,
        "bands": bands_out,
    }


def synphot_abmag_native(
    wavelength_aa: np.ndarray,
    f_lambda: np.ndarray,
    filter_path: str,
) -> float:
    """Single-epoch AB mag via synphot (diagnostic vs trapz). Angstrom, erg/s/cm²/Å."""
    import astropy.units as u
    from synphot import Observation, SourceSpectrum, SpectralElement
    from synphot.models import Empirical1D

    w = np.asarray(wavelength_aa, dtype=float).ravel()
    f = np.asarray(f_lambda, dtype=float).ravel()
    m = np.isfinite(w) & np.isfinite(f)
    w, f = w[m], f[m]
    w, f, _ = deduplicate_wavelength_flux(w, f)
    if w.size < 2:
        return float("nan")
    fw, ft = load_filter_curve(filter_path)
    band = SpectralElement(Empirical1D, points=fw * u.AA, lookup_table=ft)
    src = SourceSpectrum(
        Empirical1D,
        points=w * u.AA,
        lookup_table=f * u.erg / u.s / (u.cm**2) / u.AA,
    )
    obs = Observation(src, band, force="taper")
    return float(obs.effstim("abmag").value)


def compare_trapz_gp_to_observed_photometry(
    data_dir: str,
    coco_path: str,
    snname: str,
    photometry_csv: str,
    bands: list[str],
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
    filters_parent: str | None = None,
    mjd0: float | None = None,
    dt_match_days: float = 2.0,
    gp_n_predict: int = 512,
    gp_optimize: bool = True,
    gp_length_scale_days: float | None = None,
    gp_length_scale_factor: float = 1.0,
    plot_trapz: bool = True,
    plot_gp: bool = True,
    figsize: tuple[float, float] = (9, 6),
    title: str | None = None,
    log_time_axis: bool = True,
    log_time_linthresh_days: float = 1e-2,
    gp_line_color: str | None = None,
    trapz_alpha: float = 0.38,
) -> dict[str, Any]:
    """
    .. deprecated::
        Use :func:`prepare_trapz_gp_comparison` and plot locally for full control.

    Plot observed AB magnitudes vs trapz (+ GP) synthetics on a days-since-mjd0 axis.
    ``plot_trapz`` / ``plot_gp`` select which synthetic series are drawn.
    """
    import matplotlib.pyplot as plt

    warnings.warn(
        "compare_trapz_gp_to_observed_photometry is deprecated; use "
        "prepare_trapz_gp_comparison() and plot in your notebook.",
        DeprecationWarning,
        stacklevel=2,
    )

    prep = prepare_trapz_gp_comparison(
        data_dir,
        coco_path,
        snname,
        photometry_csv,
        bands,
        flux_on_disk=flux_on_disk,
        datalc_path=datalc_path,
        final_suffixes=final_suffixes,
        filters_parent=filters_parent,
        mjd0=mjd0,
        dt_match_days=dt_match_days,
    )
    mjd0 = float(prep["mjd0"])

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    summary: dict[str, Any] = {"mjd0": mjd0, "bands": {}}

    for k, band in enumerate(bands):
        if band not in prep["bands"]:
            continue
        color = colors[k % len(colors)]
        d = prep["bands"][band]
        mjd = d["mjd"]
        flx = d["flux"]
        fe = d["flux_err"]
        obs = d["obs"]
        zp = float(d["zp"])
        n_zp = int(d["n_zp_matches"])

        pseudo = -2.5 * np.log10(np.maximum(flx, 1e-300)) + zp
        obs_time = obs["MJD"].values - mjd0

        ax.errorbar(
            obs_time,
            obs["Mag"].values,
            yerr=obs["Mag_err"].values,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            capthick=1.4,
            alpha=0.95,
            zorder=5,
            label=None,
        )
        ax.scatter(
            obs_time,
            obs["Mag"].values,
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=2.0,
            marker="o",
            zorder=6,
            label="Obs %s" % band,
        )

        syn_time = mjd - mjd0
        if plot_trapz:
            ax.scatter(
                syn_time,
                pseudo,
                s=28,
                marker="s",
                c=color,
                alpha=max(0.05, float(trapz_alpha)),
                edgecolors="0.2",
                linewidths=0.75,
                zorder=4,
                label="Trapz %s" % band,
            )

        m_tr = np.isfinite(mjd) & np.isfinite(flx) & np.isfinite(fe) & (fe > 0)
        if plot_gp and m_tr.sum() >= 3:
            t_syn_lo = float(np.min(mjd[m_tr]))
            t_syn_hi = float(np.max(mjd[m_tr]))
            t_obs_lo = float(np.nanmin(obs["MJD"].values))
            t_obs_hi = float(np.nanmax(obs["MJD"].values))
            t_lo = min(t_syn_lo, t_obs_lo)
            t_hi = max(t_syn_hi, t_obs_hi)
            tgrid = np.linspace(t_lo, t_hi, int(max(8, gp_n_predict)))
            mu, sig = smooth_lightcurve_gp(
                mjd[m_tr],
                flx[m_tr],
                fe[m_tr],
                tgrid,
                optimize=gp_optimize,
                length_scale_days=gp_length_scale_days,
                length_scale_factor=gp_length_scale_factor,
            )
            mag_mu = -2.5 * np.log10(np.maximum(mu, 1e-300)) + zp
            lc = color if gp_line_color is None else gp_line_color
            ax.plot(
                tgrid - mjd0,
                mag_mu,
                "-",
                color=lc,
                lw=2.6,
                alpha=0.92,
                zorder=2,
                label="GP %s" % band,
            )

        summary["bands"][band] = {
            "empirical_zp_ab": zp,
            "n_zp_matches": n_zp,
            "n_syn": int(mjd.size),
        }

    if log_time_axis:
        lt = max(float(log_time_linthresh_days), 1e-6)
        ax.set_xscale("symlog", linthresh=lt, linscale=1.0, base=10)
        ax.set_xlabel(
            "Days since %.4f MJD (symlog: linear for |Δt| < %.3g)"
            % (mjd0, lt)
        )
    else:
        ax.set_xlabel("Days since %.4f MJD" % mjd0)
    ax.set_ylabel("AB mag")
    ax.invert_yaxis()
    ax.set_title(title or "%s: observed vs trapz (+ GP)" % snname)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.35, linestyle="-", linewidth=0.6)
    fig.tight_layout()
    summary["figure"] = fig
    summary["prepare"] = prep
    return summary


def collect_synphot_abmag_per_epoch(
    data_dir: str,
    coco_path: str,
    snname: str,
    band: str,
    *,
    flux_on_disk: FluxOnDisk = "auto",
    datalc_path: str | None = None,
    final_suffixes: tuple[str, ...] | None = None,
    filters_parent: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synphot AB mags at each FINAL epoch (native λ grid). For diagnostics vs trapz.
    """
    if filters_parent is None:
        filters_parent = os.path.join(
            os.path.normpath(coco_path), "Inputs", "Filters"
        )
    fpath = resolve_filter_path(band, filters_parent=filters_parent, snname=snname)
    spectra_Files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if final_suffixes:
        spectra_Files = [
            f for f in spectra_Files if any(f.endswith(s) for s in final_suffixes)
        ]
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    for fname in spectra_Files:
        stem_f = parse_final_stem(fname)
        smjd = stem_to_spec_mjd(stem_f, coco_path, snname, datalc_path=datalc_path)
        path = os.path.join(data_dir, fname)
        wl, fl, _ = read_final_spectrum_linear(path, flux_on_disk=flux_on_disk)
        m = np.isfinite(wl) & np.isfinite(fl)
        wl, fl = wl[m], fl[m]
        if wl.size < 2:
            continue
        order = np.argsort(wl)
        wl, fl = wl[order], fl[order]
        rows.append((float(smjd), wl, fl))
    rows.sort(key=lambda r: r[0])
    mjds = np.array([r[0] for r in rows], dtype=float)
    mags = np.array(
        [synphot_abmag_native(r[1], r[2], fpath) for r in rows], dtype=float
    )
    return mjds, mags
