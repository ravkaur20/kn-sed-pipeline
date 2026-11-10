"""Flux uncertainties for clipped spectra (notebook 0.3).

Reads the output of notebook 0.2 (``1.5_spec_clipped``) and gives every pixel
a flux uncertainty. The spectra themselves are NOT smoothed or resampled: the
output has the same pixels as the input, so the GP later sees the real data.

Spectra with an error column can trust it, rescale it to the estimate, or
replace it with the estimate (``err_mode``). Spectra without errors always
use the estimate. The estimate follows kaepora's ``genivar``: smooth, take
|residual|, smooth that, then convert mean |residual| to sigma with a factor
computed for this spectrum (its pixel size and the smoothing width). The
smoothing is only used to *measure* the noise; it never touches the output
flux.

Nothing here depends on the instrument: only on whether a file has errors and
its pixel size.

A spectrum is a plain dict: ``{"name", "mjd", "wl", "flux", "err", "weight"}``
with ``err = None`` when the file has no usable errors. Pixels clipped in 0.2
keep ``weight = 0`` (their flux is a bridge, not data).
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import pipeline_config as pconf

SPEC_ROOT = pconf.spectroscopy_root()
CLIP_DIR = os.path.join(SPEC_ROOT, "1.5_spec_clipped")
CLIP_LIST_DIR = os.path.join(SPEC_ROOT, "1.5_spec_lists_clipped")
ERR_DIR = os.path.join(SPEC_ROOT, pconf.REDUCED_SPEC_SUBDIR)
ERR_LIST_DIR = os.path.join(SPEC_ROOT, pconf.REDUCED_SPEC_LIST_SUBDIR)

DEFAULTS = dict(
    err_mode="trust",          # spectra WITH errors: "trust" | "rescale" | "estimate"
    rescale_shape="constant",  # for "rescale": "constant" | "linear" (in wavelength)
    est_vexp_smooth=0.002,     # kaepora genivar: smoothing whose residuals are the noise
    est_vexp_resid=0.015,      # kaepora genivar: smoothing of |residual|
    nsig=5.0,                  # smoothing kernels truncated at +-nsig sigma (kaepora)
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def list_clipped(snname):
    """[(mjd, filename), ...] sorted by MJD, from notebook 0.2's list file."""
    rows = []
    with open(os.path.join(CLIP_LIST_DIR, snname + ".list")) as f:
        for line in f:
            parts = line.split()
            if parts and not parts[0].startswith("#"):
                rows.append((float(parts[0]), os.path.basename(parts[2])))
    return sorted(rows)


def load_clipped(snname, filename, mjd=None):
    """Load a 0.2 output (wls flux flux_err weight).

    ``err`` is None if the file has no finite, positive errors. Pixels whose
    own error is unusable are given weight 0.
    """
    wl, flux, err, weight = np.loadtxt(
        os.path.join(CLIP_DIR, snname, filename), unpack=True)
    usable = np.isfinite(err) & (err > 0)
    if usable.any():
        weight = np.where(usable, weight, 0.0)
        err = np.where(usable, err, np.nan)
    else:
        err = None
    return {"name": filename, "mjd": mjd, "wl": wl, "flux": flux,
            "err": err, "weight": weight}


def has_errors(spec):
    return spec["err"] is not None


def is_usable(spec, min_good=100):
    """False when (almost) every pixel was clipped in 0.2."""
    return int((spec["weight"] > 0).sum()) >= min_good


# ---------------------------------------------------------------------------
# Gaussian smoothing (kaepora gsmooth, vectorised)
# ---------------------------------------------------------------------------

def gsmooth(wl, y, ivar, vexp, x_out=None, nsig=5.0, good=None, chunk=256):
    """Inverse-variance weighted Gaussian smoothing, sigma = vexp * x.

    Same weighting as kaepora's ``gsmooth``: each input pixel j contributes
    ``K_j * ivar_j`` to the output at x, with K a Gaussian of sigma
    ``vexp * x`` truncated at ``nsig`` sigma. Pixels with ``good == False``
    contribute nothing.

    Returns a dict of arrays at ``x_out`` (default: ``wl``):
        y        smoothed values
        var      propagated variance, sum(w^2/ivar) / sum(w)^2
        bad_frac fraction of the (unweighted) kernel on non-good pixels
        k0, k2   own-pixel kernel weight and sum of squared kernel weights
                 (normalised over good pixels; used for the noise factor)
    """
    wl = np.asarray(wl, float)
    y = np.asarray(y, float)
    ivar = np.broadcast_to(np.asarray(ivar, float), wl.shape)
    good = np.ones(wl.shape, bool) if good is None else np.asarray(good, bool)
    good = good & np.isfinite(y) & np.isfinite(ivar) & (ivar > 0)
    x_out = wl if x_out is None else np.asarray(x_out, float)

    keys = ("y", "var", "bad_frac", "k0", "k2")
    out = {k: np.full(len(x_out), np.nan) for k in keys}
    y0 = np.where(good, y, 0.0)
    iv0 = np.where(good, ivar, 0.0)

    for s in range(0, len(x_out), chunk):
        x = x_out[s:s + chunk]
        sig = vexp * x
        j0 = np.searchsorted(wl, x[0] - nsig * sig[0])
        j1 = np.searchsorted(wl, x[-1] + nsig * sig[-1], side="right")
        if j1 <= j0:
            continue
        d = (wl[None, j0:j1] - x[:, None]) / sig[:, None]
        k = np.where(np.abs(d) <= nsig, np.exp(-0.5 * d * d), 0.0)
        g = good[None, j0:j1]
        ktot = k.sum(1)
        kg = k * g
        kgsum = kg.sum(1)
        w = k * iv0[None, j0:j1]
        wsum = w.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["y"][s:s + chunk] = (w * y0[None, j0:j1]).sum(1) / wsum
            out["var"][s:s + chunk] = (w * w / np.where(iv0 > 0, iv0, np.inf)[None, j0:j1]).sum(1) / wsum ** 2
            out["bad_frac"][s:s + chunk] = 1.0 - kgsum / ktot
            kn = kg / kgsum[:, None]
            out["k2"][s:s + chunk] = (kn * kn).sum(1)
        # own-pixel weight: only meaningful when x_out is the input grid
        if x_out is wl:
            idx = np.arange(s, s + len(x)) - j0
            out["k0"][s:s + chunk] = kn[np.arange(len(x)), idx]
    return out


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def noise_factor(k0, k2):
    """Factor converting the smoothed |residual| into sigma.

    For white noise n with sigma s, the residual n_i - sum_j K_j n_j has
    variance s^2 (1 - 2 K_0 + sum K^2), and E|N(0, v)| = sqrt(2 v / pi).
    So sigma = sqrt(pi / 2) / sqrt(1 - 2 K_0 + sum K^2) * mean|residual|.
    """
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.pi / 2) / np.sqrt(1.0 - 2.0 * k0 + k2)


def estimate_errors(spec, vexp_smooth=0.002, vexp_resid=0.015, nsig=5.0):
    """Kaepora-style error spectrum from the data alone.

    Returns (sigma_est, info) where info holds the intermediate arrays used
    in the diagnostic plot.
    """
    wl, flux = spec["wl"], spec["flux"]
    good = spec["weight"] > 0
    flat = np.ones_like(wl)
    sm = gsmooth(wl, flux, flat, vexp_smooth, nsig=nsig, good=good)
    resid = np.abs(flux - sm["y"])
    sm_resid = gsmooth(wl, resid, flat, vexp_resid, nsig=nsig, good=good)["y"]
    factor = noise_factor(sm["k0"], sm["k2"])
    factor = np.where(np.isfinite(factor), factor, np.sqrt(np.pi / 2))
    sigma = factor * sm_resid
    return sigma, {"smooth": sm["y"], "abs_resid": resid,
                   "sm_abs_resid": sm_resid, "factor": factor}


def rescale_errors(spec, sigma_est, shape="constant", vexp_resid=0.015, nsig=5.0):
    """Scale the real errors to match the estimate (per file).

    The real errors are smoothed on the same scale as the estimate before the
    ratio estimate / real is taken over good pixels. ``shape="constant"``
    uses the median ratio; ``"linear"`` fits the ratio as a straight line in
    wavelength (kaepora's polyfit).

    Returns (scale_array, scale_info).
    """
    wl = spec["wl"]
    good = spec["weight"] > 0
    real_sm = gsmooth(wl, spec["err"], np.ones_like(wl), vexp_resid,
                      nsig=nsig, good=good)["y"]
    ratio = sigma_est / real_sm
    ok = good & np.isfinite(ratio) & (ratio > 0)
    if shape == "constant":
        c = float(np.median(ratio[ok]))
        return np.full_like(wl, c), {"shape": "constant", "scale": c}
    if shape == "linear":
        # robust-ish: fit only ratios within 5x of the median
        med = np.median(ratio[ok])
        fit = ok & (ratio > med / 5) & (ratio < med * 5)
        p = np.polyfit(wl[fit], ratio[fit], 1)
        return np.polyval(p, wl), {"shape": "linear", "coeffs": [float(v) for v in p]}
    raise ValueError("rescale_shape must be 'constant' or 'linear'")


def choose_errors(spec, params):
    """Apply ``err_mode`` and return (sigma, info) for this spectrum."""
    p = params
    sigma_est, est = estimate_errors(spec, p["est_vexp_smooth"],
                                     p["est_vexp_resid"], p["nsig"])
    info = {"sigma_est": sigma_est, "est": est, "scale": None,
            "has_errors": has_errors(spec)}
    if not has_errors(spec):
        info["mode"] = "estimate (no errors in file)"
        return sigma_est, info

    mode = p["err_mode"]
    real_sm = gsmooth(spec["wl"], spec["err"], np.ones_like(spec["wl"]),
                      p["est_vexp_resid"], nsig=p["nsig"],
                      good=spec["weight"] > 0)["y"]
    info["real_sm"] = real_sm
    info["ratio_median"] = float(np.nanmedian((sigma_est / real_sm)[spec["weight"] > 0]))
    if mode == "trust":
        info["mode"] = "trust"
        return spec["err"], info
    if mode == "rescale":
        scale, sinfo = rescale_errors(spec, sigma_est, p["rescale_shape"],
                                      p["est_vexp_resid"], p["nsig"])
        info["mode"] = "rescale (%s)" % p["rescale_shape"]
        info["scale"] = sinfo
        return spec["err"] * scale, info
    if mode == "estimate":
        info["mode"] = "estimate (ignoring file errors)"
        return sigma_est, info
    raise ValueError("err_mode must be 'trust', 'rescale' or 'estimate'")


# ---------------------------------------------------------------------------
# Whole spectrum
# ---------------------------------------------------------------------------

def low_snr_flags(snname, filename):
    """Low-S/N regions flagged in 0.2 (for plotting only); [] if none."""
    path = os.path.join(CLIP_DIR, snname, snname + "_clip_regions.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        entry = json.load(f).get(filename, {})
    return [tuple(r) for r in entry.get("low_snr_flags", [])]


def reduce_one(spec, params):
    """Give every pixel of one spectrum an uncertainty.

    Returns a result dict for the plots and ``save_errors``. ``chi`` is
    (flux - smooth) / sigma, using the estimate's own smoothing: its robust
    std should be about 1 when sigma matches the real pixel-to-pixel scatter.
    """
    if not is_usable(spec):
        raise ValueError("%s: almost every pixel is clipped (weight 0)" % spec["name"])
    sigma, err_info = choose_errors(spec, params)
    good = spec["weight"] > 0
    chi = (spec["flux"] - err_info["est"]["smooth"]) / sigma
    chi_good = chi[good & np.isfinite(chi)]
    chi_std = float(1.4826 * np.median(np.abs(chi_good - np.median(chi_good))))
    snr = spec["flux"][good] / sigma[good]
    return {"spec": spec, "params": dict(params), "sigma": sigma,
            "err_info": err_info, "chi": chi, "chi_std": chi_std,
            "snr_per_pixel": float(np.median(snr))}


def summary_row(res):
    """One-line summary (for a table of all spectra)."""
    e = res["err_info"]
    return {
        "name": res["spec"]["name"], "mjd": res["spec"]["mjd"],
        "err_mode": e["mode"],
        "est/real": round(e.get("ratio_median", np.nan), 2),
        "chi_std": round(res["chi_std"], 2),
        "S/N per pixel": round(res["snr_per_pixel"], 1),
        "factor": round(float(np.nanmedian(e["est"]["factor"])), 3),
        "frac_w0": round(float(1 - (res["spec"]["weight"] > 0).mean()), 2),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def settings_path(snname):
    return os.path.join(ERR_DIR, snname, snname + "_error_settings.json")


def load_settings(snname):
    path = settings_path(snname)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_errors(snname, res, customised=False, note=""):
    """Write ``<REDUCED_SPEC_SUBDIR>/<SN>/<file>`` (wls flux flux_err weight,
    same pixels as 0.2), record the settings, and rewrite the list file."""
    spec = res["spec"]
    out_dir = os.path.join(ERR_DIR, snname)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ERR_LIST_DIR, exist_ok=True)
    np.savetxt(os.path.join(out_dir, spec["name"]),
               np.column_stack([spec["wl"], spec["flux"], res["sigma"],
                                spec["weight"]]),
               fmt=["%.4f", "%.6e", "%.6e", "%d"],
               header="wls\tflux\tflux_err\tweight", delimiter="\t")

    e = res["err_info"]
    allset = load_settings(snname)
    allset[spec["name"]] = {
        "mjd": spec["mjd"], "customised": bool(customised), "note": note,
        "has_errors": e["has_errors"], "err_mode": e["mode"],
        "rescale": e["scale"], "est_over_real_median": e.get("ratio_median"),
        "err_factor_median": float(np.nanmedian(e["est"]["factor"])),
        "chi_std": res["chi_std"], "snr_per_pixel": res["snr_per_pixel"],
        "params": dict(res["params"]),
    }
    with open(settings_path(snname), "w") as f:
        json.dump(allset, f, indent=1, sort_keys=True)

    rows = sorted((s["mjd"], n) for n, s in allset.items())
    mjd0 = rows[0][0]
    with open(os.path.join(ERR_LIST_DIR, snname + ".list"), "w") as f:
        for m, n in rows:
            path = os.path.relpath(os.path.join(out_dir, n), pconf.COCO_PATH)
            f.write("%.6f\t%.6f\t%s\n" % (m, m - mjd0, path))
    print("saved %s | %s | chi_std %.2f" % (spec["name"], e["mode"], res["chi_std"]))


# ---------------------------------------------------------------------------
# Plotting (each takes an ``ax``)
# ---------------------------------------------------------------------------

def _ylim(ax, y, lo=1, hi=99, pad=0.15):
    y = y[np.isfinite(y)]
    if len(y):
        a, b = np.percentile(y, [lo, hi])
        ax.set_ylim(a - pad * (b - a), b + pad * (b - a))


def _shade_flags(ax, flags):
    for i, (lo, hi) in enumerate(flags):
        ax.axvspan(lo, hi, facecolor="none", edgecolor="tab:orange", hatch="///",
                   lw=0.8, alpha=0.8, label="low S/N flag" if i == 0 else None)


def plot_spectrum(res, ax=None, flags=()):
    """The pixels that go downstream, with their +-1 sigma band. Clipped
    (weight 0) stretches are shown in light grey."""
    ax = ax or plt.subplots(figsize=(13, 4))[1]
    spec = res["spec"]
    wl, good = spec["wl"], spec["weight"] > 0
    f = np.where(good, spec["flux"], np.nan)
    s = np.where(good, res["sigma"], np.nan)
    ax.fill_between(wl, f - s, f + s, color="tab:blue", alpha=0.25, lw=0,
                    label="$\\pm1\\sigma$")
    ax.plot(wl, f, color="k", lw=0.4, label="flux (weight 1)")
    ax.plot(wl, np.where(good, np.nan, spec["flux"]), color="0.75", lw=0.6,
            label="clipped in 0.2 (weight 0)")
    _shade_flags(ax, flags)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    _ylim(ax, spec["flux"][good])
    ax.set_ylabel("Flux")
    ax.set_title(spec["name"], fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_errors(res, ax=None, flags=()):
    """Error spectra: estimate, real (if any), and what is used."""
    ax = ax or plt.subplots(figsize=(13, 3))[1]
    spec, e = res["spec"], res["err_info"]
    wl, good = spec["wl"], spec["weight"] > 0
    ax.plot(wl, np.where(good, e["est"]["abs_resid"], np.nan), color="0.85",
            lw=0.4, label="|residual|")
    ax.plot(wl, np.where(good, e["sigma_est"], np.nan), color="tab:orange",
            lw=1.2, label="estimate")
    if e["has_errors"]:
        ax.plot(wl, np.where(good, e["real_sm"], np.nan), color="tab:blue",
                lw=1.2, label="real (smoothed)  est/real = %.2f" % e["ratio_median"])
    ax.plot(wl, np.where(good, res["sigma"], np.nan), color="k", lw=0.5,
            alpha=0.6, label="used: " + e["mode"])
    _shade_flags(ax, flags)
    _ylim(ax, np.concatenate([e["sigma_est"][good], res["sigma"][good]]), 1, 99)
    ax.set_ylabel("$\\sigma$")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_chi(res, ax=None):
    """(flux - smooth) / sigma: should scatter around +-1 if the errors are right."""
    ax = ax or plt.subplots(figsize=(13, 2.5))[1]
    spec = res["spec"]
    good = spec["weight"] > 0
    ax.plot(spec["wl"], np.where(good, res["chi"], np.nan), color="k", lw=0.3)
    for y in (-1, 1):
        ax.axhline(y, color="tab:red", lw=0.8, ls="--")
    ax.set_ylim(-5, 5)
    ax.set_ylabel("resid / $\\sigma$")
    ax.set_xlabel("Observed wavelength [$\\AA$]")
    ax.text(0.01, 0.9, "robust std = %.2f (1 = errors match scatter)" % res["chi_std"],
            transform=ax.transAxes, fontsize=8)
    return ax


def plot_summary(res, flags=(), figsize=(15, 9)):
    """Spectrum, errors, and residual check for one spectrum.
    Returns (fig, axes dict)."""
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True,
                            gridspec_kw={"height_ratios": [1.3, 1, 0.7]})
    axs = {"spectrum": axs[0], "errors": axs[1], "chi": axs[2]}
    plot_spectrum(res, axs["spectrum"], flags)
    plot_errors(res, axs["errors"], flags)
    plot_chi(res, axs["chi"])
    fig.tight_layout()
    return fig, axs
