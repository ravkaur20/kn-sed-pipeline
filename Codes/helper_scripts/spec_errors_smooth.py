"""Error estimation and smoothing for clipped spectra (notebook 0.3).

Reads the output of notebook 0.2 (``1.5_spec_clipped``) and, per spectrum:

1. **Errors.** Spectra with an error column can trust it, rescale it to the
   estimate, or replace it with the estimate (``err_mode``). Spectra without
   errors always use the estimate. The estimate follows kaepora's
   ``genivar``: smooth, take |residual|, smooth that, then convert mean
   |residual| to sigma with a factor computed for this spectrum.
2. **Width.** Pick the Gaussian width (sigma = vexp * lambda, i.e. constant
   velocity) at the knee of the +-delta statistic, where it stops falling
   like smoothed noise ("knee"); or with kaepora's S/N rule ("kaepora"); or
   at the 100 A cap ("cap"); or set ``vexp`` by hand.
3. **Smooth + downsample.** Inverse-variance weighted Gaussian (kaepora's
   ``gsmooth``), truncated at ``nsig`` sigma, evaluated at one point per
   kernel sigma so the saved points are roughly independent. Clipped pixels
   (weight 0) get no weight; an output point gets weight 0 when more than
   ``max_bad_frac`` of its kernel weight falls on clipped pixels.

Nothing here depends on the instrument: only on whether a file has errors,
its pixel size, and its wavelength range.

A spectrum is a plain dict: ``{"name", "mjd", "wl", "flux", "err", "weight"}``
with ``err = None`` when the file has no usable errors.
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
SMOOTH_DIR = os.path.join(SPEC_ROOT, "3_spec_smoothed")
SMOOTH_LIST_DIR = os.path.join(SPEC_ROOT, "3_spec_lists_smoothed")

DEFAULTS = dict(
    # --- errors ---
    err_mode="trust",          # spectra WITH errors: "trust" | "rescale" | "estimate"
    rescale_shape="constant",  # for "rescale": "constant" | "linear" (in wavelength)
    est_vexp_smooth=0.002,     # kaepora genivar: smoothing whose residuals are the noise
    est_vexp_resid=0.015,      # kaepora genivar: smoothing of |residual|
    # --- width ---
    width_rule="knee",         # "knee" | "kaepora" | "cap"
    vexp=None,                 # a number here forces the width and skips the rule
    max_sigma_A=100.0,         # cap: kernel sigma never above this anywhere in the spectrum
    vexp_min=0.0003,           # smallest width tried (~90 km/s)
    n_widths=20,
    noise_slope=-2.0,          # knee: the statistic must first fall at least this steeply...
    knee_slope=-1.5,           # ...and the knee is the first width after that where it flattens past this
    stat_delta_vexp=0.0007,    # +-delta for the statistic, as a fraction of lambda (~200 km/s)
    # --- smoothing ---
    nsig=5.0,                  # kernel truncated at +-nsig sigma (kaepora)
    max_bad_frac=0.5,          # output weight 0 if more kernel weight than this is clipped
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
# Choosing the width
# ---------------------------------------------------------------------------

def vexp_cap(wl, max_sigma_A=100.0):
    """Largest vexp whose kernel sigma stays <= max_sigma_A everywhere."""
    return max_sigma_A / float(np.max(wl))


def width_statistic(spec, sigma, vexps, delta_vexp=0.0007, nsig=5.0, n_eval=2000):
    """Your +-delta statistic for each width.

    For each smoothing width: S = (|f(l - d)| + |f(l + d)|) / (2 |f(l)|)
    with d = delta_vexp * l, and the statistic is median |S - 1| over good
    pixels. S = 1 means no scatter left on scale d. Evaluated at ``n_eval``
    points spread evenly in ln(lambda).
    """
    wl = spec["wl"]
    good = spec["weight"] > 0
    lo, hi = wl[good][0] * (1 + 2 * delta_vexp), wl[good][-1] * (1 - 2 * delta_vexp)
    x = np.exp(np.linspace(np.log(lo), np.log(hi), n_eval))
    x3 = np.concatenate([x, x * (1 - delta_vexp), x * (1 + delta_vexp)])
    ivar = 1.0 / sigma ** 2
    stats = []
    for v in vexps:
        r = gsmooth(wl, spec["flux"], ivar, v, x_out=x3, nsig=nsig, good=good)
        f0, fm, fp = np.split(r["y"], 3)
        bad = np.split(r["bad_frac"], 3)[0]
        ok = np.isfinite(f0 + fm + fp) & (np.abs(f0) > 0) & (bad < 0.5)
        S = (np.abs(fm[ok]) + np.abs(fp[ok])) / (2 * np.abs(f0[ok]))
        stats.append(float(np.median(np.abs(S - 1))) if ok.any() else np.nan)
    return np.array(stats)


def stat_slope(vexps, stats):
    """Local slope d log(stat) / d log(vexp), averaged over 3 neighbours.

    Smoothed white noise gives about -2.5; a true plateau gives 0.
    """
    slope = np.gradient(np.log(stats), np.log(vexps))
    return np.convolve(np.pad(slope, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def knee_vexp(vexps, stats, noise_slope=-2.0, knee_slope=-1.5):
    """First width where the statistic stops falling like smoothed noise.

    The slope has to reach the noise regime (below ``noise_slope``) first;
    the knee is the first width after that with slope above ``knee_slope``.
    Returns the largest width tried (the cap) if there is no knee below it.
    """
    slope = stat_slope(vexps, stats)
    noisy = np.where(slope < noise_slope)[0]
    if len(noisy):
        after = np.where((np.arange(len(slope)) > noisy[0]) & (slope > knee_slope))[0]
        if len(after):
            return float(vexps[after[0]])
    return float(vexps[-1])


def kaepora_vexp(spec, sigma):
    """kaepora ``find_vexp``: vexp linear in S/N, 0.0045 at S/N 2.5 down to
    0.001 at S/N 80.

    Kaepora's S/N is per pixel on ~2 A pixels, so S/N is converted to per
    2 A here to keep the rule instrument-independent.
    """
    good = spec["weight"] > 0
    pix = np.median(np.diff(spec["wl"]))
    snr = np.median(spec["flux"][good] / sigma[good]) * np.sqrt(2.0 / pix)
    v = -4.51612903e-05 * snr + 4.61290323e-03
    return float(np.clip(v, 0.001, 0.0045)), float(snr)


# ---------------------------------------------------------------------------
# Whole spectrum
# ---------------------------------------------------------------------------

def downsample_grid(wl, vexp):
    """One point per kernel sigma: uniform in ln(lambda) with step vexp."""
    return np.exp(np.arange(np.log(wl[0]), np.log(wl[-1]), vexp))


def reduce_one(spec, params):
    """Errors -> width -> smooth -> downsample, for one spectrum.

    Returns a result dict with everything the plots and ``save_smoothed``
    need.
    """
    p = params
    if not is_usable(spec):
        raise ValueError("%s: almost every pixel is clipped (weight 0)" % spec["name"])
    wl = spec["wl"]
    good = spec["weight"] > 0
    sigma, err_info = choose_errors(spec, p)

    # width
    vmax = vexp_cap(wl, p["max_sigma_A"])
    vexps = np.geomspace(min(p["vexp_min"], vmax / 2), vmax, p["n_widths"])
    stats = width_statistic(spec, sigma, vexps, p["stat_delta_vexp"], p["nsig"])
    v_knee = knee_vexp(vexps, stats, p["noise_slope"], p["knee_slope"])
    v_kaepora, snr2A = kaepora_vexp(spec, sigma)
    v_kaepora = min(v_kaepora, vmax)
    if p["vexp"] is not None:
        vexp, rule = float(p["vexp"]), "manual"
    elif p["width_rule"] == "knee":
        vexp, rule = v_knee, "knee"
    elif p["width_rule"] == "kaepora":
        vexp, rule = v_kaepora, "kaepora"
    elif p["width_rule"] == "cap":
        vexp, rule = vmax, "cap"
    else:
        raise ValueError("width_rule must be 'knee', 'kaepora' or 'cap'")

    # smooth at full resolution (for the residual check) and downsampled
    ivar = 1.0 / sigma ** 2
    full = gsmooth(wl, spec["flux"], ivar, vexp, nsig=p["nsig"], good=good)
    x_out = downsample_grid(wl[good][[0, -1]], vexp)
    ds = gsmooth(wl, spec["flux"], ivar, vexp, x_out=x_out, nsig=p["nsig"], good=good)
    ok = np.isfinite(ds["y"]) & np.isfinite(ds["var"])
    weight_out = np.where(ok & (ds["bad_frac"] <= p["max_bad_frac"]), 1.0, 0.0)

    chi = (spec["flux"] - full["y"]) / sigma
    chi_good = chi[good & np.isfinite(chi)]
    chi_std = float(1.4826 * np.median(np.abs(chi_good - np.median(chi_good))))

    return {
        "spec": spec, "params": dict(p), "sigma": sigma, "err_info": err_info,
        "vexps": vexps, "stats": stats, "vexp_max": vmax,
        "vexp_knee": v_knee, "vexp_kaepora": v_kaepora, "snr_per_2A": snr2A,
        "vexp": vexp, "width_rule_used": rule,
        "full_smooth": full["y"], "chi": chi, "chi_std": chi_std,
        "out_wl": x_out[ok], "out_flux": ds["y"][ok],
        "out_err": np.sqrt(ds["var"][ok]), "out_weight": weight_out[ok],
    }


def summary_row(res):
    """One-line summary (for a table of all spectra)."""
    e = res["err_info"]
    return {
        "name": res["spec"]["name"], "mjd": res["spec"]["mjd"],
        "err_mode": e["mode"],
        "est/real": round(e.get("ratio_median", np.nan), 2),
        "vexp": round(res["vexp"], 5), "rule": res["width_rule_used"],
        "sigma@mid [A]": round(res["vexp"] * np.median(res["spec"]["wl"]), 1),
        "vexp_knee": round(res["vexp_knee"], 5),
        "vexp_kaepora": round(res["vexp_kaepora"], 5),
        "n_out": len(res["out_wl"]),
        "frac_w0": round(1 - res["out_weight"].mean(), 2),
        "chi_std": round(res["chi_std"], 2),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def settings_path(snname):
    return os.path.join(SMOOTH_DIR, snname, snname + "_smooth_settings.json")


def load_settings(snname):
    path = settings_path(snname)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_smoothed(snname, res, customised=False, note=""):
    """Write ``3_spec_smoothed/<SN>/<file>`` (wls flux flux_err weight),
    record the settings, and rewrite the list file."""
    spec = res["spec"]
    out_dir = os.path.join(SMOOTH_DIR, snname)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(SMOOTH_LIST_DIR, exist_ok=True)
    np.savetxt(os.path.join(out_dir, spec["name"]),
               np.column_stack([res["out_wl"], res["out_flux"],
                                res["out_err"], res["out_weight"]]),
               fmt=["%.4f", "%.6e", "%.6e", "%d"],
               header="wls\tflux\tflux_err\tweight", delimiter="\t")

    e = res["err_info"]
    p = {k: v for k, v in res["params"].items()}
    allset = load_settings(snname)
    allset[spec["name"]] = {
        "mjd": spec["mjd"], "customised": bool(customised), "note": note,
        "has_errors": e["has_errors"], "err_mode": e["mode"],
        "rescale": e["scale"], "est_over_real_median": e.get("ratio_median"),
        "err_factor_median": float(np.nanmedian(e["est"]["factor"])),
        "vexp": res["vexp"], "width_rule": res["width_rule_used"],
        "vexp_knee": res["vexp_knee"], "vexp_kaepora": res["vexp_kaepora"],
        "vexp_max": res["vexp_max"], "chi_std": res["chi_std"],
        "n_out": int(len(res["out_wl"])),
        "frac_weight0": float(1 - res["out_weight"].mean()),
        "params": p,
    }
    with open(settings_path(snname), "w") as f:
        json.dump(allset, f, indent=1, sort_keys=True)

    rows = sorted((s["mjd"], n) for n, s in allset.items())
    mjd0 = rows[0][0]
    with open(os.path.join(SMOOTH_LIST_DIR, snname + ".list"), "w") as f:
        for m, n in rows:
            path = os.path.relpath(os.path.join(out_dir, n), pconf.COCO_PATH)
            f.write("%.6f\t%.6f\t%s\n" % (m, m - mjd0, path))
    print("saved %s | %s | vexp %.4f (%s) | %d points"
          % (spec["name"], e["mode"], res["vexp"], res["width_rule_used"],
             len(res["out_wl"])))


# ---------------------------------------------------------------------------
# Plotting (each takes an ``ax``)
# ---------------------------------------------------------------------------

def _ylim(ax, y, lo=1, hi=99, pad=0.15):
    y = y[np.isfinite(y)]
    if len(y):
        a, b = np.percentile(y, [lo, hi])
        ax.set_ylim(a - pad * (b - a), b + pad * (b - a))


def plot_errors(res, ax=None):
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
    _ylim(ax, np.concatenate([e["sigma_est"][good], res["sigma"][good]]), 1, 99)
    ax.set_ylabel("$\\sigma$")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_width(res, ax=None):
    """The +-delta statistic against width (black), its local slope (grey,
    right axis), and each rule's choice."""
    ax = ax or plt.subplots(figsize=(5, 3))[1]
    p = res["params"]
    ax.plot(res["vexps"], res["stats"], "o-", color="k", ms=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vexp  ($\\sigma$ = vexp $\\cdot\\lambda$)")
    ax.set_ylabel("median |S - 1|")
    ax2 = ax.twinx()
    ax2.plot(res["vexps"], stat_slope(res["vexps"], res["stats"]), "-",
             color="0.6", lw=1)
    ax2.axhline(p["knee_slope"], color="0.6", ls=":", lw=0.8)
    ax2.axhline(-2.5, color="0.8", ls=":", lw=0.8)
    ax2.set_ylim(-3.5, 0.5)
    ax2.set_ylabel("slope", color="0.5", fontsize=8)
    for v, c, lab in [(res["vexp_knee"], "tab:green", "knee"),
                      (res["vexp_kaepora"], "tab:purple", "kaepora"),
                      (res["vexp_max"], "0.4", "cap (%g A)" % p["max_sigma_A"])]:
        ax.axvline(v, color=c, ls="--", lw=1, label="%s %.4f" % (lab, v))
    ax.axvline(res["vexp"], color="tab:red", lw=3, alpha=0.4,
               label="used (%s)" % res["width_rule_used"])
    ax.legend(fontsize=7, loc="lower left")
    return ax


def plot_smoothed(res, ax=None):
    """Clipped data, smoothed curve, and the saved downsampled points."""
    ax = ax or plt.subplots(figsize=(13, 4))[1]
    spec = res["spec"]
    wl, good = spec["wl"], spec["weight"] > 0
    ax.plot(wl, np.where(good, spec["flux"], np.nan), color="0.75", lw=0.4,
            label="clipped data")
    ax.plot(wl, np.where(good, res["full_smooth"], np.nan), color="tab:red",
            lw=0.8, label="smoothed, vexp %.4f" % res["vexp"])
    w1 = res["out_weight"] > 0
    ax.errorbar(res["out_wl"][w1], res["out_flux"][w1], res["out_err"][w1],
                fmt="o", ms=2, color="k", lw=0.8, label="saved points (%d)" % w1.sum())
    if (~w1).any():
        ax.plot(res["out_wl"][~w1], res["out_flux"][~w1], "x", color="tab:orange",
                ms=4, label="saved, weight 0 (%d)" % (~w1).sum())
    _ylim(ax, spec["flux"][good])
    ax.set_ylabel("Flux")
    ax.set_title(spec["name"], fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_chi(res, ax=None):
    """(data - smoothed) / sigma: should scatter around +-1 if errors are right."""
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


def plot_summary(res, figsize=(15, 10)):
    """All four panels for one spectrum. Returns (fig, axes dict)."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 0.7], hspace=0.35, wspace=0.3)
    axs = {"smooth": fig.add_subplot(gs[0, :])}
    axs["errors"] = fig.add_subplot(gs[1, :3], sharex=axs["smooth"])
    axs["width"] = fig.add_subplot(gs[1, 3])
    axs["chi"] = fig.add_subplot(gs[2, :3], sharex=axs["smooth"])
    plot_smoothed(res, axs["smooth"])
    plot_errors(res, axs["errors"])
    plot_width(res, axs["width"])
    plot_chi(res, axs["chi"])
    return fig, axs
