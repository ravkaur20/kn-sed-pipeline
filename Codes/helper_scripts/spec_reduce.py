"""Small helpers for reducing input spectra one at a time (notebook 0.2).

A spectrum is a plain dict: ``{"name", "wl", "flux", "err"}``. ``err`` is
``None`` when the file has no uncertainty column (e.g. LDSS3).

A list of regions is a plain list of ``(wl_lo, wl_hi)`` tuples in Angstrom, so
it can be printed, edited by hand, and saved to JSON.

Clipping step:
    1. ``propose_clips`` suggests regions (spikes, low S/N, telluric) and edges.
    2. You edit those lists by hand in the notebook.
    3. ``apply_clip`` trims the ends and bridges the flux across each interior
       region (a straight line between local medians); those pixels get
       ``weight = 0``.
    4. ``save_clipped`` writes the spectrum and records your choices.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

import pipeline_config as pconf

SPEC_ROOT = pconf.spectroscopy_root()
RAW_DIR = os.path.join(SPEC_ROOT, "1_spec_original")
RAW_LIST_DIR = os.path.join(SPEC_ROOT, "1_spec_lists_original")
CLIP_DIR = os.path.join(SPEC_ROOT, "1.5_spec_clipped")
CLIP_LIST_DIR = os.path.join(SPEC_ROOT, "1.5_spec_lists_clipped")

# Common telluric absorption bands (observed frame, Angstrom). Edit freely.
TELLURIC_BANDS = [
    (6860, 6890),    # O2 B band
    (7590, 7700),    # O2 A band
    (9300, 9650),    # H2O
    (11100, 11500),  # H2O (between z and J)
    (13300, 14500),  # H2O (between J and H)
    (17900, 19600),  # H2O (between H and K)
]

# Input files that are already de-redshifted (rest frame), matched by a
# substring of the file name. ``load_raw`` moves them to the observed frame
# (wl * (1 + z)) so every spectrum starts in the same frame as the photometry.
# AT2017gfo LDSS3: O2 A/B telluric residuals sit at 7594/(1+z), 6867/(1+z).
REST_FRAME_FILES = {
    "AT2017gfo": ["LDSS3"],
}

# Shading colours for each kind of proposed region.
REGION_COLORS = {
    "spike": "tab:red",
    "low_snr": "tab:orange",
    "telluric": "tab:blue",
    "manual": "tab:purple",
    "clip": "tab:green",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def list_spectra(snname):
    """Return [(mjd, filename), ...] sorted by MJD, from the original list."""
    path = os.path.join(RAW_LIST_DIR, snname + ".list")
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            rows.append((float(parts[2]), os.path.basename(parts[0])))
    return sorted(rows)


def load_raw(snname, filename):
    """Load an original spectrum.

    2-column files: wavelength, flux (no errors).
    4-column XSHOOTER files: wavelength, telluric-corrected flux,
    uncorrected flux, sigma. We use the telluric-corrected flux.

    Files listed in ``REST_FRAME_FILES`` are shifted to the observed frame.
    Only the wavelengths are shifted; any (1+z) flux factor is a constant
    that mangling absorbs.
    """
    data = np.genfromtxt(os.path.join(RAW_DIR, snname, filename),
                         comments="#", skip_header=1)
    wl, flux = data[:, 0], data[:, 1]
    err = data[:, 3] if data.shape[1] >= 4 else None

    good = np.isfinite(wl) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err)
    order = np.argsort(wl[good])
    spec = {
        "name": filename,
        "wl": wl[good][order],
        "flux": flux[good][order],
        "err": None if err is None else err[good][order],
    }
    if spec["err"] is not None and not np.any(spec["err"] > 0):
        spec["err"] = None

    spec["shifted_to_observed"] = is_rest_frame(snname, filename)
    if spec["shifted_to_observed"]:
        z = pconf.SN_REDSHIFT[snname]
        spec["wl"] = spec["wl"] * (1 + z)
        print("%s is rest frame: shifted to observed frame (z = %g)"
              % (filename, z))
    return spec


def is_rest_frame(snname, filename):
    """True if this input file is listed as already de-redshifted."""
    return any(key in filename for key in REST_FRAME_FILES.get(snname, []))


def pixel_size(wl):
    """Median pixel size in Angstrom."""
    return float(np.median(np.diff(wl)))


def n_pixels(wl, width_A):
    """Number of pixels spanning ``width_A`` Angstrom (at least 1, odd)."""
    n = max(1, int(round(width_A / pixel_size(wl))))
    return n + 1 if n % 2 == 0 else n


# ---------------------------------------------------------------------------
# Masks <-> regions
# ---------------------------------------------------------------------------

def mask_runs(mask):
    """Index ranges [(i_first, i_last), ...] of each run of True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    steps = np.diff(mask.astype(int))
    starts = list(np.where(steps == 1)[0] + 1)
    stops = list(np.where(steps == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask) - 1)
    return list(zip(starts, stops))


def mask_to_regions(wl, mask):
    """Turn a boolean mask into a list of (wl_lo, wl_hi) tuples.

    Edges are rounded outward to 0.1 A so every flagged pixel stays inside.
    """
    return [(np.floor(wl[i0] * 10) / 10, np.ceil(wl[i1] * 10) / 10)
            for i0, i1 in mask_runs(mask)]


def regions_to_mask(wl, regions):
    """Boolean mask that is True inside any of the regions."""
    mask = np.zeros(len(wl), dtype=bool)
    for lo, hi in regions:
        mask |= (wl >= lo) & (wl <= hi)
    return mask


def merge_regions(regions, gap_A=0.0):
    """Sort regions and merge any that overlap or are closer than ``gap_A``."""
    merged = []
    for lo, hi in sorted(regions):
        if merged and lo <= merged[-1][1] + gap_A:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def remove_between(regions, lo, hi):
    """Drop every region that lies entirely inside [lo, hi]."""
    return [(a, b) for a, b in regions if not (a >= lo and b <= hi)]


def drop_narrow(regions, min_width_A):
    """Remove regions narrower than ``min_width_A``."""
    return [(lo, hi) for lo, hi in regions if hi - lo >= min_width_A]


# ---------------------------------------------------------------------------
# Finding things to clip
# ---------------------------------------------------------------------------

def rolling_median(spec, window_A):
    """Running median of the flux over ``window_A`` Angstrom."""
    return median_filter(spec["flux"], size=n_pixels(spec["wl"], window_A),
                         mode="nearest")


def estimate_noise(spec, window_A=50.0):
    """Per-pixel noise estimated from the data (ignores ``spec['err']``).

    Scatter of the flux about its running median, measured with a running
    median absolute deviation (robust to spikes).
    """
    n = n_pixels(spec["wl"], window_A)
    resid = spec["flux"] - median_filter(spec["flux"], size=n, mode="nearest")
    return 1.4826 * median_filter(np.abs(resid), size=n, mode="nearest")


def find_spikes(spec, window_A=50.0, nsigma=5.0, grow_A=2.0):
    """Regions where the flux jumps more than ``nsigma`` from the running
    median. Each flagged pixel is grown by ``grow_A`` on both sides."""
    resid = spec["flux"] - rolling_median(spec, window_A)
    noise = estimate_noise(spec, window_A)
    regions = mask_to_regions(spec["wl"], np.abs(resid) > nsigma * noise)
    return merge_regions([(lo - grow_A, hi + grow_A) for lo, hi in regions])


def snr_per_bin(spec, bin_A=10.0, window_A=50.0, use_input_err=True):
    """S/N per ``bin_A``-Angstrom bin, so instruments with different pixel
    sizes can share one threshold.

    Uses the input errors if the file has them (and ``use_input_err``),
    otherwise the scatter from ``estimate_noise``.
    """
    flux = rolling_median(spec, window_A)
    if use_input_err and spec["err"] is not None:
        noise = median_filter(spec["err"], size=n_pixels(spec["wl"], window_A),
                              mode="nearest")
    else:
        noise = estimate_noise(spec, window_A)
    pix_per_bin = bin_A / pixel_size(spec["wl"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return flux / noise * np.sqrt(pix_per_bin)


def find_low_snr(spec, snr_min=5.0, bin_A=10.0, window_A=50.0,
                 min_width_A=20.0, use_input_err=True):
    """Stretches at least ``min_width_A`` wide with S/N per bin below
    ``snr_min``."""
    snr = snr_per_bin(spec, bin_A, window_A, use_input_err)
    regions = mask_to_regions(spec["wl"], ~(snr >= snr_min))
    return drop_narrow(regions, min_width_A)


def find_edges(spec, snr_min=5.0, hold_A=100.0, bin_A=10.0, window_A=50.0,
               use_input_err=True):
    """Blue and red cut wavelengths.

    The blue cut is where S/N first stays above ``snr_min`` for at least
    ``hold_A`` Angstrom; the red cut is where it last does.
    Returns (wl_blue, wl_red); falls back to the full range.
    """
    wl = spec["wl"]
    snr = snr_per_bin(spec, bin_A, window_A, use_input_err)
    good_runs = drop_narrow(mask_to_regions(wl, snr >= snr_min), hold_A)
    if not good_runs:
        return (float(wl[0]), float(wl[-1]))
    return (good_runs[0][0], good_runs[-1][1])


def find_telluric(spec, bands=TELLURIC_BANDS):
    """Telluric bands that fall inside this spectrum's coverage."""
    lo, hi = spec["wl"][0], spec["wl"][-1]
    return [(max(a, lo), min(b, hi)) for a, b in bands if b > lo and a < hi]


def propose_clips(spec, params):
    """Run every finder with the settings in ``params`` (a dict).

    Returns (proposed, edges), where ``proposed`` is
    ``{"spike": [...], "low_snr": [...], "telluric": [...]}``.
    """
    p = params
    proposed = {
        "spike": find_spikes(spec, p["spike_window_A"], p["spike_nsigma"],
                             p["spike_grow_A"]),
        "low_snr": find_low_snr(spec, p["snr_min"], p["snr_bin_A"],
                                p["snr_window_A"], p["low_snr_min_width_A"],
                                p["use_input_err"]),
        "telluric": find_telluric(spec, p["telluric_bands"]),
    }
    edges = find_edges(spec, p["snr_min"], p["edge_hold_A"], p["snr_bin_A"],
                       p["snr_window_A"], p["use_input_err"])
    return proposed, edges


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def bridge(wl, y, bad, anchor_A=10.0):
    """Fill each run of ``bad`` pixels with a straight line between the median
    of the good values within ``anchor_A`` on either side (so the line isn't
    pinned to one noisy pixel)."""
    y = y.copy()
    good = ~bad
    for i0, i1 in mask_runs(bad):
        lo, hi = wl[i0], wl[i1]
        left = good & (wl < lo) & (wl >= lo - anchor_A)
        right = good & (wl > hi) & (wl <= hi + anchor_A)
        # No good pixels that close: fall back to the nearest good one.
        if not left.any() and good[:i0].any():
            left = np.arange(len(wl)) == np.where(good[:i0])[0][-1]
        if not right.any() and good[i1 + 1:].any():
            right = np.arange(len(wl)) == i1 + 1 + np.where(good[i1 + 1:])[0][0]
        anchors_x, anchors_y = [], []
        if left.any():
            anchors_x.append(np.median(wl[left]))
            anchors_y.append(np.median(y[left]))
        if right.any():
            anchors_x.append(np.median(wl[right]))
            anchors_y.append(np.median(y[right]))
        if anchors_x:
            y[i0:i1 + 1] = np.interp(wl[i0:i1 + 1], anchors_x, anchors_y)
    return y


def apply_clip(spec, regions, edges, anchor_A=10.0):
    """Trim to ``edges`` and bridge the flux across ``regions``.

    Returns a new spectrum dict with an extra ``weight`` array:
    1 = real data, 0 = interpolated (should be ignored downstream).
    """
    blue, red = edges
    keep = (spec["wl"] >= blue) & (spec["wl"] <= red)
    wl = spec["wl"][keep]
    flux = spec["flux"][keep]
    err = None if spec["err"] is None else spec["err"][keep]

    bad = regions_to_mask(wl, regions)
    if bad.all():
        print("WARNING: every pixel inside the edges is clipped; "
              "nothing to keep for %s" % spec["name"])
    flux = bridge(wl, flux, bad, anchor_A)
    if err is not None:
        err = bridge(wl, err, bad, anchor_A)

    return {"name": spec["name"], "wl": wl, "flux": flux, "err": err,
            "weight": (~bad).astype(float),
            "shifted_to_observed": spec.get("shifted_to_observed", False)}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def choices_path(snname):
    return os.path.join(CLIP_DIR, snname, snname + "_clip_regions.json")


def load_choices(snname, filename=None):
    """Saved clip choices: all of them, or the entry for one file (or None)."""
    path = choices_path(snname)
    if not os.path.exists(path):
        return {} if filename is None else None
    with open(path) as f:
        allc = json.load(f)
    return allc if filename is None else allc.get(filename)


def save_clipped(snname, mjd, clipped, regions, edges, note=""):
    """Write the clipped spectrum, record the choices, update the list file.

    The spectrum file has columns ``wls flux flux_err weight``, always in the
    observed frame; ``flux_err`` is NaN when the input had no errors (filled
    in at the error step).
    """
    out_dir = os.path.join(CLIP_DIR, snname)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CLIP_LIST_DIR, exist_ok=True)

    name = clipped["name"]
    err = clipped["err"] if clipped["err"] is not None \
        else np.full(len(clipped["wl"]), np.nan)
    np.savetxt(os.path.join(out_dir, name),
               np.column_stack([clipped["wl"], clipped["flux"], err,
                                clipped["weight"]]),
               fmt=["%.4f", "%.6e", "%.6e", "%d"],
               header="wls\tflux\tflux_err\tweight", delimiter="\t")

    allc = load_choices(snname)
    allc[name] = {"mjd": mjd, "edges": list(edges),
                  "shifted_to_observed": bool(clipped.get("shifted_to_observed")),
                  "regions": [list(r) for r in merge_regions(regions)],
                  "note": note}
    with open(choices_path(snname), "w") as f:
        json.dump(allc, f, indent=1, sort_keys=True)

    # List file: one line per saved spectrum, sorted by MJD.
    rows = sorted((c["mjd"], n) for n, c in allc.items())
    mjd0 = rows[0][0]
    with open(os.path.join(CLIP_LIST_DIR, snname + ".list"), "w") as f:
        for m, n in rows:
            path = os.path.relpath(os.path.join(out_dir, n), pconf.COCO_PATH)
            f.write("%.6f\t%.6f\t%s\n" % (m, m - mjd0, path))
    print("saved", name, "| %d regions, edges %.0f-%.0f A"
          % (len(allc[name]["regions"]), edges[0], edges[1]))


# ---------------------------------------------------------------------------
# Plotting (each takes an ``ax`` so you can restyle or zoom per cell)
# ---------------------------------------------------------------------------

def new_ax(figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return ax


def robust_ylim(ax, flux, lo_pct=0.5, hi_pct=99.5, pad=0.1):
    """Set y limits from percentiles so a few spikes don't squash the plot."""
    lo, hi = np.nanpercentile(flux, [lo_pct, hi_pct])
    span = hi - lo
    ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_raw(spec, ax=None, show_err=True, edges=None, **kw):
    """The original spectrum (and its error, if any)."""
    ax = ax or new_ax()
    ax.plot(spec["wl"], spec["flux"], color="k", lw=0.5, label="raw flux", **kw)
    if show_err and spec["err"] is not None:
        ax.plot(spec["wl"], spec["err"], color="0.6", lw=0.5, label="input error")
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    inside = np.ones(len(spec["wl"]), dtype=bool) if edges is None else \
        (spec["wl"] >= edges[0]) & (spec["wl"] <= edges[1])
    robust_ylim(ax, spec["flux"][inside])
    ax.set_xlabel("Observed wavelength [$\\AA$]")
    ax.set_ylabel("Flux")
    ax.set_title(spec["name"], fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def shade(ax, regions, color, label=None, alpha=0.25):
    """Shade each (lo, hi) region; only the first one gets a legend label."""
    for i, (lo, hi) in enumerate(regions):
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0,
                   label=label if i == 0 else None)


def plot_proposed(spec, proposed, edges, ax=None):
    """Raw spectrum with each kind of proposed region shaded in its own
    colour, and the area outside the edges greyed out."""
    ax = plot_raw(spec, ax=ax, show_err=False, edges=edges)
    for kind, regions in proposed.items():
        shade(ax, regions, REGION_COLORS.get(kind, "tab:gray"),
              label="%s (%d)" % (kind, len(regions)))
    shade(ax, [(spec["wl"][0], edges[0]), (edges[1], spec["wl"][-1])],
          "0.3", label="trimmed ends", alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_clipped(spec, clipped, ax=None):
    """Raw spectrum in grey, kept data in black, interpolated parts in red."""
    ax = ax or new_ax()
    ax.plot(spec["wl"], spec["flux"], color="0.75", lw=0.5, label="raw")
    flux_kept = np.where(clipped["weight"] > 0, clipped["flux"], np.nan)
    flux_interp = np.where(clipped["weight"] == 0, clipped["flux"], np.nan)
    ax.plot(clipped["wl"], flux_kept, color="k", lw=0.6, label="kept")
    ax.plot(clipped["wl"], flux_interp, color="tab:red", lw=1.2,
            label="interpolated (weight 0)")
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    robust_ylim(ax, clipped["flux"])
    ax.set_xlabel("Observed wavelength [$\\AA$]")
    ax.set_ylabel("Flux")
    ax.set_title(spec["name"] + "  (clipped)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_snr(spec, params, ax=None):
    """S/N per bin with the threshold, to see why low_snr/edges were chosen."""
    ax = ax or new_ax(figsize=(13, 2.5))
    snr = snr_per_bin(spec, params["snr_bin_A"], params["snr_window_A"],
                      params["use_input_err"])
    ax.plot(spec["wl"], snr, color="k", lw=0.6)
    ax.axhline(params["snr_min"], color="tab:orange", ls="--", lw=1,
               label="snr_min = %g" % params["snr_min"])
    ax.set_yscale("log")
    ax.set_ylabel("S/N per %g $\\AA$" % params["snr_bin_A"])
    ax.set_xlabel("Observed wavelength [$\\AA$]")
    ax.legend(fontsize=8, loc="upper right")
    return ax
