"""Review of AT2017gfo spectra in an Open Kilonova Catalog JSON (review notebook).

Read-only: nothing here writes into Inputs/. Functions to
- load every spectrum in the catalog JSON with its metadata and paper,
- test which frame a spectrum is in and whether tellurics were removed
  (depth of the O2 A and B bands at their observed vs rest-frame wavelengths),
- compare a catalog spectrum with a local file (identical? flux ratio per band),
- find pairs that are the same data at a different flux scale,
- check whether a spectrum was dereddened (ratio to the as-observed copy vs
  the Milky Way extinction curve).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

# Short names for the papers the catalog cites (ADS bibcodes, incl. arXiv versions)
PAPERS = {
    "2017arXiv171005432S": "Shappee+2017 (Science 358, 1574)",
    "2017Sci...358.1574S": "Shappee+2017 (Science 358, 1574)",
    "2017ApJ...848L..18N": "Nicholl+2017 (ApJL 848, L18)",
    "2017arXiv171005841S": "Smartt+2017 (Nature 551, 75)",
    "2017Natur.551...75S": "Smartt+2017 (Nature 551, 75)",
    "2017Natur.551...67P": "Pian+2017 (Nature 551, 67)",
    "2017ApJ...848L..19C": "Chornock+2017 (ApJL 848, L19)",
    "2017Sci...358.1559K": "Kasliwal+2017 (Science 358, 1559)",
    "2012PASP..124..668Y": "WISeREP",
}

O2_A = (7594.0, 7640.0)
O2_B = (6867.0, 6890.0)


def load_catalog(path):
    """Return (event dict, {source alias: bibcode or name})."""
    with open(path) as f:
        data = json.load(f)
    ev = data[list(data)[0]]
    src = {s["alias"]: (s.get("bibcode") or s.get("name") or s.get("url", "")) for s in ev["sources"]}
    return ev, src


def papers_for(source_field, src):
    names = []
    for a in str(source_field).split(","):
        b = src.get(a.strip(), a)
        names.append(PAPERS.get(b, b))
    return "; ".join(dict.fromkeys(names))


def spectrum_array(sp):
    d = np.array(sp["data"], dtype=float)
    return d[:, 0], d[:, 1]


def telluric_depths(w, f, band, z, width_A=120.0):
    """Mean fractional depth of a telluric band at its observed wavelengths and
    at the wavelengths it would have if the spectrum were de-redshifted.
    Positive = flux dips below the local continuum. None if not covered."""
    lo, hi = band[0] / (1 + z) - 150, band[1] + 150
    s = (w > lo) & (w < hi)
    if s.sum() < 20:
        return None
    ww, ff = w[s], f[s]
    pix = np.median(np.diff(ww))
    n = max(5, int(width_A / pix)) | 1
    r = ff / median_filter(ff, n, mode="nearest")

    def depth(a, b):
        t = (ww > a) & (ww < b)
        return float(1 - np.nanmedian(r[t])) if t.sum() > 2 else np.nan

    return depth(*band), depth(band[0] / (1 + z), band[1] / (1 + z))


def frame_verdict(a_band, b_band, thresh=0.03):
    """Read the O2 depths: which frame, and were tellurics removed?

    Uses the A band (the B band is too noisy at late times) and falls back to
    the B band only when the A band is not covered. Calibrated on local files:
    the observed-frame ENGRAVE X-shooter spectra never show a rest-position
    A-band dip above 0.012 at any epoch (so kilonova features do not mimic
    it), while the rest-frame LDSS3 spectra show 0.03-0.28.
    """
    band, tag = (a_band, "") if a_band is not None else (b_band, " (B band only)")
    if band is None:
        return "no O2 coverage"
    obs, rest = band
    if not (np.isfinite(obs) and np.isfinite(rest)) or max(abs(obs), abs(rest)) > 1:
        return "unreliable (continuum near zero)"
    if rest > thresh and rest > obs + 0.02:
        return "rest frame (O2 dip at λ/(1+z))" + tag
    if obs > 0.2:
        return "observed, tellurics NOT removed" + tag
    if obs > thresh:
        return "observed, telluric residuals" + tag
    if obs < -thresh:
        return "observed, tellurics over-corrected" + tag
    return "no clear O2 signal (corrected?)" + tag


def inventory(ev, src, z):
    rows = []
    for i, sp in enumerate(ev["spectra"]):
        w, f = spectrum_array(sp)
        a = telluric_depths(w, f, O2_A, z)
        b = telluric_depths(w, f, O2_B, z)
        rows.append(dict(
            idx=i,
            mjd_json=float(sp["time"]),
            instrument=sp.get("instrument", ""),
            telescope=sp.get("telescope", ""),
            filename=sp.get("filename", ""),
            paper=papers_for(sp.get("source", ""), src),
            wl_min=float(w.min()),
            wl_max=float(w.max()),
            n_pix=len(w),
            pix_A=float(np.median(np.diff(w))),
            has_errors=len(sp["data"][0]) > 2,
            O2A_obs=None if a is None else round(a[0], 3),
            O2A_rest=None if a is None else round(a[1], 3),
            frame=frame_verdict(a, b),
        ))
    return pd.DataFrame(rows)


def read_local(path):
    with open(path) as fh:
        first = fh.readline()
    d = np.loadtxt(path, comments="#", skiprows=0 if first.strip()[:1].isdigit() else 1)
    return d


def identical(w1, f1, w2, f2, rtol=1e-6):
    return len(w1) == len(w2) and np.allclose(w1, w2, rtol=rtol, atol=0) and np.allclose(f1, f2, rtol=rtol, atol=0)


def band_ratios(w1, f1, w2, f2, bands, nbin=25):
    """Median of binned f1/f2 in each (lo, hi) band (NaN where not covered)."""
    out = []
    for lo, hi in bands:
        e = np.linspace(lo, hi, nbin + 1)
        r = []
        for k in range(nbin):
            s1 = (w1 > e[k]) & (w1 < e[k + 1])
            s2 = (w2 > e[k]) & (w2 < e[k + 1])
            if s1.sum() > 2 and s2.sum() > 2:
                r.append(np.median(f1[s1]) / np.median(f2[s2]))
        out.append(float(np.nanmedian(r)) if r else np.nan)
    return out


def same_data_pairs(ev, max_scatter_dex=0.005, min_overlap_A=1000.0):
    """Pairs whose spectra agree in shape to better than ``max_scatter_dex``:
    the same data, possibly at a different flux scale."""
    D = [spectrum_array(sp) for sp in ev["spectra"]]
    out = []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            (wa, fa), (wb, fb) = D[i], D[j]
            lo, hi = max(wa[0], wb[0]), min(wa[-1], wb[-1])
            if hi - lo < min_overlap_A:
                continue
            g = np.linspace(lo, hi, 400)
            a, b = np.interp(g, wa, fa), np.interp(g, wb, fb)
            ok = (a > 0) & (b > 0)
            if ok.sum() < 100:
                continue
            r = np.log10(a[ok] / b[ok])
            if np.std(r) < max_scatter_dex:
                out.append(dict(i=i, j=j, scatter_dex=float(np.std(r)), scale=float(10 ** np.median(r))))
    return pd.DataFrame(out)


def mw_extinction_factor(w, ebv, rv=3.1):
    """10^(0.4 A_lambda): what MW dereddening multiplies the flux by (CCM89)."""
    import extinction
    a = extinction.ccm89(np.asarray(w, dtype=float), rv * ebv, rv)
    return 10 ** (0.4 * a)


# ---------------------------------------------------------------------------
# WISeREP downloads (wiserep_manifest.csv written next to the files)
# ---------------------------------------------------------------------------

def read_wiserep(path):
    """(wl, flux, err or None) from a WISeREP ascii/csv file. Selsing/ENGRAVE
    files: wl, flux (telluric corrected), flux (not corrected), error.
    PESSTO csv: WAVE, FLUX, ERR, BGFLUX. Others: wl, flux."""
    if path.endswith(".csv"):
        d = np.genfromtxt(path, delimiter=",", names=True)
        return d["WAVE"], d["FLUX"], d["ERR"]
    d = np.loadtxt(path, comments="#")
    err = d[:, 3] if d.shape[1] >= 4 else (d[:, 2] if d.shape[1] == 3 else None)
    return d[:, 0], d[:, 1], err


def compare_versions(w1, f1, w2, f2, bands=None, nbin=25):
    """How two versions of one spectrum differ: same pixels? same flux?
    Median flux ratio 1/2 in bands, and the scatter of the ratio (shape)."""
    out = dict(same_pixels=bool(len(w1) == len(w2) and np.allclose(w1, w2, rtol=1e-7, atol=0)))
    out["identical"] = bool(out["same_pixels"] and np.allclose(f1, f2, rtol=1e-4, atol=0))
    lo, hi = max(w1.min(), w2.min()), min(w1.max(), w2.max())
    out["overlap"] = "%.0f-%.0f" % (lo, hi)
    if hi - lo < 200:
        return out
    if bands is None:
        bands = [(lo + k * (hi - lo) / 4, lo + (k + 1) * (hi - lo) / 4) for k in range(4)]
    r = band_ratios(w1, f1, w2, f2, bands, nbin)
    out["ratio_by_quarter"] = [round(x, 3) for x in r]
    out["ratio_median"] = round(float(np.nanmedian(r)), 3)
    e = np.linspace(lo, hi, 4 * nbin + 1)
    rr = band_ratios(w1, f1, w2, f2, list(zip(e[:-1], e[1:])), 1)
    rr = np.array([x for x in rr if np.isfinite(x) and x > 0])
    out["shape_scatter_dex"] = round(float(np.std(np.log10(rr / np.median(rr)))), 3) if rr.size > 5 else np.nan
    return out
