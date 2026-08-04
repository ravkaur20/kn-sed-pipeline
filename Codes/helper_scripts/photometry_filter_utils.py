"""Per-filter MJD span from **raw** photometry only (for mangling eligibility)."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "filter_mjd_ranges_dict_from_raw_file",
    "write_band_mjd_ranges_json",
    "load_band_mjd_ranges_json",
]


def filter_mjd_ranges_dict_from_raw_file(photometry_path: str) -> dict[str, dict[str, float]]:
    """Global min/max MJD per ``band`` column (CSV/TSV with header: MJD, …, band, …)."""
    if not os.path.isfile(photometry_path):
        raise FileNotFoundError(photometry_path)
    df = pd.read_csv(photometry_path)
    if "MJD" not in df.columns or "band" not in df.columns:
        raise ValueError("Expected columns MJD and band in %s" % photometry_path)
    mjd = pd.to_numeric(df["MJD"], errors="coerce")
    bands = df["band"].astype(str)
    ok = np.isfinite(mjd.values)
    out: dict[str, dict[str, Any]] = {}
    for b in np.unique(bands[ok].values):
        m = (bands.values == b) & ok
        if not np.any(m):
            continue
        t = mjd.values[m].astype(float)
        out[str(b)] = {
            "min_mjd": float(np.min(t)),
            "max_mjd": float(np.max(t)),
            "n_obs": int(np.sum(m)),
        }
    return out


def write_band_mjd_ranges_json(photometry_path: str, json_path: str) -> dict:
    d = filter_mjd_ranges_dict_from_raw_file(photometry_path)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    payload = {
        "source_photometry_file": os.path.abspath(photometry_path),
        "bands": d,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_band_mjd_ranges_json(json_path: str) -> dict[str, dict[str, float]]:
    """Return the inner ``bands`` dict suitable for ``filter_mjd_dict`` in mangle code."""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    bands = payload.get("bands", payload)
    # Normalize to min_mjd / max_mjd only for consumers
    return {
        k: {"min_mjd": float(v["min_mjd"]), "max_mjd": float(v["max_mjd"])}
        for k, v in bands.items()
    }
