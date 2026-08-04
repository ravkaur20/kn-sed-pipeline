"""Per-band GP kernel settings for LC fit (step 3)."""

from __future__ import annotations

import json
import os
from typing import Any

_UV_BANDS = frozenset(
    {"swift_UVW1", "swift_UVM2", "swift_UVW2", "Bessell_U", "swift_U"}
)


def set_default_kernel_settings(filt: str) -> tuple[float, bool, float | None]:
    if filt in _UV_BANDS:
        return 150.0, True, None
    return 10.0, False, None


def _normalize_band_entry(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(raw["scale"]),
        "opt": bool(raw.get("opt", False)),
        "mean": raw.get("mean"),
    }


def load_kernel_settings_file(path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Load JSON kernel overrides.

    Accepts either ``{band: {scale, opt, mean}}`` or legacy
    ``{snname: {band: {...}}}`` layout.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not raw:
        return {}
    sample = next(iter(raw.values()))
    if isinstance(sample, dict) and "scale" in sample:
        return {"__file__": {str(k): _normalize_band_entry(v) for k, v in raw.items()}}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for sn, bands in raw.items():
        out[str(sn)] = {str(b): _normalize_band_entry(v) for b, v in bands.items()}
    return out


def load_legacy_kernel_settings(coco_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    path = os.path.join(
        os.path.normpath(coco_path), "Inputs", "kernels", "lc_gp_kernels_legacy.json"
    )
    if os.path.isfile(path):
        return load_kernel_settings_file(path)
    return {}


def resolve_kernel_for_band(
    snname: str,
    filt: str,
    kernel_settings: dict[str, dict[str, dict[str, Any]]] | None,
) -> tuple[float, bool, float | None]:
    settings = kernel_settings or {}
    for key in (snname, "__file__"):
        if key in settings and filt in settings[key]:
            entry = settings[key][filt]
            mean = entry.get("mean")
            return entry["scale"], entry["opt"], mean
    return set_default_kernel_settings(filt)
