"""Load collaborator ``*_meta.json`` next to bundle ``*.npz`` for axes / flux scaling."""

from __future__ import annotations

import json
import os
from typing import Any, Optional


def bundle_meta_json_path(bundle_npz_path: str) -> str:
    """gp_minimal_bundle.npz → gp_minimal_bundle_meta.json convention."""
    base, _ext = os.path.splitext(bundle_npz_path)
    return base + "_meta.json"


def load_bundle_meta(bundle_npz_path: str, meta_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Parse full meta JSON if the file exists; else None."""
    path = meta_path or bundle_meta_json_path(bundle_npz_path)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def identity_grid_norm() -> dict[str, Any]:
    """Fallback when meta is missing — leave plots in normalized target / coordinate space."""
    return {
        "x1_mean": 0.0,
        "x1_std": 1.0,
        "x2_mean": 0.0,
        "x2_std": 1.0,
        "offset": 0.0,
        "scale_factor": 1.0,
        "_normalized_only": True,
    }


def grid_norm_from_bundle_or_meta(
    bundle_npz_path: str,
    *,
    meta_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return ``grid_norm_info`` merged with internal flags for plotting helpers."""
    meta = load_bundle_meta(bundle_npz_path, meta_path=meta_path)
    if meta is None or "grid_norm_info" not in meta:
        return identity_grid_norm()
    gn = dict(meta["grid_norm_info"])
    gn["_normalized_only"] = False
    gn["_bundle_meta_snippet"] = {k: meta[k] for k in ("snname", "mode", "gp_module", "column_order") if k in meta}
    return gn
