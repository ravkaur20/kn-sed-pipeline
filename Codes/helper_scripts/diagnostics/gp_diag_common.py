"""GP diagnostics shared helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np

PHOT = "phot"
SPEC = "spec"


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def classify_points(X: np.ndarray, threshold: int = 50) -> np.ndarray:
    phase_round = np.round(X[:, 1], 9)
    uphases, inv = np.unique(phase_round, return_inverse=True)
    counts = np.zeros(uphases.size, dtype=int)
    for k in range(uphases.size):
        rows = inv == k
        counts[k] = np.unique(np.round(X[rows, 0], 9)).size
    is_phot = counts < threshold
    return np.where(is_phot[inv], PHOT, SPEC)


def effective_point_class(
    X: np.ndarray,
    *,
    threshold: int = 50,
    train_obs_class: np.ndarray | None = None,
) -> np.ndarray:
    if train_obs_class is None:
        return classify_points(X, threshold=threshold)
    raw = np.asarray(train_obs_class).ravel()
    if raw.shape[0] != X.shape[0]:
        return classify_points(X, threshold=threshold)
    out = np.empty(raw.shape[0], dtype="<U8")
    for i, v in enumerate(raw):
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode()
        s = str(v).lower()
        out[i] = PHOT if s.startswith("phot") else SPEC
    return out


def compute_sigma_eff(
    yerr: np.ndarray,
    point_class: np.ndarray,
    *,
    sigma_phot: float = 0.012,
    sigma_spec: float = 0.005,
) -> np.ndarray:
    yerr = np.asarray(yerr, dtype=float).ravel()
    pc = np.asarray(point_class).ravel()
    out = np.maximum(yerr, 1e-30)
    for i in range(out.size):
        floor = sigma_phot if pc[i] == PHOT else sigma_spec
        out[i] = float(np.sqrt(out[i] ** 2 + floor ** 2))
    return out
