"""Assign ``spec_bundle_id`` per training row without mutating flux (IDs-only path).

Mirrors the labeling used at the start of ``bundle_scale_pipeline`` (epoch detection,
``time_per_epoch``, ``spectrum_bundles.cluster_by_time``) without importing ``gp_utils``
(so **no george** dependency for :mod:`gp2dim_export`).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import spectrum_bundles as sb

__all__ = ["assign_spec_bundle_ids_only", "train_obs_class_strings_from_X"]

# Match ``bundle_scale_pipeline`` / ``GP2dim_utils`` rounding
ROUND_PHASE = 9

PHOT = "phot"
SPEC = "spec"


def _classify_points(
    X: np.ndarray, threshold: int = 50, round_decimals: int = 9
) -> np.ndarray:
    """Same as ``gp_utils.classify_points`` (phot vs spec by λ-count per phase)."""
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be (N, 2); got %s" % (X.shape,))
    phase_round = np.round(X[:, 1], round_decimals)
    uphases, inv = np.unique(phase_round, return_inverse=True)
    n_phases = uphases.size
    counts = np.zeros(n_phases, dtype=int)
    for k in range(n_phases):
        rows = inv == k
        counts[k] = np.unique(np.round(X[rows, 0], round_decimals)).size
    is_phot_phase = counts < threshold
    return np.where(is_phot_phase[inv], PHOT, SPEC)


def _effective_point_class(
    X: np.ndarray,
    *,
    threshold: int,
    train_obs_class: Optional[np.ndarray],
) -> np.ndarray:
    """Same as ``gp_utils.effective_point_class``."""
    if train_obs_class is None:
        return _classify_points(X, threshold=threshold, round_decimals=9)
    n = X.shape[0]
    raw = np.asarray(train_obs_class).ravel()
    if raw.shape[0] != n:
        raise ValueError(
            "train_obs_class length %d != N=%d" % (raw.shape[0], n)
        )
    out = np.empty(n, dtype="<U8")
    for i in range(n):
        v = raw[i]
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("ascii", errors="ignore")
        s = str(v).strip().lower()
        if s in ("phot", "p", "1", "true", "yes"):
            out[i] = PHOT
        elif s in ("spec", "s", "0", "false", "no"):
            out[i] = SPEC
        elif isinstance(raw[i], (np.integer, int)):
            out[i] = PHOT if int(raw[i]) != 0 else SPEC
        else:
            raise ValueError(
                "train_obs_class[%d]=%r not recognized (phot/spec or 0/1)"
                % (i, raw[i])
            )
    return out


def _phase_days_from_norm_x2(x2: np.ndarray, gn: dict[str, Any]) -> np.ndarray:
    """Same as ``plot_results.phase_days_from_norm_x2``."""
    u = np.asarray(x2, dtype=float)
    if gn.get("_normalized_only"):
        return u
    x2m = float(gn["x2_mean"])
    x2s = float(gn["x2_std"])
    log10_phase = x2m + x2s * u
    return np.power(10.0, log10_phase)


def unique_spec_epochs(
    X: np.ndarray,
    spec_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy of ``bundle_scale_pipeline.unique_spec_epochs``."""
    n_train = X.shape[0]
    epoch_of_row = np.full(n_train, -1, dtype=np.int32)
    rows = np.where(spec_mask)[0]
    if rows.size == 0:
        return np.zeros(0, dtype=float), epoch_of_row

    rnd_keys, inv_rows = np.unique(np.round(X[rows, 1], ROUND_PHASE), return_inverse=True)
    n_eps = rnd_keys.size
    canonical_phases = np.zeros(n_eps, dtype=float)
    epoch_row_lists: list[list[int]] = []
    for e in range(n_eps):
        sel = rows[inv_rows == e]
        canonical_phases[e] = float(np.median(X[sel, 1]))
        epoch_row_lists.append(list(sel.astype(int).tolist()))

    for e, rl in enumerate(epoch_row_lists):
        for r in rl:
            epoch_of_row[int(r)] = int(e)

    return canonical_phases, epoch_of_row


def time_per_epoch(
    X: np.ndarray,
    canonical_phases: np.ndarray,
    gn: dict[str, Any],
    enrich: Optional[dict[str, np.ndarray]],
) -> np.ndarray:
    """Copy of ``bundle_scale_pipeline.time_per_epoch``."""
    mj_raw = enrich.get("mjd") if enrich else None
    mj_arr = np.asarray(mj_raw, dtype=float).ravel() if mj_raw is not None else None
    times = np.zeros(canonical_phases.size, dtype=float)
    for e in range(canonical_phases.size):
        ph = canonical_phases[e]
        mrows = (
            np.isfinite(X[:, 0])
            & np.isfinite(X[:, 1])
            & np.isclose(X[:, 1], ph, rtol=0.0, atol=10 ** (-(ROUND_PHASE - 1)))
        )
        idx = np.flatnonzero(mrows)
        if mj_arr is not None and mj_arr.size >= max(1, X.shape[0] - 1) and idx.size:
            mj_sub = mj_arr[np.minimum(idx, mj_arr.size - 1)]
            mj_sub = mj_sub[np.isfinite(mj_sub)]
            if mj_sub.size:
                times[e] = float(np.median(mj_sub))
                continue
        times[e] = float(_phase_days_from_norm_x2(np.array([ph], dtype=float), gn)[0])
    return times


def train_obs_class_strings_from_X(
    X: np.ndarray,
    *,
    phot_spec_threshold: int = 50,
) -> np.ndarray:
    """Length-N Unicode ``phot`` / ``spec`` via ``_classify_points``."""
    cls = _classify_points(
        np.asarray(X, dtype=float),
        threshold=int(phot_spec_threshold),
    )
    return np.asarray(cls, dtype="<U8")


def assign_spec_bundle_ids_only(
    X: np.ndarray,
    grid_norm_info: dict[str, Any],
    *,
    train_obs_class: Optional[np.ndarray] = None,
    enrich: Optional[dict[str, np.ndarray]] = None,
    phot_spec_threshold: int = 50,
    max_bundle_minutes: float = 5.0,
) -> np.ndarray:
    """Return ``spec_bundle_id`` int32 (N,), ``-1`` phot, non-negative spectroscopy bundles."""
    Xf = np.asarray(X, dtype=float)
    n = int(Xf.shape[0])
    out = np.full(n, -1, dtype=np.int32)
    if Xf.ndim != 2 or Xf.shape[1] != 2:
        raise ValueError("assign_spec_bundle_ids_only: X must be (N, 2); got %s" % (Xf.shape,))

    tobs = np.asarray(train_obs_class) if train_obs_class is not None else None
    pc = _effective_point_class(
        Xf,
        threshold=int(phot_spec_threshold),
        train_obs_class=tobs,
    )
    spec_m = pc == SPEC
    canonical_phases, epoch_of_row = unique_spec_epochs(Xf, spec_m)
    n_eps = int(canonical_phases.size)
    if n_eps <= 0:
        return out

    t_epoch = time_per_epoch(Xf, canonical_phases, grid_norm_info, enrich)
    labels_eps = sb.cluster_by_time(
        t_epoch, max_delta_minutes=float(max_bundle_minutes)
    )
    bundle_of_epoch = np.zeros(n_eps, dtype=np.int32)
    for epi in range(n_eps):
        bundle_of_epoch[epi] = int(labels_eps[epi])

    for epi in range(n_eps):
        ridx = np.flatnonzero(epoch_of_row == epi)
        if ridx.size == 0:
            continue
        sbid = int(bundle_of_epoch[epi])
        out[ridx] = sbid
    return out
