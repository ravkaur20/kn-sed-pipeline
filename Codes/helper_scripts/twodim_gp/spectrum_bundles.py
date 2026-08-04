"""Time-based spectrum bundles and λ-space composites with small-gap interpolation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def cluster_by_time(
    times_days: np.ndarray,
    *,
    max_delta_minutes: float = 5.0,
) -> np.ndarray:
    """Cluster indices with pairwise transitive closure: link if |Δt| ≤ threshold (same time units)."""
    t = np.asarray(times_days, dtype=float).ravel()
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    dt_day = max_delta_minutes / (24.0 * 60.0)
    parent = np.arange(n, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(t[i]) - float(t[j])) <= dt_day:
                union(i, j)

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    uniq, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int32)


def _sort_merge_duplicates(
    wave_aa: np.ndarray,
    flux: np.ndarray,
    weights: Optional[np.ndarray],
    *,
    rel_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(wave_aa, dtype=float).ravel()
    f = np.asarray(flux, dtype=float).ravel()
    ok = np.isfinite(w) & np.isfinite(f) & (w > 0)
    w, f = w[ok], f[ok]
    if w.size == 0:
        return w, f
    if weights is not None:
        wt = np.asarray(weights, dtype=float).ravel()[ok]
    else:
        wt = np.ones_like(f)
    order = np.argsort(w)
    w, f, wt = w[order], f[order], wt[order]
    w_out: list[float] = []
    f_out: list[float] = []
    cw = float(w[0])
    num = float(f[0] * wt[0])
    den = float(wt[0])
    for i in range(1, w.size):
        if abs(w[i] - cw) <= rel_tol * max(abs(cw), abs(w[i]), 1.0):
            num += float(f[i] * wt[i])
            den += float(wt[i])
        else:
            w_out.append(cw)
            f_out.append(num / max(den, 1e-300))
            cw = float(w[i])
            num = float(f[i] * wt[i])
            den = float(wt[i])
    w_out.append(cw)
    f_out.append(num / max(den, 1e-300))
    return np.asarray(w_out, dtype=float), np.asarray(f_out, dtype=float)


def fill_gaps_linear(
    wave_aa: np.ndarray,
    flux: np.ndarray,
    *,
    max_gap_aa: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation across interior gaps strictly smaller than ``max_gap_aa`` (Å)."""
    w = np.asarray(wave_aa, dtype=float).ravel()
    f = np.asarray(flux, dtype=float).ravel()
    if w.size < 2:
        return w, f, np.zeros_like(w, dtype=bool)
    wn: list[float] = []
    fn: list[float] = []
    filln: list[bool] = []
    for i in range(w.size - 1):
        wn.append(float(w[i]))
        fn.append(float(f[i]))
        filln.append(False)
        gap = float(w[i + 1] - w[i])
        if 0 < gap < max_gap_aa:
            nstep = max(2, int(np.ceil(gap / max(1.0, max_gap_aa / 20))))
            xs = np.linspace(w[i], w[i + 1], nstep + 1)[1:-1]
            ys = np.interp(xs, w[i : i + 2], f[i : i + 2])
            for x, y in zip(xs, ys):
                wn.append(float(x))
                fn.append(float(y))
                filln.append(True)
    wn.append(float(w[-1]))
    fn.append(float(f[-1]))
    filln.append(False)
    return (
        np.asarray(wn, dtype=float),
        np.asarray(fn, dtype=float),
        np.asarray(filln, dtype=bool),
    )


def composite_spectra(
    wavelength_list: list[np.ndarray],
    flux_list: list[np.ndarray],
    *,
    max_gap_aa: float = 100.0,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge many spectra onto one sorted λ grid, average duplicate λ, then small-gap fill.

    Returns ``(wave, flux, gap_filled_mask)``.
    """
    if len(wavelength_list) != len(flux_list) or not wavelength_list:
        raise ValueError("wavelength_list and flux_list must be same nonempty length")
    n = len(wavelength_list)
    if weights is None:
        wt = np.ones(n, dtype=float)
    else:
        wt = np.asarray(weights, dtype=float).ravel()
        if wt.size != n:
            raise ValueError("weights must match number of spectra")

    chunks_w: list[np.ndarray] = []
    chunks_f: list[np.ndarray] = []
    chunks_ptwt: list[np.ndarray] = []
    for i in range(n):
        wi, fi = _sort_merge_duplicates(wavelength_list[i], flux_list[i], None)
        if wi.size:
            chunks_w.append(wi)
            chunks_f.append(fi)
            chunks_ptwt.append(np.full_like(wi, float(wt[i])))

    if not chunks_w:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=bool),
        )

    wall = np.concatenate(chunks_w)
    fall = np.concatenate(chunks_f)
    wpoint = np.concatenate(chunks_ptwt)
    w_merged, f_merged = _sort_merge_duplicates(wall, fall, wpoint)
    wf, ff, fm = fill_gaps_linear(w_merged, f_merged, max_gap_aa=max_gap_aa)
    return wf, ff, fm
