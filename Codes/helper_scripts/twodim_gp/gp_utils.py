"""Shared utilities for the GP-SED runner / plotter / comparator.

Public surface:
- ``classify_points``: phot vs spec heuristic with configurable threshold.
- ``build_kernel``: george kernel factory with optional sum-of-two-scales
  (additive) per axis. Default per-axis kernels are normalized so the outer
  amplitude is the sole overall scale.
- ``build_mean``: returns a george-compatible mean model (or ``None``).
- ``compute_diagonal``: per-class jitter folded into ``gp.compute`` diagonal.
- ``KernelConfig``: dataclass holding all kernel hyperparameters used by the
  optimizer (so callers don't keep separate, drift-prone variable lists).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
)

import george
from george import kernels
from george.modeling import ConstantModel, Model


PHOT = "phot"
SPEC = "spec"
KERNEL_NAMES = ("matern32", "matern52", "expsq", "rq")
MEAN_NAMES = ("nearest", "linear", "constant", "none")


def classify_points(X: np.ndarray, threshold: int = 50, round_decimals: int = 9) -> np.ndarray:
    """Classify each row of X as 'phot' or 'spec'.

    A row is 'phot' if its phase column (X[:, 1]) belongs to a group with
    fewer than ``threshold`` unique wavelengths in the training set.
    Phases are rounded to ``round_decimals`` decimals to absorb float noise
    on near-simultaneous epochs.
    """
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be (N, 2); got {X.shape}")
    phase_round = np.round(X[:, 1], round_decimals)
    uphases, inv = np.unique(phase_round, return_inverse=True)
    n_phases = uphases.size
    counts = np.zeros(n_phases, dtype=int)
    for k in range(n_phases):
        rows = inv == k
        counts[k] = np.unique(np.round(X[rows, 0], round_decimals)).size
    is_phot_phase = counts < threshold
    cls = np.where(is_phot_phase[inv], PHOT, SPEC)
    return cls


def effective_point_class(
    X: np.ndarray,
    *,
    threshold: int = 50,
    round_decimals: int = 9,
    train_obs_class: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``classify_points(X, ...)`` unless ``train_obs_class`` length-N overrides row-wise.

    Integer coding in ``train_obs_class``: **0 means spectroscopy**, any non-zero integer
    means photometry (``1`` is typical). Strings ``spec`` / ``phot`` are also accepted.
    """
    if train_obs_class is None:
        return classify_points(X, threshold=threshold, round_decimals=round_decimals)
    n = X.shape[0]
    raw = np.asarray(train_obs_class).ravel()
    if raw.shape[0] != n:
        raise ValueError(f"train_obs_class length {raw.shape[0]} != N={n}")
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
            raise ValueError(f"train_obs_class[{i}]={raw[i]!r} not recognized (use phot/spec or 0/1)")
    return out


def compute_diagonal(
    yerr: np.ndarray,
    point_class: np.ndarray,
    sigma_phot: float,
    sigma_spec: float,
    floor: float = 1e-6,
) -> np.ndarray:
    """Return the per-point std passed to ``gp.compute``.

    diag_i = sqrt(yerr_i**2 + sigma_class(i)**2 + floor**2).
    The ``floor`` mirrors the 1e-6 the collaborator used for numerical
    safety on the smallest yerr values.
    """
    sigma_class = np.where(point_class == PHOT, sigma_phot, sigma_spec)
    return np.sqrt(np.asarray(yerr, dtype=float) ** 2 + sigma_class**2 + floor**2)


def _normalized_axis_kernel(name: str, metric: float, axis: int) -> kernels.Kernel:
    """Return a george axis-restricted kernel that returns 1.0 at zero distance.

    All george stationary kernels are unit-amplitude by default, so we just
    pick the right family.
    """
    if name == "matern32":
        return kernels.Matern32Kernel(metric=metric, ndim=2, axes=axis)
    if name == "matern52":
        return kernels.Matern52Kernel(metric=metric, ndim=2, axes=axis)
    if name == "expsq":
        return kernels.ExpSquaredKernel(metric=metric, ndim=2, axes=axis)
    if name == "rq":
        # Rational Quadratic with mid-range alpha; alpha is also a kernel param,
        # but for now we keep it fixed in the warm-start. The optimizer
        # expression is sufficient with metric only.
        return kernels.RationalQuadraticKernel(
            log_alpha=0.0, metric=metric, ndim=2, axes=axis
        )
    raise ValueError(f"unknown kernel {name!r}; supported: {KERNEL_NAMES}")


def _additive_axis_kernel(
    name: str,
    metric_short: float,
    metric_long: float,
    weight_short: float,
    axis: int,
) -> kernels.Kernel:
    """Return ``w * k_short + (1 - w) * k_long``.

    Each base kernel returns 1 at zero distance, so the sum returns 1 at
    zero distance and acts as a normalized axis kernel with a tunable
    short-vs-long mixture.
    """
    weight_short = float(np.clip(weight_short, 1e-6, 1.0 - 1e-6))
    k_short = _normalized_axis_kernel(name, metric_short, axis)
    k_long = _normalized_axis_kernel(name, metric_long, axis)
    return weight_short * k_short + (1.0 - weight_short) * k_long


@dataclass
class KernelConfig:
    """Configuration + optimizable hyperparameters for the GP kernel.

    Each of (wls, time) axes is either a single normalized kernel
    (one ``log_metric``) or a sum of two same-family kernels
    (``log_metric_short`` + ``log_metric_long`` + ``logit_weight_short``).
    The outer amplitude ``log_amp`` multiplies the final product.

    Per-class jitters live here because they're optimized jointly.
    """

    name_t: str = "matern52"
    name_w: str = "matern52"
    additive_t: bool = False
    additive_w: bool = False

    log_amp: float = 0.0
    log_metric_t: float = np.log(0.04)
    log_metric_w: float = np.log(0.01)
    log_metric_t2: float = np.log(0.04 * 16)
    log_metric_w2: float = np.log(0.01 * 16)
    logit_weight_t: float = 0.0  # short vs long logit; 0 -> 50/50
    logit_weight_w: float = 0.0

    log_sigma_phot: float = np.log(0.02)
    log_sigma_spec: float = np.log(0.02)

    def free_param_names(self) -> list[str]:
        names = ["log_amp", "log_sigma_phot", "log_sigma_spec",
                 "log_metric_t", "log_metric_w"]
        if self.additive_t:
            names += ["log_metric_t2", "logit_weight_t"]
        if self.additive_w:
            names += ["log_metric_w2", "logit_weight_w"]
        return names

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.free_param_names()], dtype=float)

    def update_from_vector(self, theta: np.ndarray) -> "KernelConfig":
        names = self.free_param_names()
        if len(theta) != len(names):
            raise ValueError(f"theta length {len(theta)} != param count {len(names)}")
        for n, v in zip(names, theta):
            setattr(self, n, float(v))
        return self

    def default_bounds(
        self,
        *,
        sigma_spec_min: Optional[float] = None,
    ) -> list[tuple[float, float]]:
        # When the axis is additive we keep the short and long scales
        # disjoint (short wavelength metric in [exp(-6), exp(3.5)] by default,
        # long metric in [exp(1), exp(8)]). This avoids the
        # short and long scales collapsing into degeneracy and ensures the
        # long scale really *is* long (e.g. kilonova color spans the full
        # spectrum, intra-cluster smoothing is short).
        # When the axis is single-scale we keep the wider [-8, 6] range.
        spec_floor = 0.005 if sigma_spec_min is None else max(float(sigma_spec_min), 1e-6)
        bounds: dict[str, tuple[float, float]] = {
            "log_amp": (np.log(1e-6), np.log(10.0)),
            # Phot floor at log(0.012); a few-percent floor matches typical
            # photometric calibration systematics and prevents the optimizer
            # from over-fitting through cluster scatter.
            "log_sigma_phot": (np.log(0.012), 0.0),
            # Spec floor: legacy default 0.005; callers may raise via sigma_spec_min.
            "log_sigma_spec": (np.log(spec_floor), 0.0),
            "log_metric_t": (-6.0, 1.0) if self.additive_t else (-8.0, 6.0),
            # Wavelength short scale: upper bound allows much shorter correlation in x₁
            # (less smoothing across the spectrum) when the data demand it.
            "log_metric_w": (-6.0, 3.5) if self.additive_w else (-8.0, 6.0),
            "log_metric_t2": (1.0, 8.0),
            "log_metric_w2": (1.0, 8.0),
            "logit_weight_t": (-6.0, 6.0),
            "logit_weight_w": (-6.0, 6.0),
        }
        return [bounds[n] for n in self.free_param_names()]

    def apply_saved_inner_config(self, inner: dict) -> list[str]:
        """Overwrite optimizable hyperparameters from a prior ``run_gp`` ``config.json`` ``"config"`` block.

        Only keys listed in :meth:`free_param_names` are applied. Caller should verify
        ``additive_t`` / ``additive_w`` match the current optimization layout.
        """
        applied: list[str] = []
        for n in self.free_param_names():
            if n not in inner:
                continue
            v = float(inner[n])
            if np.isfinite(v):
                setattr(self, n, v)
                applied.append(n)
        return applied

    def as_dict(self) -> dict:
        d = {n: float(getattr(self, n)) for n in self.free_param_names()}
        d["amp"] = float(np.exp(self.log_amp))
        d["sigma_phot"] = float(np.exp(self.log_sigma_phot))
        d["sigma_spec"] = float(np.exp(self.log_sigma_spec))
        d["metric_t"] = float(np.exp(self.log_metric_t))
        d["metric_w"] = float(np.exp(self.log_metric_w))
        if self.additive_t:
            d["metric_t2"] = float(np.exp(self.log_metric_t2))
            d["weight_t_short"] = float(_sigmoid(self.logit_weight_t))
        if self.additive_w:
            d["metric_w2"] = float(np.exp(self.log_metric_w2))
            d["weight_w_short"] = float(_sigmoid(self.logit_weight_w))
        d["name_t"] = self.name_t
        d["name_w"] = self.name_w
        d["additive_t"] = self.additive_t
        d["additive_w"] = self.additive_w
        return d


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def build_kernel(cfg: KernelConfig) -> kernels.Kernel:
    """Construct a george kernel from a ``KernelConfig``.

    The full kernel is ``amp * k_wls(axes=1) * k_time(axes=0)``, where each
    axis kernel is either a normalized base kernel or a sum-of-two-scales
    of the same family.
    """
    metric_t = float(np.exp(cfg.log_metric_t))
    metric_w = float(np.exp(cfg.log_metric_w))
    if cfg.additive_w:
        metric_w2 = float(np.exp(cfg.log_metric_w2))
        w_short = _sigmoid(cfg.logit_weight_w)
        k_w = _additive_axis_kernel(cfg.name_w, metric_w, metric_w2, w_short, axis=1)
    else:
        k_w = _normalized_axis_kernel(cfg.name_w, metric_w, axis=1)
    if cfg.additive_t:
        metric_t2 = float(np.exp(cfg.log_metric_t2))
        t_short = _sigmoid(cfg.logit_weight_t)
        k_t = _additive_axis_kernel(cfg.name_t, metric_t, metric_t2, t_short, axis=0)
    else:
        k_t = _normalized_axis_kernel(cfg.name_t, metric_t, axis=0)

    amp = float(np.exp(cfg.log_amp))
    return amp * (k_w * k_t)


def _prior_cache_path(workdir: str) -> str:
    return os.path.join(workdir, "prior_linear_interp.pkl")


def _build_linear_with_nearest_fallback(
    prior_pts: np.ndarray,
    prior_val: np.ndarray,
    cache_path: Optional[str] = None,
) -> tuple[LinearNDInterpolator, NearestNDInterpolator]:
    """Build (and optionally cache) the linear + nearest interpolator pair."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            print(f"[gp_utils] loaded cached prior interpolators from {cache_path}")
            return cached
        except Exception as exc:
            print(f"[gp_utils] cache load failed ({exc!r}); rebuilding")
    print(
        f"[gp_utils] building LinearNDInterpolator over {prior_pts.shape[0]} "
        "points (this can take ~10-30 s)"
    )
    lin = LinearNDInterpolator(prior_pts, prior_val)
    near = NearestNDInterpolator(prior_pts, prior_val)
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((lin, near), f)
            print(f"[gp_utils] cached prior interpolators to {cache_path}")
        except Exception as exc:
            print(f"[gp_utils] cache save failed ({exc!r})")
    return lin, near


class _NearestPriorMeanModel(Model):
    parameter_names = ()

    def __init__(self, prior_pts: np.ndarray, prior_val: np.ndarray):
        super().__init__()
        self._near = NearestNDInterpolator(prior_pts, prior_val)

    def get_value(self, t):
        z = self._near(t[:, 0], t[:, 1])
        return np.where(np.isnan(z), 0.0, z)


class _LinearPriorMeanModel(Model):
    parameter_names = ()

    def __init__(
        self,
        lin: LinearNDInterpolator,
        near: NearestNDInterpolator,
    ):
        super().__init__()
        self._lin = lin
        self._near = near

    def get_value(self, t):
        z = self._lin(t[:, 0], t[:, 1])
        # outside the convex hull -> NaN; fall back to nearest.
        nan_mask = np.isnan(z)
        if nan_mask.any():
            z = z.copy()
            z[nan_mask] = self._near(t[nan_mask, 0], t[nan_mask, 1])
        return z


def build_mean(
    name: str,
    prior_pts: Optional[np.ndarray] = None,
    prior_val: Optional[np.ndarray] = None,
    cache_workdir: Optional[str] = None,
) -> Optional[Model]:
    """Return a george mean model for the requested ``name``.

    Returns ``None`` when ``name == 'none'``, which george treats as zero mean.
    """
    if name not in MEAN_NAMES:
        raise ValueError(f"unknown mean {name!r}; supported: {MEAN_NAMES}")
    if name == "none":
        return None
    if name == "constant":
        if prior_val is None or prior_val.size == 0:
            return ConstantModel(0.0)
        return ConstantModel(float(np.mean(prior_val)))
    if prior_pts is None or prior_val is None or prior_pts.size == 0 or prior_val.size == 0:
        raise ValueError(f"mean='{name}' requires prior_points + prior_values")
    if name == "nearest":
        return _NearestPriorMeanModel(prior_pts, prior_val)
    if name == "linear":
        cache_path = _prior_cache_path(cache_workdir) if cache_workdir else None
        lin, near = _build_linear_with_nearest_fallback(prior_pts, prior_val, cache_path)
        return _LinearPriorMeanModel(lin, near)
    raise ValueError(f"unhandled mean {name!r}")


def make_gp(cfg: KernelConfig, mean_model: Optional[Model]) -> george.GP:
    """Construct (but do not compute) a george.GP from a config + mean."""
    kernel = build_kernel(cfg)
    if mean_model is None:
        return george.GP(kernel)
    return george.GP(kernel, mean=mean_model)


__all__ = [
    "PHOT",
    "SPEC",
    "KERNEL_NAMES",
    "MEAN_NAMES",
    "KernelConfig",
    "classify_points",
    "effective_point_class",
    "compute_diagonal",
    "build_kernel",
    "build_mean",
    "make_gp",
]
