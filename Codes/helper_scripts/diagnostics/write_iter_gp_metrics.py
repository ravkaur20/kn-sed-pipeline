#!/usr/bin/env python3
"""Ryan-style iteration metrics plots from existing GP config.json files (no refit)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.gp_diag_common import PHOT, SPEC, compute_sigma_eff, effective_point_class

RYAN_V5_REF = {
    "metric_t": 0.12556053979570903,
    "metric_w": 0.025778672835520284,
    "metric_t2": 7.225195271638228,
    "metric_w2": 5.899400191764839,
    "weight_t_short": 0.02705542196347214,
    "weight_w_short": 0.003102548931327165,
    "sigma_phot": 0.012,
    "sigma_spec": 0.005,
}


def _read_config_metrics(cfg_path: str, iter_label: str) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as fh:
        cj = json.load(fh)
    inner = cj.get("config", {}) if isinstance(cj.get("config"), dict) else cj
    if not isinstance(inner, dict):
        inner = {}
    n_phot = cj.get("n_phot")
    n_spec = cj.get("n_spec")
    if n_phot is None or n_spec is None:
        bundle_hint = os.path.join(
            os.path.dirname(cfg_path), "gp_minimal_export", "gp_minimal_bundle.npz"
        )
        if os.path.isfile(bundle_hint):
            with np.load(bundle_hint) as bd:
                X = bd["X"]
                toc = bd["train_obs_class"] if "train_obs_class" in bd.files else None
                pc = effective_point_class(X, train_obs_class=toc)
                n_phot = int((pc == PHOT).sum())
                n_spec = int((pc == SPEC).sum())
    return {
        "iter": iter_label,
        "iter_index": _iter_index(iter_label),
        "config_path": cfg_path,
        "chi2_per_n_total": cj.get("chi2_per_n_total"),
        "chi2_per_n_phot": cj.get("chi2_per_n_phot"),
        "chi2_per_n_spec": cj.get("chi2_per_n_spec"),
        "log_likelihood": cj.get("log_likelihood") or cj.get("log_likelihood_at_compute"),
        "log_likelihood_at_compute": cj.get("log_likelihood_at_compute"),
        "n_phot": n_phot,
        "n_spec": n_spec,
        "n_train": (int(n_phot) + int(n_spec)) if n_phot is not None and n_spec is not None else None,
        "config": dict(inner),
        "optimize_seconds": cj.get("optimize_seconds"),
        "total_runtime_seconds": cj.get("total_runtime_seconds"),
    }


def _iter_index(label: str) -> int:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else -1


def _collect_iter_records(iter_root: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cfg_path in sorted(glob.glob(os.path.join(iter_root, "iter_*", "gp_runs", "config.json"))):
        iter_dir = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))
        records.append(_read_config_metrics(cfg_path, iter_dir))
    records.sort(key=lambda r: r["iter_index"])
    return records


def _residual_chi2_from_predictions(gp_runs_dir: str) -> dict[str, float | None]:
    pred_path = os.path.join(gp_runs_dir, "predictions.npz")
    bundle_path = os.path.join(gp_runs_dir, "gp_minimal_export", "gp_minimal_bundle.npz")
    if not (os.path.isfile(pred_path) and os.path.isfile(bundle_path)):
        return {"chi2_per_n_total": None, "chi2_per_n_phot": None, "chi2_per_n_spec": None}
    with np.load(pred_path) as pred, np.load(bundle_path) as bd:
        if "mu_train" not in pred.files or "point_class_train" not in pred.files:
            return {"chi2_per_n_total": None, "chi2_per_n_phot": None, "chi2_per_n_spec": None}
        mu = pred["mu_train"].ravel()
        cls = np.asarray(pred["point_class_train"]).ravel()
        y = bd["y"].ravel()
        yerr = bd["yerr"].ravel()
        if mu.size != y.size:
            return {"chi2_per_n_total": None, "chi2_per_n_phot": None, "chi2_per_n_spec": None}
        inner = {}
        cfg_path = os.path.join(gp_runs_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                cj = json.load(fh)
            inner = cj.get("config", {}) if isinstance(cj.get("config"), dict) else cj
        sp = float(inner.get("sigma_phot", 0.012))
        ss = float(inner.get("sigma_spec", 0.005))
        pc = effective_point_class(bd["X"], train_obs_class=bd.get("train_obs_class"))
        sig = compute_sigma_eff(yerr, pc, sigma_phot=sp, sigma_spec=ss)
        chi2 = ((mu - y) / sig) ** 2
        out: dict[str, float | None] = {}
        for name, mask in (
            ("chi2_per_n_total", np.ones_like(chi2, dtype=bool)),
            ("chi2_per_n_phot", cls == PHOT),
            ("chi2_per_n_spec", cls == SPEC),
        ):
            m = mask & np.isfinite(chi2)
            out[name] = float(np.mean(chi2[m])) if m.any() else None
        return out


def write_metrics_plots(records: list[dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not records:
        return
    it = np.array([r["iter_index"] for r in records], dtype=int)
    labels = [r["iter"] for r in records]

    chi_t = np.array([float(r.get("chi2_per_n_total") or np.nan) for r in records])
    chi_s = np.array([float(r.get("chi2_per_n_spec") or np.nan) for r in records])
    ll = np.array(
        [float(r.get("log_likelihood_at_compute") or r.get("log_likelihood") or np.nan) for r in records]
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(it, chi_t, "o-", label=r"$\chi^2/N$ total")
    ax.plot(it, chi_s, "s-", label=r"$\chi^2/N$ spec")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\chi^2/N$")
    ax.set_xticks(it)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chi2_vs_iter.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(it, ll, "o-", color="darkgreen")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log likelihood")
    ax.set_xticks(it)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loglik_vs_iter.png"), dpi=150)
    plt.close(fig)

    keys = [
        ("metric_t", "metric_t"),
        ("metric_w", "metric_w"),
        ("metric_t2", "metric_t2"),
        ("metric_w2", "metric_w2"),
        ("weight_t_short", "weight_t_short"),
        ("weight_w_short", "weight_w_short"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, title) in zip(np.asarray(axes).ravel(), keys):
        ys = [float(r.get("config", {}).get(k, np.nan)) for r in records]
        ax.plot(it, ys, "o-", ms=3, label="iter")
        if k in RYAN_V5_REF:
            ax.axhline(RYAN_V5_REF[k], color="crimson", ls="--", lw=0.9, label="Ryan v5")
        ax.set_title(title)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lengthscales_vs_iter.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sig_p = [float(r.get("config", {}).get("sigma_phot", np.nan)) for r in records]
    sig_s = [float(r.get("config", {}).get("sigma_spec", np.nan)) for r in records]
    ax.plot(it, sig_p, "o-", label="sigma_phot")
    ax.plot(it, sig_s, "s-", label="sigma_spec")
    ax.axhline(RYAN_V5_REF["sigma_phot"], color="C0", ls="--", lw=0.7, alpha=0.7)
    ax.axhline(RYAN_V5_REF["sigma_spec"], color="C1", ls="--", lw=0.7, alpha=0.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel("jitter floor")
    ax.set_xticks(it)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sigma_floors_vs_iter.png"), dpi=150)
    plt.close(fig)

    n_train = [float(r.get("n_train") or np.nan) for r in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, n_train, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("N_train")
    ax.set_xticks(it)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "n_train_vs_iter.png"), dpi=150)
    plt.close(fig)


def run_iter_gp_metrics(iter_root: str, out_dir: str, *, ryan_v5_config: str | None = None) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    records = _collect_iter_records(iter_root)
    for r in records:
        gp_runs = os.path.dirname(r["config_path"])
        chi = _residual_chi2_from_predictions(gp_runs)
        for k, v in chi.items():
            if r.get(k) is None and v is not None:
                r[k] = v
    if ryan_v5_config and os.path.isfile(ryan_v5_config):
        ref = _read_config_metrics(ryan_v5_config, "ryan_v5")
        ref["iter_index"] = -1
        ref["iter"] = "ryan_v5"
        records = [ref] + records
    metrics_json = os.path.join(out_dir, "iter_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    write_metrics_plots(records, out_dir)
    return {"metrics_json": metrics_json, "n_records": len(records)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iter-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ryan-v5-config", default=None)
    args = p.parse_args()
    run_iter_gp_metrics(args.iter_root, args.out_dir, ryan_v5_config=args.ryan_v5_config)
    print("[metrics] wrote under %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
