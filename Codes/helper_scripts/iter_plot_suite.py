"""Unified per-iteration QA plots under ``iter_KK/figs/``."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np

from twodim_grid_prep import mangled_filename_to_mjd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False
    plt = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _interp_log_flux(wls_src: np.ndarray, log_src: np.ndarray, wls_tgt: np.ndarray) -> np.ndarray:
    if wls_src.size < 2 or wls_tgt.size < 2:
        n = min(wls_src.size, log_src.size, wls_tgt.size)
        return np.asarray(log_src[:n], dtype=float)
    order = np.argsort(wls_src)
    ws = wls_src[order]
    ls = log_src[order]
    return np.interp(wls_tgt, ws, ls, left=np.nan, right=np.nan)


def stitch_group_log_spectra(
    members: Sequence[str],
    spectra: Mapping[str, Mapping[str, np.ndarray]],
    *,
    merge_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate member log-flux arrays in merge order (or sorted λ)."""
    arms: list[tuple[np.ndarray, np.ndarray]] = []
    order = list(merge_order or [])
    ordered = [m for m in order if m in spectra]
    for m in members:
        bn = os.path.basename(str(m))
        if bn not in spectra and m in spectra:
            bn = m
        if bn not in spectra and m not in ordered:
            ordered.append(bn if bn in spectra else m)
    seen = set()
    for key in ordered:
        if key in seen:
            continue
        rec = spectra.get(key) or spectra.get(os.path.basename(key))
        if rec is None:
            continue
        seen.add(key)
        w = np.asarray(rec["wls"], dtype=float)
        f = np.asarray(rec["log"], dtype=float)
        n = min(w.size, f.size)
        if n >= 2:
            arms.append((w[:n], f[:n]))
    if not arms:
        for key, rec in spectra.items():
            w = np.asarray(rec["wls"], dtype=float)
            f = np.asarray(rec["log"], dtype=float)
            n = min(w.size, f.size)
            if n >= 2:
                arms.append((w[:n], f[:n]))
    if not arms:
        return np.array([]), np.array([])
    w_all = np.concatenate([a[0] for a in arms])
    f_all = np.concatenate([a[1] for a in arms])
    idx = np.argsort(w_all)
    return w_all[idx], f_all[idx]


def _plot_gp_vs_mangled_panel(
    wls_man: np.ndarray,
    man: np.ndarray,
    wls_gp: np.ndarray,
    ext: np.ndarray,
    *,
    title: str,
    out_path: str,
    new_man: np.ndarray | None = None,
) -> None:
    gp_on_man = _interp_log_flux(wls_gp, ext, wls_man)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(wls_man, man, lw=0.9, label="input mangled")
    if new_man is not None and new_man.size == man.size:
        axes[0].plot(wls_man, new_man, lw=0.9, ls="-.", label="new mangled", alpha=0.85)
    axes[0].plot(wls_man, gp_on_man, lw=0.9, ls="--", label="GP (interp to mangled λ)")
    axes[0].set_ylabel("log10(Fλ)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    resid = gp_on_man - man
    axes[1].plot(wls_man, resid, lw=0.8, color="coral")
    axes[1].axhline(0.0, color="0.4", lw=0.6)
    axes[1].set_xlabel("Wavelength (Å)")
    axes[1].set_ylabel("Δ log10(Fλ) (GP − mangled)")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gp_vs_mangled(
    figs_root: str,
    chain_data: Sequence[Mapping[str, Any]],
    *,
    basename_to_group: Mapping[str, str] | None = None,
    group_members: Mapping[str, list[str]] | None = None,
    group_merge_order: Mapping[str, list[str]] | None = None,
    new_mangled_by_bn: Mapping[str, np.ndarray] | None = None,
) -> list[str]:
    """Per-epoch and per-group GP vs input mangled on native mangled grid."""
    if not _HAS_MPL:
        return []
    out_dir = os.path.join(figs_root, "gp_vs_mangled")
    _ensure_dir(out_dir)
    written: list[str] = []
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    grouped_basenames: set[str] = set()
    if basename_to_group:
        grouped_basenames = set(basename_to_group.keys())
        for item in chain_data:
            bn = str(item["basename"])
            gid = basename_to_group.get(bn)
            if gid:
                by_group.setdefault(gid, []).append(item)

    for item in chain_data:
        bn = str(item["basename"])
        if bn in grouped_basenames:
            continue
        wls_man = np.asarray(item.get("wls_prescaled", item["wls"]), dtype=float)
        man = np.asarray(item["mangled_log"], dtype=float)
        wls_gp = np.asarray(item["wls"], dtype=float)
        ext = np.asarray(item["extracted_log"], dtype=float)
        n_man = min(wls_man.size, man.size)
        n_gp = min(wls_gp.size, ext.size)
        if n_man < 2 or n_gp < 2:
            continue
        new_man = None
        if new_mangled_by_bn and bn in new_mangled_by_bn:
            new_arr = np.asarray(new_mangled_by_bn[bn], dtype=float)
            new_man = new_arr[: min(n_man, new_arr.size)]
        mjd = float(item["mjd"])
        path = os.path.join(out_dir, "individual_%s_%.4f.pdf" % (bn.replace(".dat", ""), mjd))
        _plot_gp_vs_mangled_panel(
            wls_man[:n_man],
            man[:n_man],
            wls_gp[:n_gp],
            ext[:n_gp],
            title="Individual %s MJD %.4f" % (bn, mjd),
            out_path=path,
            new_man=new_man,
        )
        written.append(path)

    for gid, items in by_group.items():
        man_spec: dict[str, dict[str, np.ndarray]] = {}
        gp_spec: dict[str, dict[str, np.ndarray]] = {}
        mjd = float(items[0]["mjd"]) if items else 0.0
        for item in items:
            bn = str(item["basename"])
            wls_man = np.asarray(item.get("wls_prescaled", item["wls"]), dtype=float)
            man_spec[bn] = {
                "wls": wls_man,
                "log": np.asarray(item["mangled_log"], dtype=float),
            }
            gp_spec[bn] = {
                "wls": np.asarray(item["wls"], dtype=float),
                "log": np.asarray(item["extracted_log"], dtype=float),
            }
        merge = (group_merge_order or {}).get(gid)
        members = list((group_members or {}).get(gid, [str(i["basename"]) for i in items]))
        w_man, man = stitch_group_log_spectra(members, man_spec, merge_order=merge)
        w_gp, ext = stitch_group_log_spectra(members, gp_spec, merge_order=merge)
        if w_man.size < 2:
            continue
        w_plot = w_man
        gp_on = _interp_log_flux(w_gp, ext, w_plot)
        path = os.path.join(out_dir, "group_%s_%.4f.pdf" % (gid, mjd))
        _plot_gp_vs_mangled_panel(
            w_plot,
            man,
            w_gp,
            ext,
            title="Group %s MJD %.4f" % (gid, mjd),
            out_path=path,
        )
        written.append(path)
    return written


def plot_mangle_delta(
    figs_root: str,
    chain_data: Sequence[Mapping[str, Any]],
    old_masks: Mapping[str, np.ndarray],
    new_masks: Mapping[str, np.ndarray],
) -> list[str]:
    if not _HAS_MPL:
        return []
    out_dir = os.path.join(figs_root, "mangle_delta")
    _ensure_dir(out_dir)
    written: list[str] = []
    for item in chain_data:
        bn = str(item["basename"])
        if bn not in old_masks or bn not in new_masks:
            continue
        wls = np.asarray(item.get("wls_prescaled", item["wls"]), dtype=float)
        pre = np.asarray(item["prescaled_log"], dtype=float)
        om = np.asarray(old_masks[bn], dtype=float)
        nm = np.asarray(new_masks[bn], dtype=float)
        n = min(wls.size, pre.size, om.size, nm.size)
        if n < 2:
            continue
        mjd = float(item["mjd"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(wls[:n], pre[:n] + om[:n], lw=0.9, label="input mangled")
        ax.plot(wls[:n], pre[:n] + nm[:n], lw=0.9, ls="--", label="new mangled")
        ax.plot(wls[:n], nm[:n] - om[:n], ":", lw=0.9, alpha=0.8, label="Δ mask")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("log10(Fλ) / mask")
        ax.legend(fontsize=8)
        ax.set_title("Mangle delta MJD %.4f %s" % (mjd, bn))
        fig.tight_layout()
        path = os.path.join(out_dir, "delta_%.4f_%s.pdf" % (mjd, bn.replace(".dat", "")))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_residuals_vs_unc(
    figs_root: str,
    chain_data: Sequence[Mapping[str, Any]],
) -> list[str]:
    if not _HAS_MPL:
        return []
    out_dir = os.path.join(figs_root, "residuals")
    _ensure_dir(out_dir)
    written: list[str] = []
    for item in chain_data:
        wls_man = np.asarray(item.get("wls_prescaled", item["wls"]), dtype=float)
        man = np.asarray(item["mangled_log"], dtype=float)
        wls_gp = np.asarray(item["wls"], dtype=float)
        ext = np.asarray(item["extracted_log"], dtype=float)
        ext_err = np.asarray(item.get("extracted_err", np.full_like(ext, np.nan)), dtype=float)
        man_err = np.asarray(item.get("mangled_err", np.full_like(man, np.nan)), dtype=float)
        n_man = min(wls_man.size, man.size, man_err.size)
        n_gp = min(wls_gp.size, ext.size, ext_err.size)
        if n_man < 2 or n_gp < 2:
            continue
        w_man = wls_man[:n_man]
        man_use = man[:n_man]
        gp_on = _interp_log_flux(wls_gp[:n_gp], ext[:n_gp], w_man)
        resid = gp_on - man_use
        sig_gp = _interp_log_flux(wls_gp[:n_gp], ext_err[:n_gp], w_man)
        sig = np.sqrt(np.nan_to_num(sig_gp ** 2) + np.nan_to_num(man_err[:n_man] ** 2))
        sig = np.where(sig > 0, sig, np.nan)
        mjd = float(item["mjd"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        axes[0].plot(w_man, resid, lw=0.8, color="coral")
        axes[0].axhline(0.0, color="0.4", lw=0.6)
        axes[0].set_ylabel("GP − mangled")
        if np.any(np.isfinite(sig)):
            axes[1].plot(w_man, np.abs(resid) / sig, lw=0.8)
            axes[1].axhline(1.0, color="0.4", ls="--", lw=0.6)
            axes[1].set_ylabel("|resid| / σ")
        else:
            axes[1].text(0.5, 0.5, "no uncertainty arrays", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xlabel("Wavelength (Å)")
        fig.suptitle("Residuals MJD %.4f" % mjd)
        fig.tight_layout()
        path = os.path.join(out_dir, "residual_%.4f.pdf" % mjd)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_mangled_synphot_lc(
    figs_root: str,
    *,
    snname: str,
    mangled_dir: str,
    phot4mangling_path: str,
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    avail_filters: Sequence[str],
    filter_path: str,
    csp_sne: Sequence[str] = (),
) -> list[str]:
    """Synphot from mangled spectra vs NB4 gp_fit (under figs/phot_lc/)."""
    if not _HAS_MPL or not os.path.isdir(mangled_dir):
        return []
    import pandas as pd

    from mangle_spectra_log import band_flux_trapz, load_mangled_spectrum

    out_dir = os.path.join(figs_root, "phot_lc")
    _ensure_dir(out_dir)
    phot_df = pd.read_csv(phot4mangling_path, sep="\t")
    mjd_col = "MJD" if "MJD" in phot_df.columns else "spec_mjd"
    mangled_by_mjd: dict[float, dict[str, np.ndarray]] = {}
    for fn in os.listdir(mangled_dir):
        if not fn.endswith(".txt"):
            continue
        try:
            mjd = float(mangled_filename_to_mjd(fn))
        except ValueError:
            continue
        path = os.path.join(mangled_dir, fn)
        log_spec, _mask = load_mangled_spectrum(path)
        log_f = np.asarray(log_spec["flux"], dtype=float)
        mangled_by_mjd[mjd] = {
            "wls": np.asarray(log_spec["wls"], dtype=float),
            "flux": np.power(10.0, np.clip(log_f, -50, 50)),
            "fluxerr": np.asarray(log_spec["fluxerr"], dtype=float),
        }
    written: list[str] = []
    for filt in avail_filters:
        col = "%s_fit_log_flux" % filt
        if col not in phot_df.columns:
            continue
        mjd_key = str(filt)
        if mjd_key not in filter_mjd_dict:
            continue
        band_range = filter_mjd_dict[mjd_key]
        mjds_tgt: list[float] = []
        flux_tgt: list[float] = []
        flux_syn: list[float] = []
        for _, row in phot_df.iterrows():
            mjd = float(row[mjd_col])
            if not (band_range.get("min", -1e9) <= mjd <= band_range.get("max", 1e9)):
                continue
            tgt_log = float(row[col])
            if not np.isfinite(tgt_log):
                continue
            if not mangled_by_mjd:
                continue
            closest_mjd = min(mangled_by_mjd.keys(), key=lambda m: abs(m - mjd))
            if abs(closest_mjd - mjd) > 0.5:
                continue
            spec = mangled_by_mjd[closest_mjd]
            try:
                _, _, syn, _, _, _ = band_flux_trapz(
                    spec["wls"],
                    spec["flux"],
                    spec["fluxerr"],
                    filt,
                    filter_path=filter_path,
                    snname=snname,
                    csp_sne=csp_sne,
                )
            except Exception:
                continue
            if not np.isfinite(syn):
                continue
            mjds_tgt.append(mjd)
            flux_tgt.append(float(np.power(10.0, tgt_log)))
            flux_syn.append(float(syn))
        if len(mjds_tgt) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mjds_tgt, flux_tgt, "o", ms=4, label="NB4 gp_fit")
        ax.plot(mjds_tgt, flux_syn, "s", ms=3, alpha=0.85, label="synphot mangled")
        ax.set_yscale("log")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Linear flux")
        ax.set_title("Band %s — mangled synphot vs gp_fit" % filt)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "mangled_synphot_%s.png" % filt)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _new_mangled_logs(
    chain_data: Sequence[Mapping[str, Any]],
    new_masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in chain_data:
        bn = str(item["basename"])
        if bn not in new_masks:
            continue
        pre = np.asarray(item["prescaled_log"], dtype=float)
        nm = np.asarray(new_masks[bn], dtype=float)
        n = min(pre.size, nm.size)
        out[bn] = pre[:n] + nm[:n]
    return out


def run_iter_plot_suite(
    iter_dir: str,
    *,
    chain_data: Sequence[Mapping[str, Any]],
    old_masks: Mapping[str, np.ndarray],
    new_masks: Mapping[str, np.ndarray],
    mangled_dir: str | None = None,
    phot4mangling_path: str | None = None,
    filter_mjd_dict: Mapping[str, Mapping[str, float]] | None = None,
    avail_filters: Sequence[str] | None = None,
    filter_path: str | None = None,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    basename_to_group: Mapping[str, str] | None = None,
    group_members: Mapping[str, list[str]] | None = None,
    group_merge_order: Mapping[str, list[str]] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Write all iter QA figures under ``iter_dir/figs/``."""
    figs_root = os.path.join(iter_dir, "figs")
    _ensure_dir(figs_root)
    new_mangled = _new_mangled_logs(chain_data, new_masks)
    result: dict[str, list[str]] = {
        "gp_vs_mangled": plot_gp_vs_mangled(
            figs_root,
            chain_data,
            basename_to_group=basename_to_group,
            group_members=group_members,
            group_merge_order=group_merge_order,
            new_mangled_by_bn=new_mangled,
        ),
        "mangle_delta": plot_mangle_delta(figs_root, chain_data, old_masks, new_masks),
        "residuals": plot_residuals_vs_unc(figs_root, chain_data),
    }
    if (
        mangled_dir
        and phot4mangling_path
        and filter_mjd_dict is not None
        and filter_path
        and avail_filters
    ):
        result["phot_lc_mangled"] = plot_mangled_synphot_lc(
            figs_root,
            snname=snname,
            mangled_dir=mangled_dir,
            phot4mangling_path=phot4mangling_path,
            filter_mjd_dict=filter_mjd_dict,
            avail_filters=avail_filters,
            filter_path=filter_path,
            csp_sne=csp_sne,
        )
    if metrics is not None:
        import json

        _ensure_dir(figs_root)
        with open(os.path.join(figs_root, "iteration_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(metrics), fh, indent=2)
    return result
