"""Outer iterative GP + re-mangle driver (Phase 3).

Bundle-aware remangle reuses ``mangle_spectra_log.compute_mangling_mask_bundle`` when
``pipeline_config.MANGLE_BUNDLE_AWARE`` is True.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

import pipeline_config as pconf
from gp_surface_extract import (
    demangle_extracted_spectrum,
    extract_all_observed_epochs,
    interpolate_mangling_mask_to_wls,
)
from iter_gp_grid import load_predictions_npz, run_iter_gp_fit
from mangle_spectra_log import (
    band_flux_trapz,
    build_mangle_group_map,
    compute_mangling_mask,
    compute_mangling_mask_bundle,
    load_linear_spectrum,
    load_mangled_spectrum,
    save_mangled_spectrum,
    _avail_filters_from_phot4mangling,
    _mangled_output_basename,
    _phot_row_for_mjd,
)
from photometry_filter_utils import load_band_mjd_ranges_json
from spectra_pre_scale import load_scale_groups_json, load_spec_list
from twodim_grid_prep import mangled_filename_to_mjd


def _linear_spec_from_log_path(path: str) -> np.ndarray:
    """Load mangled/demangled log file as linear-flux structured array for masking."""
    import GP2dim_utils as GP2dim

    log_spec, _ = load_mangled_spectrum(path)
    wls = GP2dim.mangled_wls_linear_angstrom(log_spec)
    flux = GP2dim.mangled_flux_linear_from_log10(log_spec["flux"])
    fluxerr = flux * np.log(10.0) * np.asarray(log_spec["fluxerr"], dtype=float)
    return np.array(
        list(zip(wls, flux, fluxerr)),
        dtype=np.dtype([("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")]),
    )


def load_masks_from_mangled_dir(mangled_dir: str) -> dict[str, dict[str, Any]]:
    """Load masks keyed by mangled filename (``<mjd>_mangled_spec.txt``)."""
    import GP2dim_utils as GP2dim

    out: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(mangled_dir):
        return out
    for fn in sorted(os.listdir(mangled_dir)):
        if "mangled_spec" not in fn or not fn.endswith(".txt"):
            continue
        path = os.path.join(mangled_dir, fn)
        log_spec, mask = load_mangled_spectrum(path)
        if mask is None:
            continue
        mjd = mangled_filename_to_mjd(fn)
        mask_wls = GP2dim.mangled_wls_linear_angstrom(log_spec)
        out[fn] = {
            "mjd": float(mjd),
            "mask": np.asarray(mask, dtype=float),
            "mask_wls": np.asarray(mask_wls, dtype=float),
            "path": path,
            "log_spec": log_spec,
        }
    return out


def _mask_info_for_mjd(
    masks_by_file: Mapping[str, dict[str, Any]], mjd: float
) -> Optional[dict[str, Any]]:
    """Exact mangled-file lookup (avoids cross-arm MJD suffix collisions)."""
    target_fn = _mangled_output_basename(mjd)
    if target_fn in masks_by_file:
        return masks_by_file[target_fn]
    mjd_f = float(mjd)
    for info in masks_by_file.values():
        if float(info["mjd"]) == mjd_f:
            return info
    return None


def spec_entries_with_mangled_files(
    spec_entries: Sequence[Any],
    masks_by_file: Mapping[str, dict[str, Any]],
) -> list[Any]:
    """Spec-list rows that have a mangled file (same criterion as GP grid / demangle)."""
    out: list[Any] = []
    for entry in spec_entries:
        if _mask_info_for_mjd(masks_by_file, float(entry.mjd)) is not None:
            out.append(entry)
    return out


def _mask_for_mjd(
    masks_by_file: Mapping[str, dict[str, Any]], mjd: float
) -> Optional[np.ndarray]:
    info = _mask_info_for_mjd(masks_by_file, mjd)
    if info is None:
        return None
    return np.asarray(info["mask"], dtype=float)


def compare_masks(
    old_masks: Mapping[str, np.ndarray], new_masks: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """RMS Δmask over basenames present in both dicts."""
    deltas = []
    for bn, om in old_masks.items():
        if bn not in new_masks:
            continue
        nm = np.asarray(new_masks[bn], dtype=float)
        om = np.asarray(om, dtype=float)
        n = min(om.size, nm.size)
        if n < 1:
            continue
        deltas.append(float(np.sqrt(np.mean((nm[:n] - om[:n]) ** 2))))
    if not deltas:
        return {"delta_mask_rms": float("nan"), "n_compared": 0}
    return {"delta_mask_rms": float(np.max(deltas)), "n_compared": len(deltas)}


def compute_photometry_closure(
    prescaled_paths: Mapping[str, str],
    masks_by_basename: Mapping[str, np.ndarray],
    phot4mangling,
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str,
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
) -> dict[str, Any]:
    """Max/median relative synphot error after applying masks to prescaled spectra."""
    rel_errors = []
    for bn, mask in masks_by_basename.items():
        if bn not in prescaled_paths:
            continue
        pre = load_linear_spectrum(prescaled_paths[bn])
        raw_log = np.log10(np.clip(pre["flux"], 1e-30, None))
        mangled_log = raw_log + np.asarray(mask, dtype=float)
        flux_lin = np.power(10.0, np.clip(mangled_log, -50, 50))
        list_path = pconf.spec_list_path_for_mangling(None, snname)
        entries = {e.basename: e for e in load_spec_list(list_path)}
        if bn not in entries:
            continue
        mjd = float(entries[bn].mjd)
        row = _phot_row_for_mjd(phot4mangling, mjd)
        for filt in avail_filters:
            mjd_key = str(filt)
            if mjd_key not in filter_mjd_dict:
                continue
            band_range = filter_mjd_dict[mjd_key]
            if not (band_range.get("min", -1e9) <= mjd <= band_range.get("max", 1e9)):
                continue
            if photometry_target == "gp_fit":
                target_col = "%s_fit_log_flux" % filt
                if target_col not in row.index:
                    continue
                target_log = float(row[target_col])
                if not np.isfinite(target_log):
                    continue
                target_lin = np.power(10.0, target_log)
            else:
                continue
            try:
                _, _, syn, _, _, _ = band_flux_trapz(
                    pre["wls"],
                    flux_lin,
                    pre["fluxerr"],
                    filt,
                    filter_path=filter_path,
                    snname=snname,
                    csp_sne=csp_sne,
                )
            except Exception:
                continue
            if target_lin > 0 and np.isfinite(syn):
                rel_errors.append(abs(syn - target_lin) / target_lin)
    if not rel_errors:
        return {
            "max_rel_phot_err": float("nan"),
            "median_rel_phot_err": float("nan"),
            "n_phot_points": 0,
        }
    arr = np.asarray(rel_errors, dtype=float)
    return {
        "max_rel_phot_err": float(np.max(arr)),
        "median_rel_phot_err": float(np.median(arr)),
        "n_phot_points": int(arr.size),
    }


def _copy_mangled_tree(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if not fn.endswith(".txt"):
            continue
        shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))


def _save_log_spectrum(
    path: str,
    wls_linear: np.ndarray,
    log10_flux: np.ndarray,
    log10_err: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> None:
    mask_use = np.zeros_like(log10_flux) if mask is None else np.asarray(mask, dtype=float)
    save_mangled_spectrum(path, wls_linear, log10_flux, log10_err, mask_use)


def remangle_spectra(
    snname: str,
    *,
    coco_path: str,
    output_dir: str,
    demangled_specs: Mapping[str, str],
    prescaled_paths: Mapping[str, str],
    mangled_out_dir: str,
    bundle_aware: Optional[bool] = None,
    groups_json: str | None = None,
    photometry_target: str | None = None,
    filter_path: str | None = None,
    csp_sne: Sequence[str] = (),
    use_gp_wavelength_grid: Optional[bool] = None,
) -> dict[str, Any]:
    """Compute new mangling masks from demangled GP spectra; apply to prescaled originals."""
    import pandas as pd

    out_root = output_dir.rstrip(os.sep) + os.sep
    bundle_aware = (
        bool(bundle_aware)
        if bundle_aware is not None
        else bool(getattr(pconf, "MANGLE_BUNDLE_AWARE", False))
    )
    pt = photometry_target or getattr(pconf, "MANGLE_PHOTOMETRY_TARGET", "gp_fit")
    use_gp_wl = (
        bool(use_gp_wavelength_grid)
        if use_gp_wavelength_grid is not None
        else bool(getattr(pconf, "ITER_MANGLE_USE_GP_WAVELENGTH_GRID", True))
    )

    phot_path = os.path.join(out_root, snname, "fitted_phot4mangling_%s.dat" % snname)
    phot4mangling = pd.read_csv(phot_path, sep="\t")
    mjd_json = pconf.band_mjd_ranges_json_path(out_root, snname)
    filter_mjd_dict = load_band_mjd_ranges_json(mjd_json)
    avail_filters = _avail_filters_from_phot4mangling(phot4mangling)
    filt_root = filter_path or os.path.join(
        (coco_path or pconf.COCO_PATH).rstrip(os.sep), "Inputs", "Filters"
    )
    if not filt_root.endswith(os.sep):
        filt_root = filt_root.rstrip(os.sep) + os.sep

    os.makedirs(mangled_out_dir, exist_ok=True)

    groups: list[Any] = []
    group_entries: dict[str, list[Any]] = {}
    basename_to_group: dict[str, str] = {}
    group_merge_order: dict[str, list[str]] = {}
    if bundle_aware:
        gj = pconf.resolve_mangle_groups_json(out_root, snname, groups_json)
        if os.path.isfile(gj):
            _, groups = load_scale_groups_json(gj)
            list_path = pconf.spec_list_path_for_mangling(coco_path, snname)
            entries = load_spec_list(list_path)
            basename_to_group, group_entries = build_mangle_group_map(entries, groups)
            for g in groups:
                group_merge_order[g.id] = list(g.merge_order)

    masks_out: dict[str, np.ndarray] = {}
    diag_groups: list[dict[str, Any]] = []
    report: dict[str, Any] = {"bundle_aware": bundle_aware, "groups": [], "epochs": []}

    def _apply_mask_to_prescaled(
        basename: str,
        mask: np.ndarray,
        mask_wls: np.ndarray,
        spec_mjd: float,
    ) -> None:
        pre = load_linear_spectrum(prescaled_paths[basename])
        m_on_pre = interpolate_mangling_mask_to_wls(mask, mask_wls, pre["wls"])
        raw_log = np.log10(np.clip(pre["flux"], 1e-30, None))
        mangled_log = raw_log + m_on_pre
        mangled_err = pre["fluxerr"] / (pre["flux"] * np.log(10.0))
        save_mangled_spectrum(
            os.path.join(mangled_out_dir, _mangled_output_basename(spec_mjd)),
            pre["wls"],
            mangled_log,
            mangled_err,
            m_on_pre,
        )
        masks_out[basename] = m_on_pre

    if bundle_aware and group_entries:
        for gid, members in group_entries.items():
            demangled = {}
            mjds = {}
            for e in members:
                bn = e.basename
                if bn not in demangled_specs or bn not in prescaled_paths:
                    continue
                demangled[bn] = _linear_spec_from_log_path(demangled_specs[bn])
                mjds[bn] = float(e.mjd)
            if len(demangled) < 1:
                continue
            out = compute_mangling_mask_bundle(
                demangled,
                phot4mangling,
                avail_filters,
                filter_mjd_dict,
                filter_path=filt_root,
                snname=snname,
                csp_sne=csp_sne,
                photometry_target=pt,
                member_mjd=mjds,
                merge_order=group_merge_order.get(gid),
                use_gp_wavelength_grid=use_gp_wl,
            )
            if out is None:
                continue
            masks, _mangled, meta = out
            prescaled_linear: dict[str, dict[str, np.ndarray]] = {}
            mangled_linear: dict[str, dict[str, np.ndarray]] = {}
            for bn, m in masks.items():
                mask_wls = np.asarray(demangled[bn]["wls"], dtype=float)
                _apply_mask_to_prescaled(bn, m, mask_wls, mjds[bn])
                pre = load_linear_spectrum(prescaled_paths[bn])
                raw_log = np.log10(np.clip(pre["flux"], 1e-30, None))
                m_on_pre = masks_out[bn]
                prescaled_linear[bn] = {
                    "wls": np.asarray(pre["wls"], dtype=float),
                    "flux": np.asarray(pre["flux"], dtype=float),
                }
                mangled_linear[bn] = {
                    "wls": np.asarray(pre["wls"], dtype=float),
                    "flux": np.power(10.0, np.clip(raw_log + m_on_pre, -50, 50)),
                }
            report["groups"].append({"id": gid, "meta": meta, "members": list(masks.keys())})
            diag_groups.append(
                {
                    "id": gid,
                    "members": list(masks.keys()),
                    "prescaled": prescaled_linear,
                    "mangled": mangled_linear,
                    "masks": {bn: masks_out[bn] for bn in masks},
                }
            )

    grouped = set(basename_to_group.keys())
    for basename, dem_path in demangled_specs.items():
        if bundle_aware and basename in grouped:
            continue
        if basename not in prescaled_paths:
            continue
        spec = _linear_spec_from_log_path(dem_path)
        list_path = pconf.spec_list_path_for_mangling(coco_path, snname)
        entries = {e.basename: e for e in load_spec_list(list_path)}
        if basename not in entries:
            continue
        mjd = float(entries[basename].mjd)
        row = _phot_row_for_mjd(phot4mangling, mjd)
        out = compute_mangling_mask(
            spec,
            row,
            avail_filters,
            filter_mjd_dict,
            filter_path=filt_root,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=pt,
        )
        if out is None:
            continue
        mask, _, _, meta = out
        mask_wls = np.asarray(spec["wls"], dtype=float)
        _apply_mask_to_prescaled(basename, mask, mask_wls, mjd)
        report["epochs"].append({"basename": basename, "meta": meta})

    return {"masks": masks_out, "report": report, "diag_groups": diag_groups}


def run_iterative_gp_mangle(
    snname: str,
    *,
    coco_path: str,
    output_dir: str,
    t0_fix: float,
    mode: str = "extrapolate_spectra",
    max_iters: Optional[int] = None,
    phot_convergence_frac: Optional[float] = None,
    bundle_aware: Optional[bool] = None,
    seed_from_nb5: Optional[bool] = None,
    nb5_mangled_dir: Optional[str] = None,
    filter_path: Optional[str] = None,
    csp_sne: Sequence[str] = (),
    save_diagnostics: Optional[bool] = None,
    warm_start: Optional[bool] = None,
    photometry_target: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Outer loop: GP surface → extract → demangle → re-mangle until convergence."""
    import pandas as pd

    out_root = output_dir.rstrip(os.sep) + os.sep
    max_iters = int(max_iters if max_iters is not None else pconf.ITER_GP_MANGLE_MAX_ITERS)
    phot_frac = float(
        phot_convergence_frac
        if phot_convergence_frac is not None
        else pconf.PHOT_CONVERGENCE_FRAC
    )
    seed_from_nb5 = (
        bool(seed_from_nb5)
        if seed_from_nb5 is not None
        else bool(getattr(pconf, "ITER_GP_SEED_FROM_NB5", True))
    )
    save_diag = (
        bool(save_diagnostics)
        if save_diagnostics is not None
        else bool(getattr(pconf, "ITER_SAVE_DIAGNOSTICS", True))
    )
    if not csp_sne:
        csp_sne = tuple(getattr(pconf, "CSP_SNE", ()))
    else:
        csp_sne = tuple(csp_sne)
    pt = photometry_target or getattr(pconf, "MANGLE_PHOTOMETRY_TARGET", "gp_fit")
    if pt != "gp_fit" and not bool(getattr(pconf, "ALLOW_RAW_PHOTOMETRY_TARGET", False)):
        raise ValueError(
            "Iterative GP+mangle requires photometry_target='gp_fit' (fitted_phot4mangling); "
            "set ALLOW_RAW_PHOTOMETRY_TARGET=True to override."
        )
    use_warm = (
        bool(warm_start)
        if warm_start is not None
        else bool(getattr(pconf, "ITER_GP_WARM_START", False))
    )
    mu_key = str(getattr(pconf, "GP_PREDICT_MU_KEY", "mu"))
    use_gp_wl = bool(getattr(pconf, "ITER_MANGLE_USE_GP_WAVELENGTH_GRID", True))
    wl_grid_mode = "gp" if use_gp_wl else "prescaled"

    iter_root = pconf.twodim_iter_root(out_root, snname, mode)
    os.makedirs(iter_root, exist_ok=True)
    log_path = os.path.join(iter_root, "iteration_log.jsonl")

    list_path = pconf.spec_list_path_for_mangling(coco_path, snname)
    spec_entries = load_spec_list(list_path)
    prescaled_paths = {e.basename: e.path for e in spec_entries}
    prescaled_wls = {
        e.basename: load_linear_spectrum(e.path)["wls"] for e in spec_entries
    }

    src_mangled = nb5_mangled_dir or os.path.join(out_root, snname, "mangled_spectra")
    iter0 = pconf.twodim_iter_dir(out_root, snname, mode, 0)
    iter0_mangled = os.path.join(iter0, "mangled_spectra")
    if seed_from_nb5:
        if not os.path.isdir(src_mangled):
            raise FileNotFoundError("NB5 mangled dir missing: %s" % src_mangled)
        _copy_mangled_tree(src_mangled, iter0_mangled)
    else:
        os.makedirs(iter0_mangled, exist_ok=True)

    phot_path = os.path.join(out_root, snname, "fitted_phot4mangling_%s.dat" % snname)
    phot4mangling = pd.read_csv(phot_path, sep="\t")
    mjd_json = pconf.band_mjd_ranges_json_path(out_root, snname)
    filter_mjd_dict = load_band_mjd_ranges_json(mjd_json)
    avail_filters = _avail_filters_from_phot4mangling(phot4mangling)
    filt_root = filter_path or os.path.join(
        (coco_path or pconf.COCO_PATH).rstrip(os.sep), "Inputs", "Filters"
    )

    observed_phases = set()
    for e in spec_entries:
        phase = max(float(e.mjd) - float(t0_fix), 1e-5)
        observed_phases.add(float(np.log10(phase)))

    summary: dict[str, Any] = {
        "snname": snname,
        "mode": mode,
        "iter_root": iter_root,
        "max_iters": max_iters,
        "iterations": [],
        "converged": False,
        "stop_reason": None,
    }

    prev_masks_by_bn: dict[str, np.ndarray] = {}
    converged = False
    stop_reason = "max_iters"

    for k in range(max_iters):
        iter_dir = pconf.twodim_iter_dir(out_root, snname, mode, k)
        mangled_dir = os.path.join(iter_dir, "mangled_spectra")
        gp_dir = os.path.join(iter_dir, "gp_runs")
        extracted_dir = os.path.join(iter_dir, "extracted")
        demangled_dir = os.path.join(iter_dir, "demangled")
        figs_dir = pconf.iter_figs_dir(out_root, snname, k, mode=mode)
        for d in (extracted_dir, demangled_dir, figs_dir):
            os.makedirs(d, exist_ok=True)

        if verbose:
            print("[iter_gp_mangle] iteration %i GP fit …" % k, flush=True)

        warm_path = None
        if use_warm and k > 0:
            diag = pconf.iter_inference_dir(out_root, snname, k - 1, mode=mode)
            for _cfg_name in (
                "gp_inference_config.json",
                "twodim_gp_config.json",
                "ryan_gp_config.json",
            ):
                _cand = os.path.join(diag, _cfg_name)
                if os.path.isfile(_cand):
                    warm_path = _cand
                    break

        gp_result = run_iter_gp_fit(
            snname,
            t0_fix=t0_fix,
            mode=mode,
            mangled_spectra_dir=mangled_dir,
            gp_runs_dir=gp_dir,
            output_dir=out_root,
            coco_path=coco_path,
            filters_dir=filt_root,
            csp_sne=csp_sne,
            warm_start_config_path=warm_path,
            verbose=verbose,
        )
        predictions = {
            "mu": gp_result["mu"],
            "mu_raw": gp_result["mu_raw"],
            "std": gp_result["std"],
            "x1_fill": gp_result["x1_fill"],
            "x2_fill": gp_result["x2_fill"],
            "wls_log_grid": gp_result["wls_log_grid"],
            "phase_log10_columns": gp_result["phase_log10_columns"],
            "grid_norm_info": gp_result["grid_norm_info"],
        }

        masks_by_file = load_masks_from_mangled_dir(mangled_dir)
        masks_by_bn: dict[str, np.ndarray] = {}
        mask_wls_by_bn: dict[str, np.ndarray] = {}
        for e in spec_entries:
            info = _mask_info_for_mjd(masks_by_file, float(e.mjd))
            if info is not None:
                masks_by_bn[e.basename] = np.asarray(info["mask"], dtype=float)
                mask_wls_by_bn[e.basename] = np.asarray(info["mask_wls"], dtype=float)

        extract_entries: list[Any] = []
        n_skipped_extract = 0
        for e in spec_entries:
            if _mask_info_for_mjd(masks_by_file, float(e.mjd)) is None:
                n_skipped_extract += 1
                if verbose:
                    print(
                        "[iter_gp_mangle] skip extract (no mangled file): MJD=%.6f %s"
                        % (float(e.mjd), e.basename),
                        flush=True,
                    )
                continue
            extract_entries.append(e)

        extractions = extract_all_observed_epochs(
            predictions,
            spec_entries=extract_entries,
            t0_fix=t0_fix,
            prescaled_wls_by_basename=prescaled_wls,
            observed_phase_log10=observed_phases,
            mu_key=mu_key,
            wavelength_grid=wl_grid_mode,
        )

        demangled_paths: dict[str, str] = {}
        chain_data: list[dict[str, Any]] = []
        for item in extractions:
            bn = item["basename"]
            ex = item["extracted"]
            mask = masks_by_bn.get(bn)
            mask_wls = mask_wls_by_bn.get(bn)
            if mask is None or mask_wls is None:
                continue
            pre = load_linear_spectrum(prescaled_paths[bn])
            wls_prescaled = np.asarray(pre["wls"], dtype=float)
            wls_extract = np.asarray(
                ex.get("target_wls_linear", wls_prescaled), dtype=float
            )
            if wls_extract.size != ex["log10_flux"].size:
                wls_extract = np.power(10.0, ex["log10_wls"])
            if use_gp_wl:
                mask_on_extract = interpolate_mangling_mask_to_wls(
                    mask, mask_wls, wls_extract
                )
            else:
                n = min(mask.size, ex["log10_flux"].size, wls_extract.size)
                mask_on_extract = interpolate_mangling_mask_to_wls(
                    mask[:n], mask_wls[:n], wls_extract[:n]
                )
            dem_log = demangle_extracted_spectrum(ex["log10_flux"], mask_on_extract)
            mjd = item["mjd"]
            ext_path = os.path.join(extracted_dir, "%.6f_gp_extracted.txt" % mjd)
            dem_path = os.path.join(demangled_dir, "%.6f_gp_demangled.txt" % mjd)
            _save_log_spectrum(
                ext_path,
                wls_extract,
                ex["log10_flux"],
                ex["log10_fluxerr"],
                mask_on_extract,
            )
            _save_log_spectrum(
                dem_path,
                wls_extract,
                dem_log,
                ex["log10_fluxerr"],
                np.zeros_like(dem_log),
            )
            demangled_paths[bn] = dem_path
            pre_log = np.log10(np.clip(pre["flux"], 1e-30, None))
            pre_flux = np.asarray(pre["flux"], dtype=float)
            pre_err = np.asarray(pre["fluxerr"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                man_err = pre_err / (pre_flux * np.log(10.0))
            man_on_pre = interpolate_mangling_mask_to_wls(
                mask, mask_wls, wls_prescaled
            )
            chain_data.append(
                {
                    "basename": bn,
                    "mjd": mjd,
                    "prescaled_log": pre_log,
                    "mangled_log": pre_log + man_on_pre,
                    "extracted_log": ex["log10_flux"],
                    "extracted_err": ex["log10_fluxerr"],
                    "mangled_err": man_err,
                    "demangled_log": dem_log,
                    "wls": wls_extract,
                    "wls_prescaled": wls_prescaled,
                    "old_mask": mask_on_extract,
                }
            )

        next_iter = k + 1
        next_mangled = os.path.join(
            pconf.twodim_iter_dir(out_root, snname, mode, next_iter), "mangled_spectra"
        )
        rem = remangle_spectra(
            snname,
            coco_path=coco_path,
            output_dir=out_root,
            demangled_specs=demangled_paths,
            prescaled_paths=prescaled_paths,
            mangled_out_dir=next_mangled,
            bundle_aware=bundle_aware,
            filter_path=filt_root,
            csp_sne=csp_sne,
            photometry_target=pt,
            use_gp_wavelength_grid=use_gp_wl,
        )
        new_masks = rem["masks"]
        phot_metrics = compute_photometry_closure(
            prescaled_paths,
            new_masks,
            phot4mangling,
            avail_filters,
            filter_mjd_dict,
            filter_path=filt_root,
            snname=snname,
            csp_sne=csp_sne,
            photometry_target=pt,
        )
        mask_metrics = compare_masks(prev_masks_by_bn, new_masks) if prev_masks_by_bn else {
            "delta_mask_rms": float("nan"),
            "n_compared": 0,
        }

        iter_metrics = {
            "iteration": k,
            "n_extractions": len(extractions),
            "n_skipped_extract": n_skipped_extract,
            "n_remangled": len(new_masks),
            **phot_metrics,
            **mask_metrics,
        }
        with open(os.path.join(iter_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(iter_metrics, fh, indent=2)

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(iter_metrics) + "\n")

        if save_diag:
            try:
                import iter_gp_mangle_diagnostics as igd

                igd.save_iteration_diagnostics(
                    iter_dir,
                    k,
                    chain_data=chain_data,
                    old_masks=prev_masks_by_bn,
                    new_masks=new_masks,
                    gp_result={
                        **predictions,
                        "mu_raw": gp_result.get("mu_raw"),
                        "X_fill": gp_result.get("X_fill"),
                    },
                    metrics=iter_metrics,
                    coco_path=coco_path,
                    snname=snname,
                    gp_runs_dir=gp_dir,
                    spec_class=gp_result.get("spec_class"),
                    y_data_nonan=gp_result.get("y_data_nonan"),
                    y_data_nonan_err=gp_result.get("y_data_nonan_err"),
                    x1_data_norm=gp_result.get("x1_data_norm"),
                    x2_data_norm=gp_result.get("x2_data_norm"),
                    diag_groups=rem.get("diag_groups") or None,
                    mangle_report=rem.get("report"),
                    t0_fix=t0_fix,
                    mangled_dir=mangled_dir,
                    phot4mangling_path=phot_path,
                    filter_mjd_dict=filter_mjd_dict,
                    avail_filters=avail_filters,
                    filter_path=filt_root,
                    csp_sne=csp_sne,
                    output_dir=out_root,
                )
            except Exception as exc:
                if verbose:
                    print("[iter_gp_mangle] diagnostics failed:", exc, flush=True)

        summary["iterations"].append(iter_metrics)
        prev_masks_by_bn = dict(new_masks)

        if (
            np.isfinite(phot_metrics["max_rel_phot_err"])
            and phot_metrics["max_rel_phot_err"] < phot_frac
        ):
            converged = True
            stop_reason = "photometry_convergence"
            break

    summary["converged"] = converged
    summary["stop_reason"] = stop_reason

    final_dir = pconf.twodim_iter_final_dir(out_root, snname, mode)
    final_gp = os.path.join(final_dir, "full_gp")
    final_mangled = os.path.join(final_dir, "mangled_spectra")
    os.makedirs(final_gp, exist_ok=True)
    os.makedirs(final_mangled, exist_ok=True)

    last_k = len(summary["iterations"]) - 1
    if last_k >= 0:
        last_iter_dir = pconf.twodim_iter_dir(out_root, snname, mode, last_k)
        pred_src = os.path.join(last_iter_dir, "gp_runs", "predictions.npz")
        if os.path.isfile(pred_src):
            shutil.copy2(pred_src, os.path.join(final_gp, "predictions.npz"))
        gp_full_src = os.path.join(last_iter_dir, "gp_runs", "full_gp")
        if os.path.isdir(gp_full_src):
            from gp_full_spectra_export import copy_full_gp_tree

            copy_full_gp_tree(os.path.join(last_iter_dir, "gp_runs"), final_gp)
        next_m = os.path.join(
            pconf.twodim_iter_dir(out_root, snname, mode, last_k + 1), "mangled_spectra"
        )
        if os.path.isdir(next_m):
            _copy_mangled_tree(next_m, final_mangled)

    if bool(getattr(pconf, "ITER_EXPORT_FINAL_SPEC_FOR_QA", True)) and os.path.isdir(final_gp):
        from gp_final_spec_export import export_final_spec_from_full_gp

        qa_dir = pconf.final_spectra_qa_dir(out_root, snname)
        export_final_spec_from_full_gp(final_gp, qa_dir, variant="")
        summary["final_spec_qa_dir"] = qa_dir

    report = {
        "converged": converged,
        "stop_reason": stop_reason,
        "phot_convergence_frac": phot_frac,
        "iterations": summary["iterations"],
    }
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "convergence_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if save_diag:
        try:
            import iter_gp_mangle_diagnostics as igd
            from diagnostics.write_iter_gp_metrics import run_iter_gp_metrics

            igd.write_diagnostics_summary(
                pconf.twodim_iter_diagnostics_summary_dir(out_root, snname, mode),
                summary["iterations"],
                iter_root,
            )
            run_iter_gp_metrics(
                iter_root,
                pconf.twodim_iter_metrics_dir(out_root, snname, mode),
            )
        except Exception:
            pass

    summary["final_dir"] = final_dir
    summary["log_path"] = log_path
    summary["report_path"] = os.path.join(final_dir, "convergence_report.json")
    return summary
