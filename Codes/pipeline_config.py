"""Shared KN log-pipeline paths and naming for kn-sed-pipeline.

Edit ``SNNAME_DEFAULT``, ``SN_EXPLOSION_MJD``, ``SN_REDSHIFT``, GP/mangle knobs here.
Set ``COCO_PATH`` env to override repo root. Edit ``filter_plot_config.json`` for band
colors/markers. Notebooks should call ``bootstrap_runtime()`` — do not redefine paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_COCO = os.path.abspath(os.path.join(_DIR, ".."))
HELPER_SCRIPTS_DIR = os.path.join(_DIR, "helper_scripts")

COCO_PATH: str = os.environ.get("COCO_PATH", _DEFAULT_COCO).rstrip("/") + os.sep

SNNAME_DEFAULT: str = "AT2017gfo"

SN_EXPLOSION_MJD: dict[str, float] = {
    "AT2017gfo": 57982.52851852,
}

REFERENCE_BAND_CANDIDATES: tuple[str, ...] = (
    "Swope_V",
    "Sinistro_V",
    "EFOSC2_V",
    "DECam_r",
    "Skymapper_r",
)

SN_REDSHIFT: dict[str, float] = {
    "AT2017gfo": 0.00984,
}

# AT2017gfo uses GeneralFilters only; empty unless you add CSP events.
CSP_SNE: tuple[str, ...] = ()

PRE_BUMP_SNAMES: tuple[str, ...] = ()

PIPELINE_WL_MIN_A: float | None = None
PIPELINE_WL_MAX_A: float | None = None

GP_WHITE_NOISE: float = 0.1**2
USE_TWO_D_GP_ZSCORE_COORDS: bool = True

GP_EXPORT_MINIMAL: bool = True
GP_EXPORT_SUBDIR: str = "gp_minimal_export"
GP_EXPORT_SPEC_BUNDLE_IDS: bool = True
GP_EXPORT_PHOT_SPEC_THRESHOLD: int = 50
GP_EXPORT_MAX_BUNDLE_MINUTES: float = 5.0

GP_LN_FLUX_ERR_FROM_RELATIVE: bool = True
GP_LN_FLUX_OFFSET_FLOOR: bool = True
GP_LN_FLUX_OFFSET_FLOOR_LINEAR: float = 1e-30
GP_YERR_FLOOR_FRAC: float | None = None
GP_YERR_ABS_FLOOR: float = 0.0

ANCHOR_T0_IN_LC_GP: bool = True
APPEND_T0_ROW_TO_LOGSPACE_AFTER_FIT: bool = False

SUBDIR_SPLICED: str = "spliced"
SUBDIR_FULL_GP: str = "full_gp"
SUBDIR_DIAGNOSTICS: str = "diagnostics"

_gp_inference_kw: dict[str, object] = dict(
    kernel_time="matern52",
    kernel_wls="matern52",
    additive_time=True,
    additive_wls=True,
    mean="linear",
    phot_spec_threshold=50,
    lw_short=0.0258,
    lw2=5.90,
    w_short_w=0.003,
    lt_short=0.126,
    lt2=7.23,
    w_short_t=0.027,
    lw=None,
    lt=None,
    log_amp=float(math.log(0.0135)),
    sigma_phot=0.012,
    sigma_spec=0.005,
    enforce_mono_early=True,
    enforce_blue_early=True,
    early_time_cutoff=-4.0,
    mono_floor_fraction=0.5,
    mono_min_slope=0.005,
    mono_smoothing_scale=0.3,
    optimize=False,
    max_iter=60,
    optimize_subsample=2500,
    seed=0,
    predict_chunk=10000,
    predict_train=True,
)

GP_PRIOR_CACHE_SUBDIR: str = "diagnostics/gp_prior_cache"
GP_PLOT_RAW_AND_PROCESSED: bool = False
GP_INFERENCE_KWARGS: dict = dict(_gp_inference_kw)

GP_MODE: str = "extrapolate_spectra"
FINAL_SPECTRA_VARIANT: str = "as_observed"

PHOTOMETRY_STAGES: dict[str, str] = {
    "raw": "1_LCs_flux_raw",
    "dust_corrected": "2_LCs_dust_corrected",
    "extrapolated": "3_LCs_extrapolated",
}

# Legacy folder names (pre-NB3-removal); resolved when new path missing.
PHOTOMETRY_LEGACY_DIRS: dict[str, str] = {
    "extrapolated": "4_LCs_late_extrapolated",
}

FILTER_PLOT_CONFIG_PATH: str = os.path.join(_DIR, "filter_plot_config.json")


def explosion_date_mjd(snname: str) -> float:
    """Explosion / merger MJD for ``snname`` from ``SN_EXPLOSION_MJD``."""
    if snname not in SN_EXPLOSION_MJD:
        raise KeyError("No explosion MJD configured for %r" % snname)
    return float(SN_EXPLOSION_MJD[snname])


def excluded_bands(style: "FilterPlotStyle | None" = None) -> list[str]:
    """Band exclusion list from ``filter_plot_config.json`` (or loaded style)."""
    if style is None:
        style = load_filter_plot_style()
    return list(style.exclude_filt)


def ensure_helper_paths() -> None:
    """Insert ``Codes/``, ``helper_scripts/``, and ``what_the_flux/`` on ``sys.path``."""
    for p in (_DIR, HELPER_SCRIPTS_DIR, what_the_flux_dir()):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def outputs_root(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "Outputs")


def what_the_flux_dir(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "what_the_flux")


def sn_info_dir(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "Inputs", "SNe_Info")


def photometry_dir(coco_path: str | None, stage: str) -> str:
    if stage not in PHOTOMETRY_STAGES:
        raise ValueError(
            "Unknown photometry stage %r (expect one of %s)"
            % (stage, ", ".join(sorted(PHOTOMETRY_STAGES)))
        )
    base = coco_path or COCO_PATH
    return os.path.join(
        os.path.normpath(base), "Inputs", "Photometry", PHOTOMETRY_STAGES[stage]
    )


def resolve_photometry_lc_path(coco_path: str | None, snname: str, stage: str) -> str:
    """Return ``{phot_dir}/{snname}.dat``, with one-release legacy fallback."""
    primary = os.path.join(photometry_dir(coco_path, stage), "%s.dat" % snname)
    if os.path.isfile(primary):
        return primary
    legacy_dir = PHOTOMETRY_LEGACY_DIRS.get(stage)
    if legacy_dir:
        legacy = os.path.join(
            os.path.normpath(coco_path or COCO_PATH),
            "Inputs",
            "Photometry",
            legacy_dir,
            "%s.dat" % snname,
        )
        if os.path.isfile(legacy):
            warnings.warn(
                "Using legacy photometry path %s (migrate to %s)"
                % (legacy, primary),
                stacklevel=2,
            )
            return legacy
    return primary


def twodim_extended_base(output_dir: str, snname: str, gp_mode: str | None = None) -> str:
    """Internal grid builder root: ``Outputs/<SN>/twodim/``."""
    del gp_mode  # production pipeline is extrapolate-only; mode segment removed
    return os.path.join(output_dir.rstrip(os.sep), snname, "twodim")


def twodim_product_dir(
    output_dir: str, snname: str, gp_mode: str | None, product: str
) -> str:
    if product not in (SUBDIR_SPLICED, SUBDIR_FULL_GP):
        raise ValueError("product must be %r or %r" % (SUBDIR_SPLICED, SUBDIR_FULL_GP))
    return os.path.join(twodim_extended_base(output_dir, snname, gp_mode), product)


def twodim_diagnostics_dir(output_dir: str, snname: str, gp_mode: str | None = None) -> str:
    return os.path.join(twodim_extended_base(output_dir, snname, gp_mode), SUBDIR_DIAGNOSTICS)


def twodim_iter_root(output_dir: str, snname: str, mode: str | None = None) -> str:
    del mode
    return os.path.join(output_dir.rstrip(os.sep), snname, TWODIM_ITER_SUBDIR_ROOT)


def twodim_iter_dir(
    output_dir: str, snname: str, mode: str | None, iter_index: int
) -> str:
    del mode
    return os.path.join(twodim_iter_root(output_dir, snname), "iter_%02d" % int(iter_index))


def twodim_iter_final_dir(output_dir: str, snname: str, mode: str | None = None) -> str:
    del mode
    return os.path.join(twodim_iter_root(output_dir, snname), "final")


def twodim_iter_diagnostics_summary_dir(
    output_dir: str, snname: str, mode: str | None = None
) -> str:
    del mode
    return os.path.join(twodim_iter_root(output_dir, snname), "diagnostics_summary")


def twodim_iter_final_spectra_dir(
    output_dir: str,
    snname: str,
    mode: str | None = None,
    product: str = SUBDIR_FULL_GP,
) -> str:
    return os.path.join(twodim_iter_final_dir(output_dir, snname, mode), product)


def sn_figs_dir(output_dir: str, snname: str, sub: str | None = None) -> str:
    """``Outputs/<SN>/figs[/sub]`` — step-5 and shared SN-level figures."""
    base = os.path.join(output_dir.rstrip(os.sep), snname, SN_FIGS_SUBDIR)
    return os.path.join(base, sub) if sub else base


def iter_figs_dir(
    output_dir: str,
    snname: str,
    iter_index: int,
    sub: str | None = None,
    mode: str | None = None,
) -> str:
    """``twodim_iter/iter_KK/figs[/sub]`` — all per-iteration QA plots."""
    del mode
    base = os.path.join(
        twodim_iter_dir(output_dir, snname, None, iter_index),
        "figs",
    )
    return os.path.join(base, sub) if sub else base


def iter_gp_runs_dir(
    output_dir: str,
    snname: str,
    iter_index: int,
    mode: str | None = None,
) -> str:
    del mode
    return os.path.join(twodim_iter_dir(output_dir, snname, None, iter_index), "gp_runs")


def iter_inference_dir(
    output_dir: str,
    snname: str,
    iter_index: int,
    mode: str | None = None,
) -> str:
    """GP inference artifacts (config cache, prior cache) — not QA plots."""
    return os.path.join(iter_gp_runs_dir(output_dir, snname, iter_index, mode), "inference")


def twodim_iter_metrics_dir(
    output_dir: str,
    snname: str,
    mode: str | None = None,
) -> str:
    del mode
    return os.path.join(twodim_iter_root(output_dir, snname), "metrics")


def final_spectra_2dim_base(output_dir: str, snname: str) -> str:
    return os.path.join(output_dir.rstrip(os.sep), snname, "FINAL_spectra_2dim")


def final_spectra_qa_dir(output_dir: str, snname: str, variant: str | None = None) -> str:
    """QA export directory: ``FINAL_spectra_2dim/<variant>/`` (default ``as_observed``)."""
    var = variant if variant is not None else FINAL_SPECTRA_VARIANT
    base = final_spectra_2dim_base(output_dir, snname)
    if not var or var in ("default", ""):
        return base
    return os.path.join(base, var)


def raw_photometry_path(coco_path: str | None, snname: str) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(
        os.path.normpath(base),
        "Inputs",
        "Photometry",
        "1_LCs_flux_raw",
        "%s.dat" % snname,
    )


def band_mjd_ranges_json_path(output_dir: str, snname: str) -> str:
    return os.path.join(
        output_dir.rstrip(os.sep), snname, "%s_band_mjd_ranges.json" % snname
    )


ORIGINAL_SPEC_SUBDIR: str = "1_spec_original"
SMOOTHED_SPEC_SUBDIR: str = "2_spec_smoothed"
SMOOTHED_SPEC_LIST_SUBDIR: str = "2_spec_lists_smoothed"
PRESCALED_SPEC_SUBDIR: str = "2_spec_prescaled"
PRESCALED_SPEC_LIST_SUBDIR: str = "2_spec_lists_prescaled"

SPEC_SCALE_OUTPUT_MODE: str = "scale_only"
SPEC_SCALE_MERGE_GAP_POLICY: str = "linear_bridge"
SPEC_SCALE_GAP_LOG10: float = 0.005
SPEC_SCALE_SAME_TIME_MINUTES: float = 5.0
SPEC_SCALE_OVERLAP_WL_TOL_A: float = 1.0
SPEC_SCALE_SAVE_DIAGNOSTICS: bool = True
SPEC_SCALE_DIAG_SUBDIR: str = "spec_scale_diagnostics"
SPEC_SCALE_CHAIN_MODE: bool = False
SPEC_SCALE_GAP_MAX_A: float = 400.0
SPEC_SCALE_EDGE_N_PIX: int = 10
SPEC_SCALE_SEAM_HALF_WIDTH_A: float = 50.0
SPEC_SCALE_STAR_BRIDGE_FALLBACK: bool = True

USE_PRESCALED_SPECTRA: bool = True
MANGLE_PHOTOMETRY_TARGET: str = "gp_fit"
MANGLE_BUNDLE_AWARE: bool = True
MANGLE_BUNDLE_GROUPS_JSON: str = "auto"
MANGLE_SAVE_DIAGNOSTICS: bool = True
MANGLE_DIAG_SUBDIR: str = "mangle_diagnostics"
MANGLE_KERNEL_DIVIDE: int = 800
MANGLE_GP_KERNEL_MODE: str = "fixed_5"  # "fixed_5" | "kernel_divide_scaled"
MANGLE_GP_KERNEL_FIXED: float = 5.0  # PyCoCo parity; used when mode == "fixed_5"
MANGLE_SEAM_REGRESSION_FACTOR: float = 1.5
MANGLE_BUNDLE_STITCH_SYNPHOT: bool = True
MANGLE_RUN_BOTH_FOR_DIAG: bool = False  # run per-arm + bundle once; diagnostics-only compare

TWODIM_ITER_SUBDIR_ROOT: str = "twodim_iter"
ITER_GP_MANGLE_MAX_ITERS: int = 20
PHOT_CONVERGENCE_FRAC: float = 0.05
MASK_CONVERGENCE_EPS: float = 1e-4
ITER_SAVE_DIAGNOSTICS: bool = True
DIAG_EPOCH_SUBSET = None
DIAG_GP_HEATMAP_INTERVAL: int = 1
DIAG_COPY_GP_FIGS: bool = False
ITER_GP_DIAG_FIGS: bool = True
DIAG_WAVELENGTH_SLICES: bool = True
DIAG_TRAINING_GRID_PLOT: bool = True
ITER_GP_DENSE_PREDICT_GRID: bool = True
ITER_GP_DENSE_PREDICT_GRID_N: int = 100
ITER_MANGLE_USE_GP_WAVELENGTH_GRID: bool = True
GP_PREDICT_MU_KEY: str = "mu"
ITER_GP_GRID_DELTA: float = 30.0
ITER_GP_KERNEL_WLS_SCALE: float = 0.01
ITER_GP_KERNEL_TIME_SCALE: float = 0.04
ITER_GP_PRIOR_FILE: str | None = "/prior_flat.txt"
ITER_GP_SEED_FROM_NB5: bool = True
ITER_GP_WARM_START: bool = True
ITER_GP_EXPORT_FULL_GP: bool = True
ITER_GP_EXPORT_SPLICED: bool = False
ITER_EXPORT_FINAL_SPEC_FOR_QA: bool = True
ALLOW_RAW_PHOTOMETRY_TARGET: bool = False
USE_ITER_GP_MANGLE_FINAL: bool = True
PLOT_GRID_REBIN: bool = True
PLOT_BUNDLE_SCALING_QA: bool = False  # spec_bundle_*_pairs debug figs under figs/debug/bundle_scaling/

# Step 5 epoch plot subdir under Outputs/<SN>/figs/
SN_FIGS_SUBDIR: str = "figs"
MANGLE_EPOCH_FIGS_SUBDIR: str = "mangle_epoch"
MANGLE_NB5_FIGS_SUBDIR: str = "mangle_nb5"


def spectroscopy_root(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "Inputs", "Spectroscopy")


def filters_root(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "Inputs", "Filters")


def general_filters_dir(coco_path: str | None = None) -> str:
    return os.path.join(filters_root(coco_path), "GeneralFilters")


def original_spec_dir(coco_path: str | None, snname: str) -> str:
    return os.path.join(spectroscopy_root(coco_path), ORIGINAL_SPEC_SUBDIR, snname)


def smoothed_spec_dir(coco_path: str | None, snname: str) -> str:
    return os.path.join(spectroscopy_root(coco_path), SMOOTHED_SPEC_SUBDIR, snname)


def prescaled_spec_dir(coco_path: str | None, snname: str) -> str:
    return os.path.join(spectroscopy_root(coco_path), PRESCALED_SPEC_SUBDIR, snname)


def smoothed_spec_list_path(coco_path: str | None, snname: str) -> str:
    return os.path.join(
        spectroscopy_root(coco_path), SMOOTHED_SPEC_LIST_SUBDIR, "%s.list" % snname
    )


def prescaled_spec_list_path(coco_path: str | None, snname: str) -> str:
    return os.path.join(
        spectroscopy_root(coco_path), PRESCALED_SPEC_LIST_SUBDIR, "%s.list" % snname
    )


def spec_list_path_for_mangling(coco_path: str | None, snname: str) -> str:
    if USE_PRESCALED_SPECTRA:
        pp = prescaled_spec_list_path(coco_path, snname)
        if os.path.isfile(pp):
            return pp
    return smoothed_spec_list_path(coco_path, snname)


def spec_scale_groups_json_path(output_dir: str, snname: str) -> str:
    return os.path.join(
        output_dir.rstrip(os.sep), snname, "%s_spec_scale_groups.json" % snname
    )


def spec_scale_report_json_path(output_dir: str, snname: str) -> str:
    return os.path.join(
        output_dir.rstrip(os.sep), snname, "%s_spec_scale_report.json" % snname
    )


def spec_scale_diagnostics_dir(output_dir: str, snname: str) -> str:
    return os.path.join(
        output_dir.rstrip(os.sep), snname, SPEC_SCALE_DIAG_SUBDIR
    )


def mangle_diagnostics_dir(output_dir: str, snname: str) -> str:
    return sn_figs_dir(output_dir, snname, MANGLE_NB5_FIGS_SUBDIR)


def resolve_mangle_groups_json(
    output_dir: str, snname: str, setting: str | None = None
) -> str:
    rel = setting if setting is not None else MANGLE_BUNDLE_GROUPS_JSON
    if rel == "auto":
        return spec_scale_groups_json_path(output_dir, snname)
    return os.path.expanduser(str(rel))


def final_spectra_branch_dir(
    coco_path: str | None,
    snname: str,
    *,
    extension_type: str = "2dim",
    twodim_branch: str | None = None,
) -> str:
    base = coco_path or COCO_PATH
    out = os.path.join(
        os.path.normpath(base), "Outputs", snname, "FINAL_spectra_%s" % extension_type
    )
    if twodim_branch:
        out = os.path.join(out, *str(twodim_branch).replace("\\", "/").split("/"))
    return out


@dataclass
class FilterPlotStyle:
    color_dict: dict[str, str]
    mark_dict: dict[str, str]
    exclude_filt: list[str]
    active_filters: list[str] | None = None
    early_bazin_keep: list[str] | None = None


def load_filter_plot_style(path: str | None = None) -> FilterPlotStyle:
    """Load band colors/markers from ``filter_plot_config.json``."""
    cfg_path = path or FILTER_PLOT_CONFIG_PATH
    if not os.path.isfile(cfg_path):
        warnings.warn(
            "Filter plot config missing (%s); using band_plot_style defaults" % cfg_path,
            stacklevel=2,
        )
        from band_plot_style import (  # noqa: PLC0415
            COLOR_DICT,
            EXCLUDE_FILT,
            MARK_DICT,
            _DEFAULT_COLOR_DICT,
            _DEFAULT_EXCLUDE,
            _DEFAULT_MARK_DICT,
        )

        return FilterPlotStyle(
            color_dict=dict(_DEFAULT_COLOR_DICT if not COLOR_DICT else COLOR_DICT),
            mark_dict=dict(_DEFAULT_MARK_DICT if not MARK_DICT else MARK_DICT),
            exclude_filt=list(_DEFAULT_EXCLUDE if not EXCLUDE_FILT else EXCLUDE_FILT),
            active_filters=None,
            early_bazin_keep=None,
        )

    with open(cfg_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    exclude = list(raw.get("exclude") or [])
    active = raw.get("active_filters")
    early_keep_raw = raw.get("early_bazin_keep")
    early_bazin_keep: list[str] | None
    if early_keep_raw is None:
        early_bazin_keep = None
    else:
        early_bazin_keep = [str(x) for x in early_keep_raw]
    bands_raw = raw.get("bands") or {}
    color_dict: dict[str, str] = {}
    mark_dict: dict[str, str] = {}
    for name, spec in bands_raw.items():
        if not isinstance(spec, dict):
            continue
        if "color" in spec:
            color_dict[str(name)] = str(spec["color"])
        if "marker" in spec:
            mark_dict[str(name)] = str(spec["marker"])

    active_filters: list[str] | None
    if active is None:
        active_filters = None
    else:
        active_filters = [str(x) for x in active]

    return FilterPlotStyle(
        color_dict=color_dict,
        mark_dict=mark_dict,
        exclude_filt=exclude,
        active_filters=active_filters,
        early_bazin_keep=early_bazin_keep,
    )


@dataclass
class NotebookRuntime:
    coco_path: str
    snname: str
    datalc_path: str
    output_dir: str
    output_path: str | None
    dataspec_path: str
    datainfo_path: str
    filter_path: str
    filter_leaf: str
    filters_parent: str
    color_dict: dict[str, str]
    mark_dict: dict[str, str]
    exclude_filt: list[str]
    active_filters: list[str] | None
    early_bazin_keep: list[str] | None
    z: float | None
    final_spectra_dir: str
    gp_mode: str


def bootstrap_runtime(
    *,
    photometry_stage: str = "extrapolated",
    output_stage: str | None = None,
    snname: str | None = None,
) -> NotebookRuntime:
    """Build notebook path bundle and register helper import paths."""
    ensure_helper_paths()
    style = load_filter_plot_style()
    # Refresh band_plot_style module globals for legacy imports
    import band_plot_style as bps  # noqa: PLC0415

    bps.COLOR_DICT.clear()
    bps.COLOR_DICT.update(style.color_dict)
    bps.MARK_DICT.clear()
    bps.MARK_DICT.update(style.mark_dict)
    bps.EXCLUDE_FILT[:] = list(style.exclude_filt)
    bps.color_dict = bps.COLOR_DICT
    bps.mark_dict = bps.MARK_DICT
    bps.exclude_filt = bps.EXCLUDE_FILT

    coco = COCO_PATH
    sn = snname or SNNAME_DEFAULT
    out_root = outputs_root(coco)
    datalc = photometry_dir(coco, photometry_stage)
    out_path = (
        photometry_dir(coco, output_stage) if output_stage is not None else None
    )
    z = SN_REDSHIFT.get(sn)

    return NotebookRuntime(
        coco_path=coco,
        snname=sn,
        datalc_path=datalc,
        output_dir=out_root + os.sep,
        output_path=out_path,
        dataspec_path=spectroscopy_root(coco) + os.sep,
        datainfo_path=sn_info_dir(coco) + os.sep,
        filter_path=filters_root(coco) + os.sep,
        filter_leaf=general_filters_dir(coco) + os.sep,
        filters_parent=filters_root(coco) + os.sep,
        color_dict=dict(style.color_dict),
        mark_dict=dict(style.mark_dict),
        exclude_filt=list(style.exclude_filt),
        active_filters=style.active_filters,
        early_bazin_keep=style.early_bazin_keep,
        z=z,
        final_spectra_dir=final_spectra_qa_dir(out_root, sn),
        gp_mode=GP_MODE,
    )
