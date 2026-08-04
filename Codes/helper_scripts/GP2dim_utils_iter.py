"""2D GP iter runner: drop-in prediction matching classic ``run_2DGP_GRID`` grid stacking.

Supports **min–max** ``grid_norm_info`` (``GP2dim_utils``) and **z-score** coords
(``GP2dim_utils``), detected via ``coord_parametrization`` / keys.

``mu_fill`` is post mono+blue; fifth return value ``mu_fill_raw`` is GP-only μ.
"""

from __future__ import annotations

import os
import time

import numpy as np

import pipeline_config as _pconf

from gp2dim_export import maybe_save_gp_minimal_export
from gp2dim_phase_merge import merge_extrap_mjds_dense_log_phase

try:
	from twodim_gp.run_inference import run_gp_from_bundle
except ImportError:  # pragma: no cover
	run_gp_from_bundle = None

import GP2dim_utils as _gp

_augment_t0_anchor_minmax = _gp.augment_2dgp_training_t0_anchor
_apply_training_yerr_floors = _gp._apply_training_yerr_floors  # noqa: SLF001
gp_dense_matrix_bytes_order_of_magnitude = _gp.gp_dense_matrix_bytes_order_of_magnitude
log_prediction_phase_coverage = _gp.log_prediction_phase_coverage


def _grid_norm_uses_zscore(gn: dict) -> bool:
	"""True when NB6 used ``GP2dim_utils.transform2LOG_reshape``."""
	if gn.get("coord_parametrization") == "zscore":
		return True
	return "norm1" not in gn and "x1_mean" in gn and "x2_mean" in gn


def _train_obs_class_for_export(GP2DIM_Class, n_train: int, n_before_augment: int) -> np.ndarray | None:
	"""Optional per-row phot/spec labels from the grid object, aligned to final training size.

	If ``gp_2d_anchor_t0`` appended pseudo points, extend a length-``n_before_augment`` vector with
	``phot`` rows so bundle IDs treat anchors like broad-band constraints (not extra spectroscopic epochs).
	"""
	if n_train <= 0:
		return None
	base = getattr(GP2DIM_Class, "train_obs_class", None)
	if base is None:
		base = getattr(GP2DIM_Class, "gp_train_obs_class", None)
	if base is None:
		return None
	raw = np.asarray(base).ravel()
	if raw.shape[0] == n_train:
		return raw
	if raw.shape[0] == n_before_augment and n_train > n_before_augment:
		n_add = n_train - n_before_augment
		tail = np.full(n_add, "phot", dtype="<U8")
		return np.concatenate([np.asarray(raw, dtype="<U8"), tail])
	return None


def maybe_plot_mu_raw_vs_post(
	GP2DIM_Class,
	x1_fill,
	x2_fill,
	mu_fill_raw,
	mu_fill_post,
	std_fill_post,
):
	"""Optional linear-flux contour comparison (raw vs mono+blue ``mu``)."""
	flag = bool(getattr(_pconf, "GP_PLOT_RAW_AND_PROCESSED", False)) or bool(
		getattr(GP2DIM_Class, "gp_plot_mu_raw_diagnostic", False)
	)
	if not flag:
		return None

	import matplotlib.pyplot as plt

	from GP2dim_utils import scaled_ln_to_linear

	gn = GP2DIM_Class.grid_norm_info
	off = float(gn["offset"])
	scale = float(gn["scale_factor"])

	out_dir = os.path.join(GP2DIM_Class.save_plot_path, "inference", "mu_raw_vs_post")
	os.makedirs(out_dir, exist_ok=True)

	if _grid_norm_uses_zscore(gn):
		x1a = np.asarray(x1_fill, dtype=float)
		x2a = np.asarray(x2_fill, dtype=float)
		logwl = _gp.log10_wavelength_from_x1_norm(x1a, gn)
		wl_lin = np.power(10.0, np.clip(logwl, -50.0, 8.0))
		phase_lin = _gp.phase_days_from_norm_x2(x2a, gn)
	else:
		norm1 = float(gn["norm1"])
		norm2 = float(gn["norm2"])
		offset2 = float(gn["offset2"])
		phase_lin = np.power(10.0, offset2 + norm2 * np.asarray(x2_fill, dtype=float))
		wl_lin = np.power(10.0, norm1 * np.asarray(x1_fill, dtype=float))

	lin_raw = scaled_ln_to_linear(np.asarray(mu_fill_raw, dtype=float), off, scale)
	lin_post = scaled_ln_to_linear(np.asarray(mu_fill_post, dtype=float), off, scale)

	fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
	vmin = float(np.nanpercentile(lin_post, 5.0))
	vmax = float(np.nanpercentile(lin_post, 95.0))
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
		vmin, vmax = float(np.nanmin(lin_post)), float(np.nanmax(lin_post))
	items = [(lin_raw, "mu_raw_linear", vmin, vmax), (lin_post, "mu_post_linear", vmin, vmax)]
	for idx, (arr, ttl, vn, vx) in enumerate(items):
		sc = axes[idx].scatter(phase_lin, wl_lin, c=arr, s=4, cmap="viridis", vmin=vn, vmax=vx)
		axes[idx].set_xscale("log")
		axes[idx].set_xlabel("Phase (days)")
		axes[idx].set_title(ttl)
		plt.colorbar(sc, ax=axes[idx], shrink=0.8)
	del_arr = lin_post - lin_raw
	sc3 = axes[2].scatter(phase_lin, wl_lin, c=del_arr, s=5, cmap="coolwarm")
	axes[2].set_xscale("log")
	axes[2].set_xlabel("Phase (days)")
	axes[2].set_title("delta (post − raw)")
	plt.colorbar(sc3, ax=axes[2], shrink=0.8)
	axes[0].set_ylabel("Wavelength (Å)")
	fig.savefig(os.path.join(out_dir, "gp_mu_raw_vs_post_linflux.pdf"), bbox_inches="tight")
	plt.close(fig)
	return out_dir


def prepare_training_bundle(
	GP2DIM_Class,
	y_data_nonan,
	y_data_nonan_err,
	x1_data_norm,
	x2_data_norm,
	kernel_wls_scale,
	kernel_time_scale,
	extrap_mjds,
	prior=False,
	points=np.nan,
	values=np.nan,
):
	"""Build training ``X,y,yerr`` and prediction ``X_fill`` without running the GP."""
	gn = GP2DIM_Class.grid_norm_info
	use_z = _grid_norm_uses_zscore(gn)

	n_before_augment = int(np.asarray(y_data_nonan).size)

	if bool(getattr(GP2DIM_Class, "gp_2d_anchor_t0", False)):
		_nu_before = int(np.unique(np.asarray(x1_data_norm, dtype=float)).size)
		if use_z:
			x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err = _gp.augment_2dgp_training_t0_anchor(
				x1_data_norm,
				x2_data_norm,
				y_data_nonan,
				y_data_nonan_err,
				GP2DIM_Class.grid_norm_info,
				log_phase_anchor=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log_phase", -8.0)),
				log10_flux_cap=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_cap", -50.0)),
				log10_flux_err=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_err", 2.0)),
			)
		else:
			x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err = _augment_t0_anchor_minmax(
				x1_data_norm,
				x2_data_norm,
				y_data_nonan,
				y_data_nonan_err,
				GP2DIM_Class.grid_norm_info,
				log_phase_anchor=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log_phase", -8.0)),
				log10_flux_cap=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_cap", -50.0)),
				log10_flux_err=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_err", 2.0)),
			)

	X = np.vstack((x1_data_norm, x2_data_norm)).T
	y = np.asarray(y_data_nonan, dtype=np.float64)
	yerr = _apply_training_yerr_floors(GP2DIM_Class, y, y_data_nonan_err, stage="compute")
	n_train = int(y.size)

	wls_min = float(np.min(GP2DIM_Class.grids[0]))
	wls_max = float(np.max(GP2DIM_Class.grids[0]))
	_wl_min_a = getattr(GP2DIM_Class, "pipeline_wl_min_a", None)
	_wl_max_a = getattr(GP2DIM_Class, "pipeline_wl_max_a", None)
	if _wl_min_a is not None:
		wls_min = min(wls_min, float(np.log10(float(_wl_min_a))))
	if _wl_max_a is not None:
		wls_max = max(wls_max, float(np.log10(float(_wl_max_a))))
	span_wl = float(wls_max - wls_min)
	if span_wl <= 0.0:
		raise ValueError("prepare_training_bundle: invalid log10 wavelength span.")

	_gp_n_wl = int(getattr(GP2DIM_Class, "gp_predict_n_wavelength", 300))
	_wl_step = float(getattr(GP2DIM_Class, "gp_predict_wl_step", 0.01))
	n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
	n_wl_use = max(2, min(_gp_n_wl, n_from_step))
	wls_log_grid = np.linspace(wls_min, wls_max, n_wl_use)
	if use_z:
		wls_normed_range = (wls_log_grid - float(gn["x1_mean"])) / float(gn["x1_std"])
	else:
		norm1 = float(gn["norm1"])
		wls_normed_range = wls_log_grid / norm1

	extrap_mjds_arr = np.asarray(extrap_mjds, dtype=float).ravel()
	if extrap_mjds_arr.size == 0:
		raise ValueError("prepare_training_bundle: extrap_mjds empty.")
	if bool(getattr(GP2DIM_Class, "gp_predict_dense_log_phase", False)):
		_dense_n = int(getattr(GP2DIM_Class, "gp_predict_dense_log_phase_n", 64))
		extrap_mjds_arr = merge_extrap_mjds_dense_log_phase(extrap_mjds_arr, _dense_n)

	if use_z:
		mjd_normed_range_full = (extrap_mjds_arr - float(gn["x2_mean"])) / float(gn["x2_std"])
	else:
		norm2 = float(gn["norm2"])
		offset2 = float(gn["offset2"])
		mjd_normed_range_full = (extrap_mjds_arr - offset2) / norm2

	x1_fill = []
	x2_fill = []
	for i in wls_normed_range:
		for k in mjd_normed_range_full:
			x1_fill.append(float(i))
			x2_fill.append(float(k))
	x1_fill = np.asarray(x1_fill, dtype=np.float64)
	x2_fill = np.asarray(x2_fill, dtype=np.float64)
	X_fill = np.column_stack([x1_fill, x2_fill])

	if prior:
		if not (isinstance(points, np.ndarray) and isinstance(values, np.ndarray)):
			raise ValueError("prepare_training_bundle: prior=True requires ndarray points/values.")
		prior_pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
		prior_vals = np.asarray(values, dtype=np.float64).ravel()
	else:
		prior_pts = np.zeros((0, 2), dtype=np.float64)
		prior_vals = np.zeros((0,), dtype=np.float64)

	return {
		"X": X,
		"y": y,
		"yerr": yerr,
		"X_fill": X_fill,
		"x1_fill": x1_fill,
		"x2_fill": x2_fill,
		"prior_points": prior_pts,
		"prior_values": prior_vals,
		"y_var_scale": float(np.var(y)),
		"n_train": n_train,
		"n_before_augment": n_before_augment,
		"wls_log_grid": wls_log_grid,
		"extrap_mjds_arr": extrap_mjds_arr,
		"kernel_wls_scale": float(kernel_wls_scale),
		"kernel_time_scale": float(kernel_time_scale),
	}


def export_minimal_bundle(
	GP2DIM_Class,
	prep: dict,
	*,
	prior: bool,
	points,
	values,
) -> None:
	"""Write ``gp_minimal_export/`` from ``prepare_training_bundle`` output (no GP fit)."""
	y_compute = np.sqrt(np.asarray(prep["yerr"], dtype=float) ** 2 + 1.0e-6 ** 2)
	_export_obs = _train_obs_class_for_export(
		GP2DIM_Class, int(prep["n_train"]), int(prep["n_before_augment"])
	)
	_obs_arg = getattr(GP2DIM_Class, "gp_export_train_obs_class", None)
	if _obs_arg is None:
		_obs_arg = _export_obs
	maybe_save_gp_minimal_export(
		GP2DIM_Class,
		X=prep["X"],
		y=prep["y"],
		yerr=prep["yerr"],
		y_compute=y_compute,
		x1_fill=prep["x1_fill"],
		x2_fill=prep["x2_fill"],
		kernel_wls_scale=prep["kernel_wls_scale"],
		kernel_time_scale=prep["kernel_time_scale"],
		prior=prior,
		points=points,
		values=values,
		grid_norm_info=GP2DIM_Class.grid_norm_info,
		gp_module="GP2dim_utils_iter",
		kernel_layout="gp_matern_additive_opt",
		train_obs_class=_obs_arg,
	)


def run_2DGP_GRID_iter(
	GP2DIM_Class,
	y_data_nonan,
	y_data_nonan_err,
	x1_data_norm,
	x2_data_norm,
	kernel_wls_scale,
	kernel_time_scale,
	extrap_mjds,
	prior=False,
	points=np.nan,
	values=np.nan,
):
	if run_gp_from_bundle is None:
		raise ImportError("twodim_gp missing; vendor under Codes/twodim_gp/.")

	_log_progress = getattr(GP2DIM_Class, "gp_predict_progress", True)

	def _gp_log(msg):
		print(msg, flush=True)

	gn = GP2DIM_Class.grid_norm_info
	use_z = _grid_norm_uses_zscore(gn)

	prep = prepare_training_bundle(
		GP2DIM_Class,
		y_data_nonan,
		y_data_nonan_err,
		x1_data_norm,
		x2_data_norm,
		kernel_wls_scale,
		kernel_time_scale,
		extrap_mjds,
		prior=prior,
		points=points,
		values=values,
	)
	X = prep["X"]
	y = prep["y"]
	yerr = prep["yerr"]
	x1_fill = prep["x1_fill"]
	x2_fill = prep["x2_fill"]
	X_fill = prep["X_fill"]
	n_train = prep["n_train"]
	n_before_augment = prep["n_before_augment"]

	_gp_log(
		"[run_2DGP_GRID_iter] start prior=%r N_train=%i X.shape=%s"
		% (bool(prior), n_train, (X.shape,))
	)
	if getattr(GP2DIM_Class, "verbose", False) or getattr(GP2DIM_Class, "gp_print_training_size", True):
		n2b = gp_dense_matrix_bytes_order_of_magnitude(n_train)
		_gp_log("[run_2DGP_GRID_iter] ~%.2f GB dense hint / Cholesky O(N³)" % (n2b / (1024.0**3),))

	bundle_kw = dict(
		X=X,
		y=y,
		yerr=yerr,
		X_fill=X_fill,
		kernel_wls_scale=np.float64(prep["kernel_wls_scale"]),
		kernel_time_scale=np.float64(prep["kernel_time_scale"]),
		y_var_scale=np.float64(prep["y_var_scale"]),
		prior_points=prep["prior_points"],
		prior_values=prep["prior_values"],
	)

	user_kw = getattr(_pconf, "GP_INFERENCE_KWARGS", {}) or {}
	if not isinstance(user_kw, dict):
		user_kw = {}
	user_kw = dict(user_kw)
	if not prior:
		user_kw["mean"] = "none"
	ws_path = getattr(GP2DIM_Class, "warm_start_config_json", None)
	if ws_path:
		user_kw["warm_start_config_json"] = ws_path
	# Avoid duplicate keyword if ``GP_INFERENCE_KWARGS`` sets ``predict_train`` (see ``pipeline_config``).
	predict_train_flag = bool(user_kw.pop("predict_train", False))

	y_compute = np.sqrt(yerr ** 2 + 1.0e-6 ** 2)

	_cache_rel = getattr(_pconf, "GP_PRIOR_CACHE_SUBDIR", "inference/gp_prior_cache")
	prior_cache = os.path.join(GP2DIM_Class.save_plot_path, _cache_rel.replace("/", os.sep))
	os.makedirs(prior_cache, exist_ok=True)

	run_json = os.path.join(GP2DIM_Class.save_plot_path, "inference", "gp_inference_config.json")
	os.makedirs(os.path.dirname(run_json), exist_ok=True)

	_gp_log("[run_2DGP_GRID_iter] collaborator fit + predict …")
	t0_inf = time.perf_counter()
	# ``early_time_cutoff`` in ``GP_INFERENCE_KWARGS`` applies on ``X_fill[:,1]`` (same units as training).
	# With min-max coords collaborator defaults assumed ~their bundles; with z-score, tune cutoff if needed.
	merged = run_gp_from_bundle(
		bundle_kw,
		cache_workdir=prior_cache,
		predict_train=predict_train_flag,
		**user_kw,
	)

	try:
		sum_meta = dict(merged["config_final"]) if merged.get("config_final") else {}
		sum_meta["log_likelihood"] = merged.get("log_likelihood")
		sum_meta["total_runtime_seconds"] = merged.get("total_runtime_seconds")
		with open(run_json, "w", encoding="utf-8") as fj:
			import json as _json

			fj.write(_json.dumps(sum_meta, indent=2))
	except OSError:
		pass

	mu_fill = np.asarray(merged["mu"], dtype=float).copy()
	mu_fill_raw = np.asarray(merged["mu_raw"], dtype=float).copy()
	std_fill = np.asarray(merged["std"], dtype=float)

	if _log_progress:
		_gp_log("[run_2DGP_GRID_iter] collaborator done in %.1f s" % (time.perf_counter() - t0_inf,))

	_export_obs = _train_obs_class_for_export(GP2DIM_Class, n_train, n_before_augment)
	_obs_arg = getattr(GP2DIM_Class, "gp_export_train_obs_class", None)
	if _obs_arg is None:
		_obs_arg = _export_obs
	maybe_save_gp_minimal_export(
		GP2DIM_Class,
		X=X,
		y=y,
		yerr=yerr,
		y_compute=y_compute,
		x1_fill=x1_fill,
		x2_fill=x2_fill,
		kernel_wls_scale=kernel_wls_scale,
		kernel_time_scale=kernel_time_scale,
		prior=prior,
		points=points,
		values=values,
		grid_norm_info=GP2DIM_Class.grid_norm_info,
		gp_module="GP2dim_utils_iter",
		kernel_layout="gp_matern_additive_opt",
		train_obs_class=_obs_arg,
	)
	# Optional overrides on ``GP2DIM_Class``: ``gp_export_train_obs_class``,
	# ``gp_export_spec_bundle_id``, ``gp_export_assign_spec_bundle_ids`` (see
	# ``gp2dim_export.maybe_save_gp_minimal_export``); optional ``train_obs_class`` /
	# ``gp_train_obs_class`` on the class are threaded above when lengths match (or after
	# t0-anchor extension with ``phot`` tails).

	if use_z:
		logwl_min = float(np.min(_gp.log10_wavelength_from_x1_norm(x1_fill, gn)))
		logwl_max = float(np.max(_gp.log10_wavelength_from_x1_norm(x1_fill, gn)))
		lp_min = float(np.min(float(gn["x2_mean"]) + float(gn["x2_std"]) * x2_fill))
		lp_max = float(np.max(float(gn["x2_mean"]) + float(gn["x2_std"]) * x2_fill))
	else:
		n1 = float(gn["norm1"])
		n2 = float(gn["norm2"])
		o2 = float(gn["offset2"])
		logwl_min = float(np.min(x1_fill * n1))
		logwl_max = float(np.max(x1_fill * n1))
		lp_min = float(np.min(o2 + n2 * x2_fill))
		lp_max = float(np.max(o2 + n2 * x2_fill))
	_gp_log(
		"[run_2DGP_GRID_iter] grid log10(λ): %s .. %s | log10(phase d): %s .. %s | N_pred=%i | coords=%s"
		% (logwl_min, logwl_max, lp_min, lp_max, mu_fill.size, "zscore" if use_z else "minmax")
	)

	maybe_plot_mu_raw_vs_post(GP2DIM_Class, x1_fill, x2_fill, mu_fill_raw, mu_fill, std_fill)

	GP2DIM_Class._twodim_gp_merged = merged
	return x1_fill, x2_fill, mu_fill, std_fill, mu_fill_raw
