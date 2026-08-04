"""Save minimal George 2D-GP training/prediction arrays for collaborators (``.npz`` + small JSON)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping, Optional

import numpy as np


def _jsonable(x: Any) -> Any:
	if isinstance(x, Mapping):
		return {str(k): _jsonable(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [_jsonable(v) for v in x]
	if isinstance(x, np.ndarray):
		return x.tolist()
	if isinstance(x, (np.floating, np.integer)):
		return float(x) if isinstance(x, np.floating) else int(x)
	if isinstance(x, (float, int, str, bool)) or x is None:
		return x
	return str(x)


def resolve_export_dir(GP2DIM_Class) -> str | None:
	"""Return directory to write under, or ``None`` if export is disabled."""
	try:
		import pipeline_config as _pc
	except ImportError:
		_pc = None
	conf_on = bool(getattr(_pc, "GP_EXPORT_MINIMAL", False)) if _pc is not None else False
	class_on = bool(getattr(GP2DIM_Class, "gp_export_minimal", False))
	if not (conf_on or class_on):
		return None
	sub = str(getattr(_pc, "GP_EXPORT_SUBDIR", "gp_minimal_export")) if _pc is not None else "gp_minimal_export"
	override = getattr(GP2DIM_Class, "gp_export_dir", None)
	base = override or os.path.join(GP2DIM_Class.save_plot_path, sub)
	os.makedirs(base, exist_ok=True)
	return base


def _prior_parts(prior: bool, points: Any, values: Any) -> tuple[bool, np.ndarray, np.ndarray]:
	if (
		prior
		and isinstance(points, np.ndarray)
		and isinstance(values, np.ndarray)
		and points.size
		and values.size
	):
		return True, np.asarray(points, dtype=np.float64), np.asarray(values, dtype=np.float64)
	return False, np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)


def _twodim_gp_dir() -> str:
	return os.path.join(os.path.dirname(os.path.abspath(__file__)), "twodim_gp")


def _import_spec_bundle_helpers():
	"""Import from ``Codes/twodim_gp`` (not on sys.path by default)."""
	_ry = _twodim_gp_dir()
	if _ry not in sys.path:
		sys.path.insert(0, _ry)
	from spec_bundle_id_assign import (  # noqa: PLC0415
		assign_spec_bundle_ids_only,
		train_obs_class_strings_from_X,
	)

	return assign_spec_bundle_ids_only, train_obs_class_strings_from_X


def _discover_enrich_npz(bundle_npz_path: str) -> Optional[str]:
	"""Same logic as ``bundle_scale_pipeline.discover_enrich_npz(..., None)`` without importing it."""
	inp = os.path.abspath(os.path.expanduser(str(bundle_npz_path).strip()))
	d = os.path.dirname(inp)
	stem = os.path.splitext(os.path.basename(inp))[0]
	for cand in (
		os.path.join(d, "%s_enrich.npz" % stem),
		os.path.join(d, "enrich.npz"),
	):
		if os.path.isfile(cand):
			return os.path.abspath(cand)
	return None


def _load_enrich_dict_for_bundle(bundle_npz_path: str) -> Optional[dict[str, np.ndarray]]:
	"""Optional ``*_enrich.npz`` beside the bundle (path need not exist on disk yet)."""
	ep = _discover_enrich_npz(bundle_npz_path)
	if ep is None or not os.path.isfile(ep):
		return None
	z = np.load(ep, allow_pickle=False)
	try:
		return {k: np.asarray(z[k]) for k in z.files}
	finally:
		z.close()


def _default_assign_spec_bundle_ids() -> bool:
	try:
		import pipeline_config as _pc  # noqa: PLC0415

		return bool(getattr(_pc, "GP_EXPORT_SPEC_BUNDLE_IDS", True))
	except ImportError:
		return True


def _export_bundle_id_knobs() -> tuple[int, float]:
	try:
		import pipeline_config as _pc  # noqa: PLC0415

		th = int(getattr(_pc, "GP_EXPORT_PHOT_SPEC_THRESHOLD", 50))
		mm = float(getattr(_pc, "GP_EXPORT_MAX_BUNDLE_MINUTES", 5.0))
	except ImportError:
		th, mm = 50, 5.0
	return th, mm


def save_gp_minimal_bundle(
	out_dir: str,
	*,
	X: np.ndarray,
	y: np.ndarray,
	yerr: np.ndarray,
	y_compute: np.ndarray,
	X_fill: np.ndarray,
	kernel_wls_scale: float,
	kernel_time_scale: float,
	y_var_scale: float,
	white_noise_variance: float,
	prior: bool,
	prior_points: np.ndarray,
	prior_values: np.ndarray,
	grid_norm_info: Mapping[str, Any],
	gp_module: str,
	snname: str,
	mode: str,
	kernel_layout: str,
	train_obs_class: Optional[np.ndarray] = None,
	spec_bundle_id: Optional[np.ndarray] = None,
	assign_spec_bundle_ids: Optional[bool] = None,
) -> tuple[str, str]:
	"""Write ``gp_minimal_bundle.npz`` and ``gp_minimal_bundle_meta.json`` under ``out_dir``.

	Optional **collaborator layout** keys (for ``twodim_gp/iterate_gp_surface_bundle_scale``):

	- ``spec_bundle_id``: int32 per row (‑1 = photometry); computed when
	  ``assign_spec_bundle_ids`` is true and ``spec_bundle_id`` not passed.
	- ``train_obs_class``: Unicode phot/spec labels; defaults to ``classify_points`` when IDs are assigned.
	"""
	os.makedirs(out_dir, exist_ok=True)
	npz_path = os.path.join(out_dir, "gp_minimal_bundle.npz")
	json_path = os.path.join(out_dir, "gp_minimal_bundle_meta.json")

	wn = float(white_noise_variance)
	wn_log = float(np.log(wn)) if wn > 0.0 else np.nan

	do_assign = assign_spec_bundle_ids
	if do_assign is None:
		do_assign = _default_assign_spec_bundle_ids()
	pth_for_enrich = npz_path
	th_phot, mm_bundle = _export_bundle_id_knobs()

	train_obs_out: Optional[np.ndarray] = None
	if train_obs_class is not None:
		train_obs_out = np.asarray(train_obs_class)

	sbid: Optional[np.ndarray] = None
	if spec_bundle_id is not None:
		sbid = np.asarray(spec_bundle_id, dtype=np.int32).ravel()
		if sbid.shape[0] != np.asarray(X, dtype=float).shape[0]:
			raise ValueError(
				"spec_bundle_id length %d != N_train %d"
				% (sbid.shape[0], np.asarray(X, dtype=float).shape[0])
			)
	elif do_assign:
		assign_fn, strings_fn = _import_spec_bundle_helpers()
		gn = dict(grid_norm_info)
		enrich = _load_enrich_dict_for_bundle(pth_for_enrich)
		tobs = train_obs_out
		sbid = assign_fn(
			np.asarray(X, dtype=float),
			gn,
			train_obs_class=tobs,
			enrich=enrich,
			phot_spec_threshold=th_phot,
			max_bundle_minutes=mm_bundle,
		)
		if tobs is None:
			train_obs_out = strings_fn(X, phot_spec_threshold=th_phot)

	payload: dict[str, np.ndarray] = {
		"X": np.asarray(X, dtype=np.float64),
		"y": np.asarray(y, dtype=np.float64),
		"yerr": np.asarray(yerr, dtype=np.float64),
		"y_compute": np.asarray(y_compute, dtype=np.float64),
		"X_fill": np.asarray(X_fill, dtype=np.float64),
		"kernel_wls_scale": np.float64(kernel_wls_scale),
		"kernel_time_scale": np.float64(kernel_time_scale),
		"y_var_scale": np.float64(y_var_scale),
		"white_noise_variance": np.float64(wn),
		"white_noise_log": np.float64(wn_log),
		"prior_used": np.int32(1 if prior else 0),
		"prior_points": np.asarray(prior_points, dtype=np.float64),
		"prior_values": np.asarray(prior_values, dtype=np.float64),
	}
	if train_obs_out is not None:
		payload["train_obs_class"] = np.asarray(train_obs_out)
	if sbid is not None:
		payload["spec_bundle_id"] = np.asarray(sbid, dtype=np.int32)

	np.savez_compressed(npz_path, **payload)

	meta = {
		"snname": snname,
		"mode": mode,
		"gp_module": gp_module,
		"kernel_layout": kernel_layout,
		"column_order": "X[:,0]=normalized_log10_wavelength, X[:,1]=normalized_log10_phase_days",
		"prior_used": bool(prior),
		"compute_note": (
			"Use array ``y_compute`` for ``gp.compute``. For ln-flux modules it equals sqrt(yerr**2 + 1e-6**2); "
			"for linear_flux it equals yerr."
		),
		"grid_norm_info": _jsonable(grid_norm_info),
		"files": {"npz": os.path.basename(npz_path), "meta": os.path.basename(json_path)},
	}
	if sbid is not None:
		meta["spec_bundle_id_export"] = (
			"int32 per row: -1 photometry; non-negative spectroscopic bundle (see twodim_gp/spec_bundle_id_assign)"
		)
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print("[gp2dim_export] wrote %s and %s" % (npz_path, json_path), flush=True)
	return npz_path, json_path


def maybe_save_gp_minimal_export(
	GP2DIM_Class,
	*,
	X: np.ndarray,
	y: np.ndarray,
	yerr: np.ndarray,
	y_compute: np.ndarray,
	x1_fill: np.ndarray,
	x2_fill: np.ndarray,
	kernel_wls_scale: float,
	kernel_time_scale: float,
	prior: bool,
	points: Any,
	values: Any,
	grid_norm_info: Mapping[str, Any],
	gp_module: str,
	kernel_layout: str,
	train_obs_class: Optional[np.ndarray] = None,
	spec_bundle_id: Optional[np.ndarray] = None,
	assign_spec_bundle_ids: Optional[bool] = None,
) -> None:
	out_dir = resolve_export_dir(GP2DIM_Class)
	if out_dir is None:
		return
	X_fill = np.vstack((np.asarray(x1_fill, dtype=float), np.asarray(x2_fill, dtype=float))).T
	prior_ok, pp, pv = _prior_parts(prior, points, values)
	wn = float(getattr(GP2DIM_Class, "gp_white_noise", 0.0))
	to_obs = train_obs_class
	if to_obs is None:
		to_obs = getattr(GP2DIM_Class, "gp_export_train_obs_class", None)
	to_sid = spec_bundle_id
	if to_sid is None:
		to_sid = getattr(GP2DIM_Class, "gp_export_spec_bundle_id", None)
	do_assign = assign_spec_bundle_ids
	if do_assign is None:
		do_assign = getattr(GP2DIM_Class, "gp_export_assign_spec_bundle_ids", None)

	save_gp_minimal_bundle(
		out_dir,
		X=X,
		y=y,
		yerr=yerr,
		y_compute=y_compute,
		X_fill=X_fill,
		kernel_wls_scale=kernel_wls_scale,
		kernel_time_scale=kernel_time_scale,
		y_var_scale=float(np.var(y)),
		white_noise_variance=wn,
		prior=prior_ok,
		prior_points=pp,
		prior_values=pv,
		grid_norm_info=grid_norm_info,
		gp_module=gp_module,
		snname=str(getattr(GP2DIM_Class, "snname", "")),
		mode=str(getattr(GP2DIM_Class, "mode", "")),
		kernel_layout=kernel_layout,
		train_obs_class=to_obs,
		spec_bundle_id=to_sid,
		assign_spec_bundle_ids=do_assign,
	)
