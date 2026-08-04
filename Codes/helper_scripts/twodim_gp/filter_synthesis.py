"""Synthetic photometry via pysynphot + TRDS; missing filters are reported and skipped."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FilterRoundReport:
    trds_roots_tried: list[str]
    pysyn_cdbS_effective: Optional[str]
    pysynphot_import_error: Optional[str]
    bands_used: list[dict[str, Any]] = field(default_factory=list)
    bands_skipped_missing: list[dict[str, Any]] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _pick_trds_root(trds_roots: list[str]) -> Optional[str]:
    for root in trds_roots:
        rp = os.path.expanduser(os.path.expandvars(root))
        if os.path.isdir(rp) and os.path.isdir(os.path.join(rp, "grp")):
            return rp
    return None


def configure_pysynphot_environment(trds_roots: list[str]) -> tuple[Optional[str], list[str]]:
    """Set ``PYSYN_CDBS`` to the first usable TRDS root. Returns (effective_root, tried_paths)."""
    tried = [os.path.expanduser(os.path.expandvars(r)) for r in trds_roots]
    eff = _pick_trds_root(trds_roots)
    if eff:
        os.environ["PYSYN_CDBS"] = eff
    return eff, tried


def import_pysynphot() -> tuple[Optional[Any], Optional[str]]:
    try:
        import pysynphot as S  # type: ignore[import-untyped]

        return S, None
    except ImportError as e:
        return None, str(e)


def _try_bandpass(S: Any, name: str) -> tuple[Any, Optional[str]]:
    try:
        bp = S.ObsBandpass(name)
        if bp is not None:
            _ = float(np.nanmax(bp.throughput))
        return bp, None
    except Exception as e:  # noqa: BLE001 — surface file-not-found and parse errors
        return None, repr(e)


def resolve_bandpasses(
    band_keys: list[str],
    *,
    band_aliases: dict[str, str],
    trds_roots: list[str],
) -> tuple[dict[str, Any], FilterRoundReport]:
    """Return ``(bandpass_by_key, report)``. Missing bands are omitted from the dict."""
    eff, tried = configure_pysynphot_environment(trds_roots)
    S, ierr = import_pysynphot()
    rep = FilterRoundReport(
        trds_roots_tried=tried,
        pysyn_cdbS_effective=eff,
        pysynphot_import_error=ierr,
    )
    if S is None:
        for bk in band_keys:
            rep.bands_skipped_missing.append(
                {
                    "band_key": bk,
                    "reason": "MISSING_FILTER",
                    "detail": rep.pysynphot_import_error or "pysynphot import failed",
                }
            )
        return {}, rep

    resolved: dict[str, Any] = {}
    for bk in band_keys:
        attempt_order: list[str] = []
        if bk in band_aliases:
            attempt_order.append(band_aliases[bk])
        attempt_order.append(bk)
        if bk.lower() != bk:
            attempt_order.append(bk.lower())

        last_err = None
        ok_bp = None
        used_attempt = ""
        tried_here: set[str] = set()
        for cand in attempt_order:
            if not cand or cand in tried_here:
                continue
            tried_here.add(cand)
            bp, err = _try_bandpass(S, cand)
            used_attempt = cand
            if bp is not None and err is None:
                ok_bp = bp
                break
            last_err = err

        if ok_bp is not None:
            resolved[bk] = ok_bp
            rep.bands_used.append(
                {
                    "band_key": bk,
                    "pysyn_name": used_attempt,
                    "trds_root": eff,
                }
            )
        else:
            rep.bands_skipped_missing.append(
                {
                    "band_key": bk,
                    "reason": "MISSING_FILTER",
                    "pysyn_name_tried": used_attempt,
                    "detail": last_err or "unknown",
                    "trds_root": eff,
                }
            )

    return resolved, rep


def synthesize_effstim(
    wave_aa: np.ndarray,
    flux: np.ndarray,
    bandpass: Any,
    *,
    system: str = "ab",
) -> tuple[Optional[float], Optional[str]]:
    """AB (or other) effective stimulus via ``Observation.effstim``."""
    S, ierr = import_pysynphot()
    if S is None:
        return None, ierr
    try:
        wave_aa = np.asarray(wave_aa, dtype=float).ravel()
        flux = np.asarray(flux, dtype=float).ravel()
        ok = np.isfinite(wave_aa) & np.isfinite(flux) & (wave_aa > 0)
        if not np.any(ok):
            return None, "no finite spectral points"
        sp = S.ArraySpectrum(
            wave=wave_aa[ok],
            flux=flux[ok],
            waveunits="angstrom",
            fluxunits="flam",
        )
        obs = S.Observation(sp, bandpass)
        return float(obs.effstim(system)), None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def synthesize_for_bands(
    wave_aa: np.ndarray,
    flux: np.ndarray,
    band_keys: list[str],
    *,
    band_aliases: dict[str, str],
    trds_roots: list[str],
    system: str = "ab",
) -> tuple[dict[str, float], FilterRoundReport]:
    """Synthetic values only for bands that resolve and synthesize successfully."""
    bps, rep = resolve_bandpasses(band_keys, band_aliases=band_aliases, trds_roots=trds_roots)
    out: dict[str, float] = {}
    for bk, bp in bps.items():
        val, err = synthesize_effstim(wave_aa, flux, bp, system=system)
        if val is not None and err is None:
            out[bk] = val
        else:
            rep.bands_skipped_missing.append(
                {
                    "band_key": bk,
                    "reason": "SYNTHESIS_FAILED",
                    "detail": err or "effstim failed",
                }
            )
    return out, rep


def save_filter_report(path: str, report: FilterRoundReport) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_jsonable(), f, indent=2)


def load_filter_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError("install PyYAML to load YAML configs (`pip install pyyaml`)") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_filter_config(path: str) -> tuple[list[str], dict[str, str], str]:
    """Return ``(trds_roots, band_aliases, photometry_system)``."""
    cfg = load_filter_yaml(path)
    roots = list(cfg.get("trds_roots") or [])
    aliases = dict(cfg.get("band_aliases") or {})
    sysname = str(cfg.get("photometry_system") or "ab").lower()
    return roots, aliases, sysname


def summarize_report(report: FilterRoundReport) -> str:
    lines = [
        f"TRDS effective: {report.pysyn_cdbS_effective!r}",
        f"pysynphot error: {report.pysynphot_import_error}",
        f"bands_used: {len(report.bands_used)}",
        f"bands_skipped_missing: {len(report.bands_skipped_missing)}",
    ]
    for b in report.bands_skipped_missing:
        lines.append(f"  SKIP {b.get('band_key')!r}: {b.get('reason')} — {b.get('detail')}")
    return "\n".join(lines)


def main_cli() -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Resolve bands and optionally write filter_report.json")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "filter_pipeline.example.yaml"))
    p.add_argument("--bands", default="", help="comma-separated band keys to try")
    p.add_argument("--out-report", default=None)
    args = p.parse_args()
    roots, aliases, _sys = load_filter_config(args.config)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    if not bands:
        print("provide --bands or use API from Python", file=sys.stderr)
        return 1
    _bps, rep = resolve_bandpasses(bands, band_aliases=aliases, trds_roots=roots)
    print(summarize_report(rep))
    if args.out_report:
        save_filter_report(args.out_report, rep)
        print("wrote", args.out_report)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main_cli())
