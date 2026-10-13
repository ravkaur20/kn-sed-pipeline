"""Milky Way + host dust for the final SED (one implementation for the whole pipeline).

The light curves and spectra are fit **as observed** (steps 0-6 never deredden). The
reddening is removed afterwards, wavelength by wavelength, from the FINAL SED::

    F_dered(lam) = F_obs(lam) * 10^(+0.4 [A_MW(lam) + A_host(lam / (1+z))])

MW extinction is evaluated on the observed-frame wavelength, host extinction on the
rest-frame wavelength. ``A(lam) = law(lam, R_V * E(B-V), R_V)`` from the ``extinction``
package; the law is ``pipeline_config.DUST_LAW``. ``E(B-V)``, ``R_V`` and ``z`` come
from ``Inputs/SNe_Info/info.dat`` (columns ``EBV_MW``, ``EBV_host``, ``RV_MW``,
``RV_host``, ``z``). Flux errors are scaled by the same factor (dust parameters exact).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

DUST_LAWS: tuple[str, ...] = ("ccm89", "odonnell94", "fitzpatrick99", "fm07")

INFO_DUST_COLUMNS: tuple[str, ...] = ("z", "EBV_MW", "EBV_host", "RV_MW", "RV_host")

DUST_PARAMS_JSON = "dust_params.json"

__all__ = [
    "DUST_LAWS",
    "INFO_DUST_COLUMNS",
    "DUST_PARAMS_JSON",
    "extinction_mag",
    "total_extinction_mag",
    "load_dust_params",
    "deredden_spectrum",
    "deredden_final_spectra",
    "dust_params_mismatch",
]


def extinction_mag(wls: np.ndarray, ebv: float, r_v: float, law: str = "ccm89") -> np.ndarray:
    """A(lam) [mag] at wavelengths ``wls`` [Angstrom, frame of the dust]."""
    if law not in DUST_LAWS:
        raise ValueError("dust law must be one of %s, got %r" % (DUST_LAWS, law))
    w = np.asarray(wls, dtype=float)
    if not ebv:
        return np.zeros_like(w)
    import extinction

    if law == "fm07":
        if not np.isclose(r_v, 3.1):
            raise ValueError("fm07 is defined only for R_V = 3.1, got R_V = %g" % r_v)
        return extinction.fm07(w, r_v * ebv)
    return getattr(extinction, law)(w, r_v * ebv, r_v)


def total_extinction_mag(
    wls_obs: np.ndarray,
    *,
    ebv_mw: float,
    rv_mw: float,
    ebv_host: float,
    rv_host: float,
    redshift: float,
    law: str = "ccm89",
) -> np.ndarray:
    """A_MW(lam_obs) + A_host(lam_obs / (1+z)) [mag] on observed-frame ``wls_obs``."""
    w = np.asarray(wls_obs, dtype=float)
    return extinction_mag(w, ebv_mw, rv_mw, law) + extinction_mag(
        w / (1.0 + redshift), ebv_host, rv_host, law
    )


def load_dust_params(info_path: str, snname: str) -> dict[str, float]:
    """``ebv_mw``, ``rv_mw``, ``ebv_host``, ``rv_host``, ``redshift`` for ``snname``.

    Raises ``KeyError`` when the object or any of :data:`INFO_DUST_COLUMNS` is missing
    (or not a number) in ``info.dat``; no default is ever substituted.
    """
    info = pd.read_csv(info_path, comment="#", sep=r"\s+")
    row = info[info["Name"] == snname]
    if row.empty:
        raise KeyError("%s not in %s" % (snname, info_path))
    r = row.iloc[0]
    vals = {}
    for col in INFO_DUST_COLUMNS:
        v = pd.to_numeric(r[col], errors="coerce") if col in r.index else np.nan
        if not np.isfinite(v):
            raise KeyError(
                "info.dat %s has no numeric %r for %s; add it (dust columns: %s)"
                % (info_path, col, snname, ", ".join(INFO_DUST_COLUMNS))
            )
        vals[col] = float(v)
    return {
        "ebv_mw": vals["EBV_MW"],
        "rv_mw": vals["RV_MW"],
        "ebv_host": vals["EBV_host"],
        "rv_host": vals["RV_host"],
        "redshift": vals["z"],
    }


def deredden_spectrum(
    wls: np.ndarray, flux: np.ndarray, fluxerr: np.ndarray, *, law: str = "ccm89", **dust: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dereddened ``(flux, fluxerr)``; ``dust`` = :func:`load_dust_params` output."""
    factor = 10.0 ** (0.4 * total_extinction_mag(wls, law=law, **dust))
    return np.asarray(flux, dtype=float) * factor, np.asarray(fluxerr, dtype=float) * factor


def deredden_final_spectra(
    src_dir: str, dst_dir: str, *, law: str = "ccm89", **dust: float
) -> list[str]:
    """Deredden every ``*.txt`` FINAL spectrum in ``src_dir`` into ``dst_dir``.

    Files keep their names and the ``wls flux fluxerr`` layout (linear Angstrom, linear
    F_lambda). ``dst_dir/dust_params.json`` records the law and parameters used.
    """
    os.makedirs(dst_dir, exist_ok=True)
    written: list[str] = []
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".txt"):
            continue
        d = np.atleast_2d(np.loadtxt(os.path.join(src_dir, fn), dtype=float))
        w, f = d[:, 0], d[:, 1]
        e = d[:, 2] if d.shape[1] > 2 else np.zeros_like(f)
        f_d, e_d = deredden_spectrum(w, f, e, law=law, **dust)
        dst = os.path.join(dst_dir, fn)
        np.savetxt(dst, np.column_stack([w, f_d, e_d]), fmt="%.6E", delimiter="\t",
                   header="wls\tflux\tfluxerr", comments="#")
        written.append(dst)
    meta: dict[str, Any] = {"law": law, "source_dir": os.path.abspath(src_dir), **dust,
                            "n_spectra": len(written)}
    with open(os.path.join(dst_dir, DUST_PARAMS_JSON), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return written


def dust_params_mismatch(dered_dir: str, *, law: str, **dust: float) -> list[str]:
    """Differences between ``dered_dir/dust_params.json`` and the given law/parameters.

    Empty when they agree; one ``"key: on disk X, now Y"`` string per difference
    (a missing JSON is reported as one entry).
    """
    path = os.path.join(dered_dir, DUST_PARAMS_JSON)
    if not os.path.isfile(path):
        return ["%s missing" % path]
    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    out = []
    for key, now in [("law", law), *dust.items()]:
        old = on_disk.get(key)
        same = old == now if isinstance(now, str) else (
            old is not None and np.isclose(float(old), float(now), rtol=0, atol=1e-9))
        if not same:
            out.append("%s: on disk %r, now %r" % (key, old, now))
    return out
