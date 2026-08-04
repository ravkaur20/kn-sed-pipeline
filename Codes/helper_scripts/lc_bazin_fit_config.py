"""Bazin early-LC fit configuration for notebook 2 / ``lc_bazin_fit``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import pipeline_config as pconf

__all__ = [
    "BazinFitConfig",
    "default_bazin_fit_config",
    "load_se_sne_names",
    "load_sn_type_lists",
]

# Legacy SN entries (non-AT2017gfo defaults from NB2 cell 8).
_LEGACY_EXPLOSION_DATES: dict[str, tuple[Any, Any, Any]] = {
    "iPTF13bvn": (56458.6, None, None),
    "SN1993J": (None, None, None),
    "SN2011bm": (None, None, None),
}

_SE_TYPES = ("IIb", "Ib", "Ic", "Ic-BL", "Ibc-pec")
_HYDR_TYPES = ("IIn", "II", "IIL", "IIP", "1987A", "87A")


@dataclass
class BazinFitConfig:
    explosion_dates: dict[str, tuple[Any, Any, Any]]
    se_sne: list[str] = field(default_factory=list)
    hydr_sne: list[str] = field(default_factory=list)
    n_sesn: float = 1.5
    n_hydr: float = 0.935
    nfree: list[str] = field(default_factory=lambda: ["iPTF13bvn", "AT2017gfo"])
    sn_n_fix: dict[str, float] = field(default_factory=dict)
    pre_bump: dict[str, list[Any]] = field(default_factory=lambda: {"mySN": [15.0, 55712.5, 2.0, None]})
    include_dict: dict[str, list[str]] = field(default_factory=lambda: {"iPTF13bvn": ["swift_UVW1"]})
    no_bessell_v_use_swift_v: list[str] = field(
        default_factory=lambda: ["SN2008aq", "SN2011ht", "SN2006aj", "SN2013ge"]
    )
    exclude_dict: dict[str, list[str]] = field(default_factory=dict)


def load_sn_type_lists(datainfo_path: str) -> tuple[list[str], list[str]]:
    """Load SE- and hydrogen-rich SN name lists from ``info.dat``."""
    info_objects = pd.read_csv(datainfo_path + "info.dat", comment="#", delimiter=" ")
    se_sne = [row.Name for _, row in info_objects.iterrows() if row.Type in _SE_TYPES]
    hydr_sne = [row.Name for _, row in info_objects.iterrows() if row.Type in _HYDR_TYPES]
    return se_sne, hydr_sne


def load_se_sne_names(datainfo_path: str) -> list[str]:
    """Return stripped-envelope SN names from ``info.dat``."""
    se_sne, _ = load_sn_type_lists(datainfo_path)
    return se_sne


def default_bazin_fit_config(snname: str, *, se_sne: list[str] | None = None, hydr_sne: list[str] | None = None) -> BazinFitConfig:
    """Build default Bazin fit config for ``snname``."""
    explosion_dates = dict(_LEGACY_EXPLOSION_DATES)
    if snname == "AT2017gfo" or snname not in explosion_dates:
        explosion_dates[snname] = (pconf.explosion_date_mjd(snname), None, None)
    return BazinFitConfig(
        explosion_dates=explosion_dates,
        se_sne=list(se_sne or []),
        hydr_sne=list(hydr_sne or []),
    )
