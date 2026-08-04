"""Shared band colors, markers, and filter exclusion lists for LC/mangle plotting.

Populated from ``filter_plot_config.json`` via ``pipeline_config.load_filter_plot_style()``.
"""

from __future__ import annotations

from typing import Optional

_DEFAULT_COLOR_DICT: dict[str, str] = {
    'Swope_i': 'indianred', 'FourStar_H': 'darkred', 'FourStar_J': 'black', 'DECam_i': 'orangered',
    'DECam_z': 'darkslategray', 'DECam_Y': 'goldenrod', 'FourStar_Ks': 'brown', 'UVOT_U': 'teal',
    'Skymapper_i': 'slateblue', 'Sinistro_g': 'darkgreen', 'Sinistro_r': 'crimson',
    'Skymapper_r': 'deeppink', 'Skymapper_g': 'forestgreen', 'EFOSC2_V': 'limegreen',
    'Sinistro_i': 'firebrick', 'DECam_r': 'darkorange', 'DECam_g': 'mediumspringgreen',
    'DECam_u': 'navy', 'Swope_V': 'green', 'FourStar_J1': 'dimgray', 'Swope_B': 'royalblue',
    'Swope_r': 'red', 'Swope_g': 'darkseagreen', 'EFOSC2_U': 'blue', 'Sinistro_V': 'mediumaquamarine',
    'Sinistro_z': 'sienna', 'IMACS_r': 'darkred', 'LRIS_I': 'mediumvioletred', 'SOFI_H': 'firebrick',
    'SOFI_Ks': 'saddlebrown', 'VISTA_Ks': 'saddlebrown', 'VISTA_J': 'dimgray', 'VISTA_Y': 'salmon',
    'FLAMINGOS-2_Ks': 'sienna', 'UVOT_M2': 'darkviolet', 'UVOT_W1': 'purple', 'UVOT_W2': 'indigo',
    'HSC_z': 'darkslategray', 'GFC_i': 'firebrick', 'GFC_y': 'goldenrod', 'GFC_z': 'maroon',
    'GFC_r': 'tomato', 'SIRIUS_H': 'darkred', 'SIRIUS_J': 'black', 'SIRIUS_Ks': 'brown',
    'T80Cam_g': 'seagreen', 'GROND_H': 'maroon', 'GROND_J': 'dimgray', 'GROND_K': 'saddlebrown',
    'GROND_g': 'mediumseagreen', 'GROND_i': 'orangered', 'GROND_r': 'crimson', 'GROND_z': 'darkslategray',
    'FLAMINGOS-2_H': 'firebrick', 'IMACS_i': 'orangered', 'FLAMINGOS-2_J': 'black', 'UVOT_B': 'dodgerblue',
    'GMOS_g': 'green', 'GMOS_i': 'indianred', 'GMOS_r': 'red', 'GMOS_z': 'darkslategray',
    'FORS2_R': 'crimson', 'VIMOS_z': 'maroon', 'FORS2_I': 'mediumvioletred', 'FORS2_B': 'blue',
    'FORS2_V': 'limegreen', 'ANDICAM_K': 'brown', 'MOIRCS_Ks': 'sienna', 'HAWKI_Ks': 'brown',
}

_DEFAULT_MARK_DICT: dict[str, str] = {
    'Swope_i': 'o', 'Swope_V': 'o', 'Swope_B': 'o', 'Swope_r': 'o', 'Swope_g': 'o',
    'FourStar_H': 'o', 'FourStar_J': 'o', 'FourStar_Ks': 'o', 'FourStar_J1': 'o',
    'VISTA_Ks': 'o', 'VISTA_J': 'o', 'VISTA_Y': 'o', 'SOFI_H': 'o', 'SOFI_Ks': 'o', 'ANDICAM_K': 'o',
    'DECam_i': '^', 'DECam_z': '^', 'DECam_Y': '^', 'DECam_r': '^', 'DECam_g': '^', 'DECam_u': '^',
    'HSC_z': '>', 'MOIRCS_Ks': '>', 'FLAMINGOS-2_Ks': 'v', 'FLAMINGOS-2_H': 'v', 'FLAMINGOS-2_J': 'v',
    'GMOS_g': '<', 'GMOS_i': '<', 'GMOS_r': '<', 'GMOS_z': '<', 'FORS2_R': '<', 'FORS2_I': '<',
    'FORS2_B': '<', 'FORS2_V': '<', 'VIMOS_z': '<', 'HAWKI_Ks': '<', 'UVOT_M2': 'X', 'UVOT_W1': 'X',
    'UVOT_U': 'X', 'UVOT_W2': 'X', 'UVOT_B': 'X', 'GFC_i': 'p', 'GFC_y': 'p', 'GFC_z': 'p', 'GFC_r': 'p',
    'Skymapper_i': 'P', 'Skymapper_r': 'P', 'Skymapper_g': 'P', 'Sinistro_g': 's', 'Sinistro_r': 's',
    'Sinistro_i': 's', 'Sinistro_V': 's', 'Sinistro_z': 's', 'IMACS_i': 's', 'IMACS_r': 's',
    'SIRIUS_H': 'D', 'SIRIUS_J': 'D', 'SIRIUS_Ks': 'D', 'EFOSC2_V': 'D', 'EFOSC2_U': 'D', 'LRIS_I': 'D',
    'T80Cam_g': '*', 'GROND_H': 'h', 'GROND_J': 'h', 'GROND_K': 'h', 'GROND_g': 'h', 'GROND_i': 'h',
    'GROND_r': 'h', 'GROND_z': 'h',
}

_DEFAULT_EXCLUDE: list[str] = []

COLOR_DICT: dict[str, str] = dict(_DEFAULT_COLOR_DICT)
MARK_DICT: dict[str, str] = dict(_DEFAULT_MARK_DICT)
EXCLUDE_FILT: list[str] = list(_DEFAULT_EXCLUDE)

color_dict = COLOR_DICT
mark_dict = MARK_DICT
exclude_filt = EXCLUDE_FILT


def refresh_from_config() -> None:
    """Reload style dicts from ``filter_plot_config.json``."""
    import pipeline_config as pc  # noqa: PLC0415

    style = pc.load_filter_plot_style()
    COLOR_DICT.clear()
    COLOR_DICT.update(style.color_dict)
    MARK_DICT.clear()
    MARK_DICT.update(style.mark_dict)
    EXCLUDE_FILT[:] = list(style.exclude_filt)
    globals()["color_dict"] = COLOR_DICT
    globals()["mark_dict"] = MARK_DICT
    globals()["exclude_filt"] = EXCLUDE_FILT


def band_color(band: str, default: Optional[str] = None) -> Optional[str]:
    """Return matplotlib color for ``band`` (handles bytes)."""
    key = band.decode() if isinstance(band, bytes) else str(band)
    return COLOR_DICT.get(key, default)


def band_marker(band: str, default: str = "o") -> str:
    """Return matplotlib marker for ``band`` (handles bytes)."""
    key = band.decode() if isinstance(band, bytes) else str(band)
    return MARK_DICT.get(key, default)


try:
    refresh_from_config()
except Exception:
    pass
