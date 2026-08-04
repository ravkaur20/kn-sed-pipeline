"""Deprecated: use ``iter_plot_suite``. Thin compatibility wrappers for tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

from iter_plot_suite import (
    plot_gp_vs_mangled,
    plot_mangled_synphot_lc,
    run_iter_plot_suite,
)


def save_mangled_vs_gp_diagnostics(diag_dir: str, chain_data: Sequence[Mapping[str, Any]]) -> list[str]:
    import os

    base = os.path.basename(diag_dir.rstrip(os.sep))
    if base == "gp_vs_mangled":
        figs_root = os.path.dirname(diag_dir.rstrip(os.sep))
    elif base == "figs":
        figs_root = diag_dir.rstrip(os.sep)
    else:
        figs_root = os.path.join(diag_dir, "figs")
    return plot_gp_vs_mangled(figs_root, chain_data)


def save_gpfit_photometry_lc_diagnostics(
    diag_dir: str,
    *,
    snname: str,
    mangled_dir: str,
    phot4mangling_path: str,
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    avail_filters: Sequence[str],
    filter_path: str,
    spec_list_entries: Sequence[Any],
    csp_sne: Sequence[str] = (),
) -> list[str]:
    del spec_list_entries
    import os

    figs_root = os.path.join(os.path.dirname(diag_dir), "figs")
    if not os.path.isdir(figs_root):
        figs_root = diag_dir
    return plot_mangled_synphot_lc(
        figs_root,
        snname=snname,
        mangled_dir=mangled_dir,
        phot4mangling_path=phot4mangling_path,
        filter_mjd_dict=filter_mjd_dict,
        avail_filters=avail_filters,
        filter_path=filter_path,
        csp_sne=csp_sne,
    )


def save_iter_compare_diagnostics(
    iter_dir: str,
    *,
    chain_data: Sequence[Mapping[str, Any]],
    snname: str,
    mangled_dir: str,
    phot4mangling_path: str,
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    avail_filters: Sequence[str],
    filter_path: str,
    spec_list_entries: Sequence[Any] | None = None,
    csp_sne: Sequence[str] = (),
    old_masks: Mapping[str, Any] | None = None,
    new_masks: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    del spec_list_entries
    om = old_masks or {}
    nm = new_masks or {}
    return run_iter_plot_suite(
        iter_dir,
        chain_data=chain_data,
        old_masks=om,
        new_masks=nm,
        mangled_dir=mangled_dir,
        phot4mangling_path=phot4mangling_path,
        filter_mjd_dict=filter_mjd_dict,
        avail_filters=avail_filters,
        filter_path=filter_path,
        snname=snname,
        csp_sne=csp_sne,
    )
