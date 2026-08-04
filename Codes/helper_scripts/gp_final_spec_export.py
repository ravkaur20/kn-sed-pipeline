"""Convert iter ``full_gp`` ``_spec_extended*.txt`` to NB7-style ``_FINAL_spec*.txt`` for 7.5 QA."""

from __future__ import annotations

import os
import shutil


_SUFFIX_MAP = (
    ("_spec_extended_FL.txt", "_FINAL_spec_FL.txt"),
    ("_spec_extended.txt", "_FINAL_spec.txt"),
)


def export_final_spec_from_full_gp(
    full_gp_dir: str,
    final_spectra_dir: str,
    *,
    variant: str = "as_observed",
) -> list[str]:
    """Copy/rename GP extended spectra into ``FINAL_spectra_2dim/<variant>/``."""
    if not os.path.isdir(full_gp_dir):
        return []
    out_dir = (
        final_spectra_dir
        if not variant
        else os.path.join(final_spectra_dir, variant)
    )
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for fn in sorted(os.listdir(full_gp_dir)):
        if not fn.endswith(".txt"):
            continue
        dst_name = None
        for src_suf, dst_suf in _SUFFIX_MAP:
            if fn.endswith(src_suf):
                dst_name = fn[: -len(src_suf)] + dst_suf
                break
        if dst_name is None:
            continue
        src_p = os.path.join(full_gp_dir, fn)
        dst_p = os.path.join(out_dir, dst_name)
        shutil.copy2(src_p, dst_p)
        written.append(dst_p)
    return written
