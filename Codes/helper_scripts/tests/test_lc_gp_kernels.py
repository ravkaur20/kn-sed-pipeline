"""Tests for lc_gp_kernels."""

import _bootstrap_paths  # noqa: F401
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_gp_kernels import (
    load_kernel_settings_file,
    resolve_kernel_for_band,
    set_default_kernel_settings,
)


def test_default_uv_band():
    scale, opt, mean = set_default_kernel_settings("swift_U")
    assert scale == 150.0
    assert opt is True


def test_default_optical():
    scale, opt, mean = set_default_kernel_settings("VISTA_J")
    assert scale == 10.0
    assert opt is False


def test_file_format_band_dict(tmp_path):
    p = tmp_path / "k.json"
    p.write_text('{"VISTA_J": {"scale": 20, "opt": true, "mean": 0.5}}')
    loaded = load_kernel_settings_file(str(p))
    scale, opt, mean = resolve_kernel_for_band("AT2017gfo", "VISTA_J", loaded)
    assert scale == 20.0
    assert opt is True
    assert mean == 0.5
