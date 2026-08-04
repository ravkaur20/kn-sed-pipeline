"""Shared bootstrap for numbered pipeline scripts in ``Codes/``."""

from __future__ import annotations

import os
import sys


def bootstrap_script_paths() -> str:
    """Insert ``Codes/`` and ``helper_scripts/`` on ``sys.path``; return Codes dir."""
    codes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    helper_dir = os.path.join(codes_dir, "helper_scripts")
    for p in (codes_dir, helper_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return codes_dir
