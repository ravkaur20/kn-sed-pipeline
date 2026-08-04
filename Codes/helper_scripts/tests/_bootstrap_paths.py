"""Shared sys.path setup for kn-sed-pipeline tests."""

from __future__ import annotations

import os
import sys

HELPER_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODES = os.path.abspath(os.path.join(HELPER_SCRIPTS, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CODES, ".."))

for _p in (CODES, HELPER_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TWODIM_GP = os.path.join(HELPER_SCRIPTS, "twodim_gp")
