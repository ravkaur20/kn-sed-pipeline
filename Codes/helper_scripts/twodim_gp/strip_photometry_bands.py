#!/usr/bin/env python3
"""Remove entire photometry light curves (fixed effective wavelength bands) from a training NPZ.

Rows are dropped **before** ``bundle_scale_pipeline`` so intra-bundle scaling and photometric
anchoring never see those bands.

Band membership matches overview pseudo-bands: ``round(X[:,0], round_digits)`` compared to
each target (defaults: ``round_digits=4``, targets ``-0.8767,-0.8217``).

Arrays with first dimension ``N_train`` are sliced; ``X_fill`` and prior interpolant grids are
copied unchanged.

Example::

    python strip_photometry_bands.py \\
        -i gp_bundle_collab_fixes.npz -o gp_bundle_collab_fixes_nophot_m8767_m8217.npz \\
        --bands -0.8767,-0.8217
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import gp_utils as gu


def _parse_bands(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument(
        "--bands",
        default="-0.8767,-0.8217",
        help="comma-separated target band keys (same units as rounded X[:,0])",
    )
    p.add_argument("--round-digits", type=int, default=4)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    ns = p.parse_args(argv)

    inp = ns.input if os.path.isabs(ns.input) else os.path.join(here, ns.input)
    outp = ns.output if os.path.isabs(ns.output) else os.path.join(here, ns.output)
    if not os.path.isfile(inp):
        print(f"ERROR: input not found {inp!r}", file=sys.stderr)
        return 2

    bands = _parse_bands(ns.bands)
    if not bands:
        print("ERROR: empty --bands", file=sys.stderr)
        return 2

    bd = np.load(inp, allow_pickle=False)
    try:
        keys = list(bd.files)
        n0 = int(np.asarray(bd["X"]).shape[0])
        X = np.asarray(bd["X"], dtype=float)
        obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
        pc = gu.effective_point_class(
            X,
            train_obs_class=np.asarray(obs) if obs is not None else None,
            threshold=int(ns.phot_spec_threshold),
        )
        rk = np.round(X[:, 0], int(ns.round_digits))
        targets = np.asarray(bands, dtype=float)
        drop = (pc == gu.PHOT) & np.isin(rk, targets)
        keep = ~drop
        n_drop = int(np.sum(drop))
        n_keep = int(np.sum(keep))
        print(
            f"[strip_phot_bands] N={n0}  drop_phot={n_drop}  keep={n_keep}  "
            f"round_digits={ns.round_digits}  targets={bands}"
        )
        if n_drop == 0:
            print("[strip_phot_bands] WARN: no rows matched; copy may be unchanged except path", file=sys.stderr)

        payload: dict[str, np.ndarray] = {}
        for k in keys:
            a = bd[k]
            if not isinstance(a, np.ndarray):
                continue
            if a.shape and int(a.shape[0]) == n0:
                payload[k] = np.asarray(a)[keep].copy()
            else:
                payload[k] = np.asarray(a).copy()
    finally:
        bd.close()

    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    np.savez_compressed(outp, **payload)
    print(f"[strip_phot_bands] wrote {outp!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
