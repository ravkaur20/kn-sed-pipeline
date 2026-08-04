# Phase 1 — Pre-scale spectra + mangling I/O (completed)

Phase 1 of the iterative GP+mangle pipeline. Phases 2–4 remain in the master plan.

## What was built

| Deliverable | Path |
|-------------|------|
| Pre-scale engine | [`spectra_pre_scale.py`](../spectra_pre_scale.py) |
| Pre-scale diagnostics | [`spec_scale_diagnostics.py`](../spec_scale_diagnostics.py) |
| Mangling helpers (I/O, mask, NB5 extract) | [`mangle_spectra_log.py`](../mangle_spectra_log.py) |
| Config / paths | [`pipeline_config.py`](../pipeline_config.py) (`SPEC_SCALE_*`, `USE_PRESCALED_SPECTRA`) |
| Notebook | [`4.5_Scale_spectra_KN.ipynb`](../4.5_Scale_spectra_KN.ipynb) |
| NB5 reads prescaled list | [`5_Mangle_spectra_KN_log.ipynb`](../5_Mangle_spectra_KN_log.ipynb) (`spec_list_path_for_mangling`) |
| Tests | [`tests/test_spectra_pre_scale.py`](../tests/test_spectra_pre_scale.py), [`tests/test_mangle_spectra_log.py`](../tests/test_mangle_spectra_log.py), [`tests/test_pipeline_config_prescale.py`](../tests/test_pipeline_config_prescale.py) |

## Run order (so far)

0.1 → 1 → 2 → 4 → **4.5** → **5** → *(Phase 2+: iterative GP)*

## Scaling methods (4.5)

All scaling uses **`2_spec_smoothed`** only. Default mode is **star-to-highest-SNR** reference arm (usually VIS).

Per non-reference arm:

1. Overlap WLS on smoothed spectra (when 0.1 left overlap)
2. Gap-seam: median flux at join edges when no overlap and gap ≤ `SPEC_SCALE_GAP_MAX_A`
3. **Star bridge fallback** (default on): if direct scale to the anchor skips because arms are non-adjacent (e.g. UVB→NIR when NIR is anchor), scale via the next arm in `merge_order` that was already aligned (e.g. UVB→scaled VIS after VIS→NIR)
4. Skip (factor 1.0) with reason in report JSON

Config knobs in [`pipeline_config.py`](../pipeline_config.py):

| Knob | Default | Meaning |
|------|---------|---------|
| `SPEC_SCALE_CHAIN_MODE` | `False` | If `True`, chain-scale along `merge_order`; else star to highest-SNR arm |
| `SPEC_SCALE_STAR_BRIDGE_FALLBACK` | `True` | One-hop bridge in star mode when direct anchor scale skips |
| `SPEC_SCALE_GAP_MAX_A` | `400` | Max join gap (Å) for gap-seam scaling |
| `SPEC_SCALE_EDGE_N_PIX` | `10` | Fallback edge pixel count for gap-seam |
| `SPEC_SCALE_SEAM_HALF_WIDTH_A` | `50` | Edge window half-width (Å) for gap-seam |

Report JSON includes `pair_links` per group: `method`, `gap_a`, `n_overlap`, and when used `fallback: bridge` with `direct_method` / `direct_gap_a`.

## Notebook 4.5 — quick start

1. Set `COCO_PATH` and `snname` in the config cell (defaults use `pipeline_config.SNNAME_DEFAULT`).
2. Run the template cell: creates `Outputs/<SN>/<SN>_spec_scale_groups.json` from MJD clustering if missing.
3. **Edit the JSON** to confirm XSHooter triplets (`members`, optional `merge_order`: `uvb`, `vis`, `nir`).
4. Run the scaling cell.

### Output modes

- **`scale_only`** (default in `pipeline_config.SPEC_SCALE_OUTPUT_MODE`): align flux; **keep separate files**.
- **`merge_join`**: after scaling, write one merged spectrum per group (global or per-group `"output_mode"` in JSON).

### Outputs

- `Inputs/Spectroscopy/2_spec_prescaled/<SN>/`
- `Inputs/Spectroscopy/2_spec_lists_prescaled/<SN>.list`
- `Outputs/<SN>/<SN>_spec_scale_report.json`
- `Outputs/<SN>/spec_scale_diagnostics/index.html` (when matplotlib works in your kernel)

## Tests

```bash
cd Codes
PYTHONPATH=. python3 -m unittest tests.test_spectra_pre_scale tests.test_mangle_spectra_log tests.test_pipeline_config_prescale -v
```

## Smoke run (CLI)

```bash
cd Codes
PYTHONPATH=. python3 -c "
import pipeline_config as p
from spectra_pre_scale import run_prescale_pipeline
run_prescale_pipeline(
    snname='AT2017gfo',
    coco_path=p.COCO_PATH,
    output_dir=p.COCO_PATH + 'Outputs/',
    diagnostics_dir=p.spec_scale_diagnostics_dir(p.COCO_PATH + 'Outputs/', 'AT2017gfo'),
)
"
```

On AT2017gfo, auto-grouping finds 10 same-time clusters (mostly XSHooter UVB/VIS/NIR + Magellan pairs); 9 ungrouped spectra are copied unchanged.

## Bundle-aware mangling (optional, NB5)

After prescale, set `MANGLE_BUNDLE_AWARE = True` in notebook 5 (or `pipeline_config.py`) so same-time arms share **one mangling mask** (pooled photometry constraints). With `MANGLE_BUNDLE_STITCH_SYNPHOT = True` (default), filters spanning multiple arms use **stitched** synphot across arms in merge order when combined coverage beats any single arm; otherwise per-arm best overlap applies. Diagnostics: `Outputs/<SN>/mangle_diagnostics/index.html`. Modules: [`mangle_spectra_log.py`](../mangle_spectra_log.py), [`mangle_diagnostics.py`](../mangle_diagnostics.py).

Set `MANGLE_RUN_BOTH_FOR_DIAG = True` to generate **`group_*_prescaled_perarm_bundle.pdf`** (prescaled, per-arm mangled, and bundle mangled on one plot) for grouped arms while keeping per-arm mangled files on disk.

## Not in Phase 1

- 2D GP sync (Phase 2)
- Iterative GP+mangle loop (Phase 3) — see [`PHASE3_ITER_GP_MANGLE.md`](PHASE3_ITER_GP_MANGLE.md)
- 7.5 path toggles for `twodim_iter` (Phase 4)

See master plan: `.cursor/plans/gp_pipeline_review_bf44b5a2.plan.md` (or workspace plan copy).
