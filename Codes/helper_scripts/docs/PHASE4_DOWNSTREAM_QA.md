# Phase 4 — Downstream QA (7.5 notebooks + FINAL export)

Phase 4 wires iterative GP+mangle **science products** into the existing 7.5 comparison / spectra / trapz notebooks.

## Product paths (two formats)

### Primary (NB6 convention, pure GP `full_gp`)

```
Outputs/<SN>/twodim_iter/<extend|extrapolate>/
  iter_KK/gp_runs/full_gp/{Log_Phase}_spec_extended.txt
  iter_KK/gp_runs/full_gp/{Log_Phase}_spec_extended_FL.txt   # extrapolate-only phases
  final/full_gp/*.txt
```

Linear Å, linear F_λ. **Not spliced.** Surface key: `GP_PREDICT_MU_KEY` (`"mu"` default).

### Legacy 7.5 QA export (`_FINAL_spec`)

When `ITER_EXPORT_FINAL_SPEC_FOR_QA = True` (default), the driver copies/renames extended spectra into:

```
Outputs/<SN>/FINAL_spectra_2dim/twodim_iter/<mode>/full_gp/as_observed/{stem}_FINAL_spec.txt
```

Same layout as NB6/7 FINAL products so existing 7.5 loaders work unchanged.

## Config toggles

| Knob | Default | Role |
|------|---------|------|
| `USE_ITER_GP_MANGLE_FINAL` | `True` | 7.5 notebooks: read `twodim_iter/...` branch |
| `ITER_EXPORT_FINAL_SPEC_FOR_QA` | `True` | Write `_FINAL_spec*` under `FINAL_spectra_2dim/twodim_iter/...` |
| `ITER_GP_EXPORT_FULL_GP` | `True` | Write `_spec_extended*` each iteration |
| `MANGLE_PHOTOMETRY_TARGET` | `"gp_fit"` | Mangling uses NB4 `fitted_phot4mangling_<SN>.dat` only |

Mutually exclusive legacy branches (RJF / RyanV2) are not included in this repo; use `twodim_iter/...` only.

## 7.5 notebook usage

1. Run production chain: **4.5 → 5 → 6_Iterative_GP_mangle_KN** (set `ITER_GP_MANGLE_MAX_ITERS` as needed).
2. In `7.5_comparison_check_log`, `7.5_spectra`, or `7.5_alternate`:
   - `USE_ITER_GP_MANGLE_FINAL = True` (default in `pipeline_config.py`)
   - Re-run config + `data_dir` / `create_lookup_table` / trapz compute cells
3. `FINAL_SUFFIXES_TO_LOAD`: when iter toggle is on and left `None`, notebooks default to `SPECTRUM_STEM_SUFFIXES` (both `_FINAL_spec*` and `_spec_extended*`).

Path helpers (`comparison_check_log_utils.py`):

- `resolve_iter_gp_directory(coco, sn, mode)` → runtime `twodim_iter/.../final/full_gp/`
- `resolve_final_directory(..., twodim_branch=final_spectra_twodim_branch(..., use_iter_gp_mangle=True))` → QA FINAL tree

## Smoke checklist (AT2017gfo)

```bash
# After 4.5 → 5 → 6_Iterative (2 iters) with USE_ITER_GP_MANGLE_FINAL in 7.5:
```

Verify:

- [ ] `twodim_iter/.../final/full_gp/*.txt` exists (not only `predictions.npz`)
- [ ] `FINAL_spectra_2dim/twodim_iter/.../as_observed/*_FINAL_spec.txt` exists (if export enabled)
- [ ] `iter_KK/gp_runs/figs/` contains 2D GP reference set (heatmap, spectra, training_coverage, training_residuals)
- [ ] `iter_KK/diagnostics/epoch_*_mangled_vs_gp.pdf` and `phot_band_*.png` present
- [ ] `diagnostics_summary/index.html` links heatmaps, `gp_figs/`, and `full_gp/` spectra

## Tests

```bash
cd Codes
PYTHONPATH=. python -m unittest \
  tests.test_gp_full_spectra_export \
  tests.test_gp_final_spec_export \
  tests.test_iter_gp_warm_start \
  tests.test_pipeline_config_resolve \
  tests.test_iter_gp_mangle_diagnostics \
  tests.test_iterative_gp_mangle \
  tests.test_iter_predictions_export \
  tests.test_iter_gp_style_diagnostics \
  tests.test_iter_gp_compare_diagnostics -v
```
