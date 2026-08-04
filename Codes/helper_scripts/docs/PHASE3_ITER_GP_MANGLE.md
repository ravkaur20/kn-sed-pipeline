# Phase 3 — Iterative GP + re-mangle loop (completed)

Phase 3 replaces the legacy NB6+NB7 legacy remangle path with an outer loop:

**GP surface → extract at observed epochs → demangle → NB5-style re-mangle → refit**

 `bundle_scale_pipeline` / `iterate_gp_surface_bundle_scale` are **not** used.

## GP2dim module map

| Module | Role |
|--------|------|
| [`GP2dim_utils.py`](../GP2dim_utils.py) | Grid prep + coordinate norm (min-max or z-score via `USE_TWO_D_GP_ZSCORE_COORDS`) |
| [`GP2dim_utils_iter.py`](../GP2dim_utils_iter.py) | Per-iteration GP fit orchestrator → `twodim_gp/run_inference.py` |
| [`gp2dim_export.py`](../gp2dim_export.py) | Minimal bundle export for inference |
| [`gp2dim_phase_merge.py`](../gp2dim_phase_merge.py) | Dense log-phase column merge on prediction grid |
| [`twodim_gp/`](../twodim_gp/) | Vendored 2D GP inference engine + diagnostic plots |

**Data-flow reference (physical inputs, synphot, mangling targets):** [`ITERATIVE_MANGLING_DATA_FLOW.md`](ITERATIVE_MANGLING_DATA_FLOW.md)

## Deliverables

| Module | Path |
|--------|------|
| GP surface extract | [`gp_surface_extract.py`](../gp_surface_extract.py) |
| Per-iteration GP fit | [`iter_gp_grid.py`](../iter_gp_grid.py) |
| Outer driver | [`iterative_gp_mangle.py`](../iterative_gp_mangle.py) — `run_iterative_gp_mangle` |
| Iteration diagnostics | [`iter_gp_mangle_diagnostics.py`](../iter_gp_mangle_diagnostics.py) |
| Production notebook | [`6_Iterative_GP_mangle_KN.ipynb`](../6_Iterative_GP_mangle_KN.ipynb) |
| Config / paths | [`pipeline_config.py`](../pipeline_config.py) — `ITER_*`, `twodim_iter_*` |
| Tests | `tests/test_gp_surface_extract.py`, `test_iterative_gp_mangle.py`, `test_iter_gp_mangle_diagnostics.py` |

Legacy comparison notebooks from PyCoCo_templates are not included in this repo.

## Prerequisites

Run in order: **0.1 → 1 → 2 → 4 → 4.5 → 5**, then this notebook.

Required under `Outputs/<SN>/`:

- `mangled_spectra/` (NB5 first pass)
- `fitted_phot4mangling_<SN>.dat`
- `fitted_phot_logspace_<SN>.dat`
- Prescaled list via `Inputs/Spectroscopy/2_spec_lists_prescaled/<SN>.list`

## Output layout

```
Outputs/<SN>/twodim_iter/
  iter_KK/
    figs/
      gp_surface/          # Ryan heatmaps, slices, training QA
      gp_vs_mangled/       # input mangled vs GP extract
      mangle_delta/        # remangle vs input
      residuals/
      phot_lc/
    gp_runs/
      inference/           # GP config cache (not plots)
      predictions.npz
      full_gp/
  metrics/                 # chi2, lengthscales vs iteration
  diagnostics_summary/
  final/
  iter_00/mangled_spectra/     # seeded from NB5
  iter_KK/
    mangled_spectra/
    gp_runs/predictions.npz    # written after each GP fit (output, not input)
    extracted/  demangled/
    diagnostics/
    metrics.json
  iteration_log.jsonl
  diagnostics_summary/
  final/full_gp/predictions.npz
  final/full_gp/{Log_Phase}_spec_extended*.txt   # pure GP (ITER_GP_EXPORT_FULL_GP)
  final/mangled_spectra/
  final/convergence_report.json
```

Optional QA export (`ITER_EXPORT_FINAL_SPEC_FOR_QA`, default on):

```
Outputs/<SN>/FINAL_spectra_2dim/as_observed/{stem}_FINAL_spec.txt
```

`predictions.npz` is **per-iteration GP output** (μ, std, X_fill, grid metadata). It is not read from  vendor tree to start a fit.

## Convergence

Primary (default): max relative synphot error after applying new mask to prescaled originals

- `PHOT_CONVERGENCE_FRAC = 0.05` (configurable)
- Stop also at `ITER_GP_MANGLE_MAX_ITERS = 20`

Secondary logging: RMS Δmask (`MASK_CONVERGENCE_EPS`).

## Key config (`pipeline_config.py`)

| Knob | Default | Role |
|------|---------|------|
| `ITER_GP_MANGLE_MAX_ITERS` | 20 | Outer loop cap |
| `PHOT_CONVERGENCE_FRAC` | 0.05 | Photometry closure tolerance |
| `ITER_GP_SEED_FROM_NB5` | True | Copy NB5 mangled → `iter_00` |
| `MANGLE_BUNDLE_AWARE` | True | Shared mask per 4.5 scale group in remangle |
| `MANGLE_BUNDLE_STITCH_SYNPHOT` | True | Multi-arm stitched synphot (NB5 + iter bundle; see data-flow doc) |
| `ITER_MANGLE_USE_GP_WAVELENGTH_GRID` | True | Extract/demangle on full GP λ grid (iter only) |
| `GP_PREDICT_MU_KEY` | `"mu"` | Surface used for extract (post mono+blue) |
| `ITER_SAVE_DIAGNOSTICS` | True | Saved PDFs/PNG under each `iter_KK/diagnostics/` |
| `ITER_GP_EXPORT_FULL_GP` | True | Write `gp_runs/full_gp/*_spec_extended*.txt` each iter |
| `ITER_GP_WARM_START` | False | Opt-in warm start from previous iter `gp_inference_config.json` |
| `ITER_GP_DENSE_PREDICT_GRID` | True | Log-uniform phase columns on X_fill (`merge_extrap_mjds_dense_log_phase`) |
| `ITER_GP_DENSE_PREDICT_GRID_N` | 100 | Number of log-spaced phase samples merged into prediction grid |
| `ITER_GP_DIAG_FIGS` | True | Subprocess 2D GP `plot_results` + `plot_bands_gp_overview` each iter |
| `ITER_EXPORT_FINAL_SPEC_FOR_QA` | True | Copy to `FINAL_spectra_2dim/twodim_iter/...` for 7.5 |
| `DIAG_COPY_GP_FIGS` | False | Legacy alias; plots go to `iter_KK/figs/gp_surface/` |
| `DIAG_WAVELENGTH_SLICES` | True | `gp_results_wavelength_slices.pdf` per iter |
| `DIAG_TRAINING_GRID_PLOT` | True | `data_for2d_interpolation.pdf` per iter |
| `MANGLE_PHOTOMETRY_TARGET` | `"gp_fit"` | NB4 GP-fitted photometry only (`fitted_phot4mangling`) |
| `MANGLE_GP_KERNEL_MODE` | `"fixed_5"` | Step 5 + iter remangle share `mangle_spectra_log` kernel (`fixed_5` = PyCoCo `Matern32Kernel(5.0)`) |
| `MANGLE_GP_KERNEL_FIXED` | `5.0` | Fixed lengthscale when mode is `fixed_5` |
| `MANGLE_KERNEL_DIVIDE` | `800` | Only when mode is `kernel_divide_scaled` |
| `PLOT_GRID_REBIN` | False | NB6-style rebin overlay PDF during grid build (`grid_rebin_<SN>.pdf` under `gp_runs/`) |
| `CSP_SNE` | tuple of CSP SN names | Site3_CSP filter curves for listed events (replaces notebook `CSP_SNe` global) |

## Troubleshooting

- **`NameError: name 'plt' is not defined`** during grid build: update `twodim_grid_prep.py` (plotting is gated behind `plot_grid_rebin`, default off) or ensure `PLOT_GRID_REBIN = False` in `pipeline_config.py`.
- **`NameError: name 'CSP_SNe' is not defined`** during grid extrapolation: update `twodim_grid_prep.py` and ensure `CSP_SNE` is set in `pipeline_config.py`; library runs must not rely on notebook globals.

## Diagnostics (per iteration)

When `ITER_SAVE_DIAGNOSTICS=True` (default), each `iter_KK/diagnostics/` includes:

**GP diagnostic suite** (`gp_figs/plot_results/`, from `twodim_gp/plot_results.py`):

- `gp_mu_heatmap.png`, `gp_mu_heatmap_raw.png`, `gp_std_heatmap.png`
- `gp_mu_phase_profiles.png`, `gp_spectra.png`
- `gp_results_wavelength_slices.pdf` (+ linear-phase variant)
- `training_coverage.png`, `training_residuals.png`

**Band / bundle overlays** (`gp_figs/plot_bands_overview/`):

- `phot_band_*.png`, `spec_bundle_*_gp.png` (with phot enrich from `fitted_phot4mangling`)

**Iter comparisons**:

- `epoch_*_mangled_vs_gp.pdf` — mangled input vs GP extraction + residual
- `phot_band_*.png` — synphot from mangled vs NB4 `gp_fit` photometry
- `epoch_*_sed_chain.pdf`, `epoch_*_mangle_mask.pdf`

Summary index: `diagnostics_summary/index.html`.

## Tests

From `Codes/` with conda `SED_clean`:

```bash
cd Codes
/opt/anaconda3/envs/SED_clean/bin/python -m unittest \
  tests.test_twodim_grid_prep \
  tests.test_gp_surface_extract \
  tests.test_iterative_gp_mangle \
  tests.test_iter_gp_mangle_diagnostics \
  tests.test_iter_predictions_export \
  tests.test_iter_gp_style_diagnostics \
  tests.test_iter_gp_compare_diagnostics \
  tests.test_pipeline_config_resolve -v
```

Full GP smoke (slow): run notebook with `ITER_GP_MANGLE_MAX_ITERS = 2` on AT2017gfo.

## Next (Phase 4)

See [`docs/PHASE4_DOWNSTREAM_QA.md`](PHASE4_DOWNSTREAM_QA.md): `USE_ITER_GP_MANGLE_FINAL`, dual `_spec_extended` / `_FINAL_spec` QA paths, 7.5 notebook toggles.
