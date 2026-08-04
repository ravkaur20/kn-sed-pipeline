# 2D GP tooling (vendored)

Python utilities for 2-D GP bundles used by **NB6 iterative mangling** (`GP2dim_utils_iter`, `gp2dim_export`, GP diagnostic plots). Kept under **`Codes/helper_scripts/twodim_gp/`** in kn-sed-pipeline.

## Production runtime modules (13)

These files are required for the production NB6 path and diagnostics:

| Module | Role |
|--------|------|
| `run_inference.py` | Fit/predict from minimal export bundles (imported as `twodim_gp.run_inference`) |
| `gp_utils.py` | George kernels, bundle I/O, coordinate helpers |
| `bundle_meta.py` | Load/write `*_meta.json` beside bundles |
| `spec_bundle_id_assign.py` | Assign `spec_bundle_id` / `train_obs_class` on export |
| **`spectrum_bundles.py`** | Time clustering + λ composites (required when `GP_EXPORT_SPEC_BUNDLE_IDS=True`) |
| `gp_grid_interp.py` | Grid interpolation for plots |
| `filter_synthesis.py` | Optional pysynphot filter synthesis |
| `plot_results.py` | Heatmaps / training coverage (NB6 `ITER_GP_DIAG_FIGS`) |
| `plot_bands_gp_overview.py` | Per-band GP overview panels |
| **`bundle_preprocess.py`** | Telluric / train_obs_class edits (imported by `plot_results`) |
| **`bundle_scale_pipeline.py`** | Intra-bundle scaling helpers (imported by `plot_bands_gp_overview`) |
| `strip_photometry_bands.py` | Drop photometry rows by band |
| `__init__.py` | Package exports |

**NB6 note:** With default `GP_EXPORT_SPEC_BUNDLE_IDS = True`, `gp2dim_export` calls `spec_bundle_id_assign`, which imports `spectrum_bundles`. Do not omit that file.

Optional CLI / comparison scripts (`iterate_gp_surface_bundle_scale.py`, `run_full_pipeline.py`, etc.) are **not** vendored in kn-sed-pipeline.

## Layout and `PYTHONPATH`

Flat imports (`import gp_utils`, `import spectrum_bundles`, …) assume **cwd = `Codes/helper_scripts/twodim_gp/`** when running scripts as subprocesses (see `iter_gp_style_diagnostics.py`).

From notebook / library code under `Codes/`:

```bash
export PYTHONPATH="/path/to/kn-sed-pipeline/Codes:/path/to/kn-sed-pipeline/Codes/helper_scripts:${PYTHONPATH}"
python -c "from twodim_gp.plot_results import _make_heatmap"
```

Subprocess drivers:

```bash
cd /path/to/kn-sed-pipeline/Codes/helper_scripts/twodim_gp
python plot_results.py --help
python plot_bands_gp_overview.py --help
```

## Notebook integration

- **NB6** (`6_Iterative_GP_mangle_KN.ipynb`): `run_iterative_gp_mangle` → 2D GP fit → optional `ITER_GP_DIAG_FIGS` subprocess plots.
- Minimal export written to `gp_runs/gp_minimal_export/gp_minimal_bundle.npz`.

Provenance: synced from `PyCoCo_templates/Codes/twodim_gp/`; refresh the production module set above when updating from upstream.
