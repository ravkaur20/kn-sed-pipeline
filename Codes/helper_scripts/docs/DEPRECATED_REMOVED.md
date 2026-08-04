# Deprecated / removed modules log

| Date | Module | What it did | Replacement | Notes |
|------|--------|-------------|-------------|-------|
| 2026-08-04 | `iter_gp_mangle_diagnostics._save_sed_chain` | 2×2 SED chain PDF per epoch | `iter_plot_suite.plot_gp_vs_mangled` | Removed from orchestrator |
| 2026-08-04 | `iter_gp_mangle_diagnostics._save_mask_compare` | Old vs new mask PDF | `iter_plot_suite.plot_mangle_delta` | Removed from orchestrator |
| 2026-08-04 | `iter_gp_mangle_diagnostics._save_gp_heatmaps` | Fallback heatmaps at diagnostics root | `iter_gp_style_diagnostics` → `figs/gp_surface/` | Only when Ryan subprocess fails |
| 2026-08-04 | `iter_gp_mangle_diagnostics.save_iter_mangle_group_diagnostics` | Prescaled vs mangled iter group QA | `iter_plot_suite` GP vs mangled group plots | Step 5 still uses `mangle_diagnostics` |
| 2026-08-04 | `iter_gp_compare_diagnostics` (logic) | Mangled vs GP + synphot LC plots | `iter_plot_suite.py` | File kept as thin wrapper for tests |
| 2026-08-04 | Output path `iter_KK/diagnostics/` | Mixed QA + duplicate GP figs | `iter_KK/figs/{gp_surface,gp_vs_mangled,...}` | Single plot home |
| 2026-08-04 | Output path `gp_runs/figs/` (retained) | Staging for plot_results | Moved to `figs/gp_surface/` after subprocess | `gp_runs/figs` deleted post-move |
| 2026-08-04 | Output path `gp_runs/diagnostics/` | GP inference cache | `gp_runs/inference/` | Warm-start reads `inference/gp_inference_config.json` |
| 2026-08-04 | Output path `Outputs/<SN>/mangle_diagnostics/` | NB5 prescaled QA | `Outputs/<SN>/figs/mangle_nb5/` | Via `mangle_diagnostics_dir()` |

## Still in use (not deprecated)

- `mangle_diagnostics.py` — step 5 prescaled mangling QA
- `mangle_epoch_plots.py` — step 5 epoch plots (+ `save_mangle_epoch_plot_bundle`)
- `iter_gp_style_diagnostics.py` — subprocess wrapper for Ryan `plot_results` / bands overview
- `iter_gp_mangle_diagnostics.py` — slim orchestrator + `write_diagnostics_summary`
- `gp2dim_phase_merge.py`, `rimangle_log_spectrum.py`, `lc_gp_diagnostics.py`
