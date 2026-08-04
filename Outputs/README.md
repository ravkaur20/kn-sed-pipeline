# Outputs

Pipeline products are written under `Outputs/<SN>/`. This repository ships with an empty scaffold only; products appear after you run the preparation notebooks and the production scripts.

## Expected layout after a full AT2017gfo run

| Stage | Driver / notebook | Key outputs |
|-------|-------------------|-------------|
| 3 | `Codes/3_LCfit_KN_log.py` | `fitted_phot4mangling_AT2017gfo.dat`, `fitted_phot_logspace_AT2017gfo.dat`, optional `fittedGP_*.pdf` |
| 4 | `Codes/4_Scale_spectra_KN.py` | `AT2017gfo_spec_scale_groups.json`, `AT2017gfo_spec_scale_report.json`, `spec_scale_diagnostics/` (prescaled spectra live under `Inputs/Spectroscopy/2_spec_prescaled/`) |
| 5 | `Codes/5_Mangle_spectra_KN_log.py` | `mangled_spectra/`, `AT2017gfo_band_mjd_ranges.json`, `figs/mangle_nb5/`, `figs/mangle_epoch/` |
| 6 | `Codes/6_Iterative_GP_mangle_KN.py` | `twodim_iter/iter_*/`, `twodim_iter/final/full_gp/`, `twodim_iter/metrics/`, `twodim_iter/diagnostics_summary/` |
| 7.5 | QA notebooks (`7.5_*.ipynb`, …) | `FINAL_spectra_2dim/as_observed/` (when export is enabled), comparison and spectrum plots |

**Band MJD windows.** `<SN>_band_mjd_ranges.json` is auto-generated from `Inputs/Photometry/1_LCs_flux_raw/<SN>.dat` when step 5 runs. You do not need legacy repository-root `AT2017gfo_mjd_ranges*.csv/json` files.

**Reset.** From `Codes/`, run `python clear_sn_run.py --snname AT2017gfo --yes` to empty this directory. See the [user guide](../docs/PIPELINE_USER_GUIDE.md) for details.

Re-run scripts in order from `Codes/` after setting `COCO_PATH` to the repository root. Full instructions: [README](../README.md) and [user guide](../docs/PIPELINE_USER_GUIDE.md).
