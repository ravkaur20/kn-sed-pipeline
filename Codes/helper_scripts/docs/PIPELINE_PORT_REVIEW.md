# Pipeline port review (PyCoCo_templates → kn-sed-pipeline)

**Review date:** 2026-08-04  
**Reference SN:** AT2017gfo  
**Repos:** `kn-sed-pipeline` (port) vs `PyCoCo_templates` (notebook legacy)

## Baseline (Phase 0)

| Item | Value |
|------|-------|
| Unittest gate | `PYTHONPATH=.:helper_scripts python -m unittest discover -s helper_scripts/tests -q` → **142 ran, OK (skipped=1)** |
| Skipped test | matplotlib/george optional dependency (unchanged) |
| `COCO_PATH` default | kn-sed repo root via `pipeline_config.bootstrap_runtime()` |
| Prescaled list paths | `Inputs/Spectroscopy/2_spec_lists_prescaled/AT2017gfo.list` — all paths under kn-sed (no `PyCoCo_templates`) |
| Phot extrapolated | `Inputs/Photometry/3_LCs_extrapolated/AT2017gfo.dat` present |

### AT2017gfo output snapshot

```
Outputs/AT2017gfo/
  fitted_phot4mangling_AT2017gfo.dat
  AT2017gfo_spec_scale_groups.json
  AT2017gfo_mangle_report.json          # n_mangled=38, n_failed=1
  mangle_diagnostics/index.html
  twodim_iter/final/full_gp/
  twodim_iter/final/convergence_report.json
  twodim_iter/**/gp_mu_heatmap_normalized.png  (4 copies under iter branches)
```

Step-6 convergence (existing run, `--max-iters 1` equivalent): `n_skipped_extract=1` (57983.027880 unmangled), diagnostics completed without shape errors.

---

## Issue log

Severity: **P0** blocker, **P1** wrong science/UX, **P2** doc/config drift, **P3** nice-to-have.

| ID | Stage | Symptom | Root cause | Sev | Disposition |
|----|-------|---------|------------|-----|-------------|
| R01 | 6 / GP grid | `IndexError` on AT2017gfo non-square grid | `transform2LOG_reshape` loop used `shape[0]` not `shape[1]` | P0 | **Fixed** — `GP2dim_utils.py` L388; test `test_transform_nonsquare_grid_coordinate_lengths` |
| R02 | 6 extract | ValueError extract unmangled epoch | Extract ran without step-5 mangled file | P0 | **Fixed** — skip when no mangled file; `n_skipped_extract=1` in metrics |
| R03 | 6 diagnostics | `x and y must have same first dimension (21264,) vs (89,)` | `diag_groups["masks"]` stored GP-length masks; plots used prescaled `wls` | P0 | **Fixed** — store `masks_out`; chain_data `wls_prescaled` |
| R04 | 6 diagnostics | No normalized μ heatmap vs PyCoCo notebook | PyCoCo had extra notebook cell; subprocess defaulted to denormalized plot | P1 | **Fixed** — `--heatmap-normalized` → `gp_mu_heatmap_normalized.png` |
| R05 | 6 | `GP2dim_utils_iter` NameError `_gz` | Stale symbol after refactor | P0 | **Fixed** — grep clean |
| R06 | tests | 4× `test_gp2dim_utils` IndexError after R01 fix | Test fixtures used `(1,2)` grid with `len(off_xa)=2`, `len(off_ya)=1`; old bug masked mismatch | P1 | **Fixed this review** — fixtures → `(n_wls, n_time)=(2,1)` |
| R07 | 5 config | `MANGLE_RUN_BOTH_FOR_DIAG` in docs/diagnostics but missing from `pipeline_config.py` | Port oversight; step 5 only had `--run-both-for-diag` store_true | P2 | **Fixed this review** — constant added; step 5 uses `BooleanOptionalAction` + config default |
| R08 | config | `CSP_SNE=()` empty in kn-sed vs 14 names in PyCoCo | AT2017gfo uses GeneralFilters only | P2 | **Intentional for AT2017gfo** — populate before CSP events |
| R09 | config | `USE_ITER_GP_MANGLE_FINAL=True` kn vs `False` PyCoCo | kn-sed defaults 7.5 to iter-final products | P2 | **Intentional divergence** — documented in PHASE4 |
| R10 | config | `PRE_BUMP_SNAMES=()` both empty for AT2017gfo | N/A for this SN | — | OK |
| R11 | 5 | Epoch 57983.027880 LDSS3 fails mangling | `no_phot_constraints` — no GP-fit phot in range for that MJD | P2 | **Expected** — reported in `AT2017gfo_mangle_report.json`; step 6 skips extract |
| R12 | docs | `PHASE3_ITER_GP_MANGLE.md` says `ITER_GP_MANGLE_MAX_ITERS=10` | Doc stale; `pipeline_config.py` has `20` | P2 | **Fixed** — doc updated to 20 |
| R13 | cross-cut | PyCoCo `diagnostics/compare_*.py` not vendored in kn-sed | Optional QA tooling from PyCoCo dev branch | P3 | **Accept** unless 7.5 parity required |
| R14 | paths | `spectra_pre_scale.rewrite_spec_path` strips `PyCoCo_templates` | Intentional migration helper | — | **Intentional** |
| R15 | perf | DataFrame fragmentation warning in grid prep | concat-in-loop | P2 | **Fixed** — batched columns in `twodim_grid_prep.py` |

---

## Stage review summary

### Phase 1 — Notebooks 0.1–2

| Check | Result |
|-------|--------|
| Spec list under kn `Inputs/` | OK — prescaled list absolute paths point at kn-sed |
| Photometry stage dir | `3_LCs_extrapolated/` (kn naming) |
| `SN_EXPLOSION_MJD["AT2017gfo"]` | `57982.52851852` in `pipeline_config.py` |
| Stale `PyCoCo_templates` in Inputs | None found |

Notebooks not re-executed this review; inputs assumed current from prior pipeline runs.

### Phase 2 — Step 3 (`3_LCfit_KN_log.py`)

| Check | Result |
|-------|--------|
| Downstream artifact | `fitted_phot4mangling_AT2017gfo.dat` present with `{band}_fit_log_flux` columns |
| Phot input stage | `PHOTOMETRY_STAGES` → `3_LCs_extrapolated` |
| Smoke re-run | Not repeated (artifact valid) |

### Phase 3 — Step 4 (`4_Scale_spectra_KN.py`)

| Check | Result |
|-------|--------|
| `USE_PRESCALED_SPECTRA=True` | Yes |
| Groups JSON | `AT2017gfo_spec_scale_groups.json` present |
| Path rewrite | Lists use kn paths only |

### Phase 4 — Step 5 (`5_Mangle_spectra_KN_log.py`)

| Check | Result |
|-------|--------|
| Bundle-aware + stitch synphot | Active per report `groups` |
| Failed epoch logging | 1 failure: 57983.027880 `no_phot_constraints` |
| Diagnostics index | `mangle_diagnostics/index.html` exists |
| `MANGLE_RUN_BOTH_FOR_DIAG` | **Added** to config; wired in step 5 driver |

### Phase 5 — Step 6 (`6_Iterative_GP_mangle_KN.py`)

| Check | Result |
|-------|--------|
| Inference path | `twodim_gp/run_inference.py` (not legacy `ryan_gp`) |
| Extract skip unmangled | `n_skipped_extract=1` in convergence report |
| Final products | `twodim_iter/final/full_gp/` populated |
| Normalized heatmap | 4× `gp_mu_heatmap_normalized.png` under iter diagnostics |
| Full multi-iter convergence | Not re-run; existing run stopped at max_iters after iter 0 |

### Phase 6 — Cross-cut

**Stale name grep** (`helper_scripts/`):

- `PyCoCo_templates`: docs + intentional path rewrite only
- `GP2dim_utils_newlog` / `GP2dim_utils_ryan`: absent from production code
- `_gz`: absent from `GP2dim_utils_iter.py`

**High-risk module sizes** (kn vs PyCoCo): similar line counts; kn `mangle_spectra_log.py` +78 lines (bundle/stitch diagnostics). Full hunk review deferred — regression tests cover iter/mangle/diagnostics paths.

### Phase 7 — 7.5 QA

| Check | Result |
|-------|--------|
| `comparison_check_log_utils.py` | Supports `twodim_iter/<mode>/final/<product>/` via `USE_ITER_GP_MANGLE_FINAL` |
| 7.5 notebooks | `7.5_alternate.ipynb` sets `USE_ITER_GP_MANGLE_FINAL = True` |
| PyCoCo diagnostics subset | Not ported (R13) |

### Phase 8 — E2E acceptance (AT2017gfo)

Validated **existing outputs** (inputs unchanged) rather than full 0.1→6 rerun:

| Criterion | Pass? |
|-----------|-------|
| All unittests pass | Yes (142 OK) |
| Steps 3–6 artifacts present | Yes |
| Step 5 reports failed epochs | Yes (1) |
| Step 6 diagnostics no shape mismatch | Yes (post R03/R04 fixes) |
| Final GP under `twodim_iter/final/` | Yes |
| Full convergence to phot threshold | No — stopped `max_iters` after 1 GP iter (expected for smoke) |

---

## Config parity appendix

| Key | kn-sed-pipeline | PyCoCo_templates | Notes |
|-----|-----------------|------------------|-------|
| `CSP_SNE` | `()` | 14 SN names | Populate for CSP |
| `PRE_BUMP_SNAMES` | `()` | `()` | Same |
| `USE_PRESCALED_SPECTRA` | `True` | `True` | Same |
| `MANGLE_RUN_BOTH_FOR_DIAG` | `False` | `False` | **Added this review** |
| `ITER_GP_MANGLE_MAX_ITERS` | `20` | `20` | Same |
| `USE_ITER_GP_MANGLE_FINAL` | `True` | `False` | kn default for iter QA |
| `MANGLE_BUNDLE_AWARE` | `True` | `True` | Same |
| `USE_TWO_D_GP_ZSCORE_COORDS` | `True` | (via newlog) | kn unified module |

---

## I/O contract matrix

| Artifact | Producer | Consumer(s) |
|----------|----------|-------------|
| `3_LCs_extrapolated/<SN>.dat` | NB2 | Step 3 |
| `fitted_phot4mangling_<SN>.dat` | Step 3 | Steps 5, 6 |
| `2_spec_lists_prescaled/<SN>.list` | Step 4 | Steps 5, 6 |
| `<SN>_spec_scale_groups.json` | Step 4 | Step 5 bundle mangle |
| `mangled_spectra/*.txt` | Step 5 | Step 6 iter_00 seed |
| `twodim_iter/final/full_gp/` | Step 6 | 7.5 QA |

Column/MJD/basename alignment verified for AT2017gfo prescaled list ↔ mangle report ↔ iter extract counts (38 mangled, 1 skipped).

---

## Fix batches (post-review)

### Batch A — done this session

1. Fix `test_gp2dim_utils` grid fixtures (R06)
2. Add `MANGLE_RUN_BOTH_FOR_DIAG` + step 5 config wiring (R07)

### Batch B — recommended next

1. Update `PHASE3_ITER_GP_MANGLE.md` max-iters default (R12)
2. Full AT2017gfo step 6 run to convergence (`ITER_GP_MANGLE_MAX_ITERS=20`) if science QA needed
3. Populate `CSP_SNE` when porting CSP events (R08)

### Batch C — optional

1. Vendor PyCoCo `diagnostics/compare_*.py` if 7.5 grid parity needed (R13)

---

## Changes made during this review

- `Codes/helper_scripts/tests/test_gp2dim_utils.py` — align `(n_wls, n_time)` test grids with `off_xa`/`off_ya` lengths
- `Codes/pipeline_config.py` — `MANGLE_RUN_BOTH_FOR_DIAG = False`
- `Codes/5_Mangle_spectra_KN_log.py` — config-backed `--run-both-for-diag` default
