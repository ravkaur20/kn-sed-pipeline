---
title: "Kilonova SED Pipeline --- Scientific Writeup"
subtitle: "Equations and rationale for every stage of kn-sed-pipeline"
date: "August 2026"
geometry: "margin=0.9in"
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{amsmath}
  - \renewcommand{\arraystretch}{1.15}
---

# 1. Overview

## 1.1 Goal

This pipeline reconstructs a time-evolving spectral energy distribution (SED) for a transient that has dense multi-band photometry and sparse spectroscopy. The scientific product is a continuous surface in wavelength and phase,

$$
\log_{10} F_\lambda(\lambda,\, t),
$$

together with wavelength-dependent mangling masks that force synthetic photometry of the observed spectra to agree with a Gaussian Process (GP) fit to the light curves.

The default worked example is AT2017gfo (GW170817). The software repository is `kn-sed-pipeline`. Operational instructions live in `README.md` and `docs/PIPELINE_USER_GUIDE.md`. This document explains **what each stage computes** and **why**.

## 1.2 Coordinate conventions

Throughout the pipeline:

- Wavelength $\lambda$ is in angstroms (Å) in the observer frame unless stated otherwise.
- Time is Modified Julian Date (MJD). Phase in days since explosion (or merger) is
  $$
  t = \mathrm{MJD} - t_0,
  $$
  with $t_0$ stored in `pipeline_config.SN_EXPLOSION_MJD`.
- Log phase used by the two-dimensional GP is
  $$
  \tau = \log_{10}\!\big(\max(t,\, t_{\min})\big),
  $$
  with a small positive floor $t_{\min}$ (typically $10^{-5}\,\mathrm{d}$) to avoid singularities.
- Fluxes entering one-dimensional and two-dimensional GPs are almost always $\log_{10} F$ (or a scaled natural-log transform of $F_\lambda$ inside the two-dimensional engine). Mangling masks are additive in $\log_{10} F$.

## 1.3 Stage map

```
Inputs (photometry + spectroscopy + filters)
    → 0.1 Spectral smoothing
    → 1   Dust correction (photometry)
    → 2   Early/late light-curve extrapolation (Bazin)
    → 3   Per-band one-dimensional GP light-curve fit
    → 4   Spectroscopic arm prescaling
    → 5   Log-space mangling (first pass)
    → 6   Iterative two-dimensional GP + remangle
    → 7.5 Downstream QA / bolometric / plots
```

Preparation notebooks are `0.1`, `1`, and `2`. Production drivers are `3_LCfit_KN_log.py`, `4_Scale_spectra_KN.py`, `5_Mangle_spectra_KN_log.py`, and `6_Iterative_GP_mangle_KN.py`.

\newpage

# 2. Spectral smoothing (notebook 0.1)

## 2.1 Motivation

Raw spectra are noisy on pixel scales that are irrelevant for broadband synthetic photometry and for a smooth two-dimensional SED surface. Smoothing reduces that noise while preserving the continuum and broad features that dominate filter integrals.

## 2.2 Procedure

Notebook `0.1_Smooth_spectra_KN.ipynb` reads each spectrum listed under `Inputs/Spectroscopy/1_spec_lists_original/`, applies a configured smoother (typically a running median, Gaussian convolution, or related local average on the native wavelength grid), and writes:

- smoothed files under `Inputs/Spectroscopy/2_spec_smoothed/<SN>/`,
- an updated list under `Inputs/Spectroscopy/2_spec_lists_smoothed/<SN>.list`.

No change is made to the absolute flux zero-point at this stage beyond whatever the smoother does to local averages. Arm-to-arm flux alignment is deferred to step 4.

If spectra are already smoothed, this notebook may be skipped and files placed directly in the smoothed folders.

\newpage

# 3. Dust correction (notebook 1)

## 3.1 Motivation

Observed fluxes are attenuated by interstellar dust along the line of sight. Correcting to an extinction-free flux puts all bands on a more physical footing before light-curve fitting and mangling.

## 3.2 Extinction law

Notebook `1_LC_DustCorrection_KN.ipynb` uses the `extinction` package (and coordinates from `Inputs/SNe_Info/info.dat`) to evaluate selective extinction $A_\lambda$ for each band’s effective wavelength. For a Cardelli–Clayton–Mathis-like law with total-to-selective ratio $R_V$ and colour excess $E(B-V)$,

$$
A_\lambda = R_V\, E(B-V)\, f\!\left(\frac{1}{\lambda};\, R_V\right),
$$

where $f$ is the normalized extinction curve. The flux correction is

$$
F_{\mathrm{corr}} = F_{\mathrm{obs}} \times 10^{0.4\, A_\lambda}.
$$

Magnitude corrections follow $\Delta m = -A_\lambda$ when magnitudes are used as intermediates.

## 3.3 Output

Dust-corrected light curves are written to `Inputs/Photometry/2_LCs_dust_corrected/<SN>.dat`. Host extinction may be applied in addition to Milky Way extinction when configured for a given object.

\newpage

# 4. Light-curve extrapolation (notebook 2)

## 4.1 Motivation

Kilonova light curves are often poorly sampled before peak and may need controlled extrapolation so that later GP fits and two-dimensional surfaces do not invent unphysical early flux. Notebook `2_LC_modelRising_KN_fullfit_log.ipynb` fits analytic rising models and commits an explosion (merger) time $t_0$.

## 4.2 Bazin functional form

Following Bazin et al. (2011; arXiv:1109.0948), a useful parametric light curve is

$$
F(t) = A\, \frac{e^{-(t-t_{\mathrm{peak}})/t_{\mathrm{fall}}}}{1 + e^{-(t-t_{\mathrm{peak}})/t_{\mathrm{rise}}}} + C.
$$

In code (`lc_bazin_models.bazin_func`) the same shape appears as

$$
F(x) = a\, \frac{\exp\!\big(-(x-t_0)/t_{\mathrm{fall}}\big)}{1 + \exp\!\big(-(x-t_0)/t_{\mathrm{rise}}\big)} + c.
$$

A forced-zero variant subtracts the model value at the explosion time so that $F(t_{\mathrm{exp}}) = 0$:

$$
F_{\mathrm{forced}}(x) = F_{\mathrm{Bazin}}(x) - F_{\mathrm{Bazin}}(t_{\mathrm{exp}}).
$$

A simpler power-law rise $F \propto (t-t_0)^{3/2}$ for $t > t_0$ is also available for early previews (`rise_func`).

## 4.3 Output and $t_0$

The notebook writes extrapolated photometry to `Inputs/Photometry/3_LCs_extrapolated/<SN>.dat`, including $\log_{10}$ flux columns used by step 3. The committed $t_0$ must match `SN_EXPLOSION_MJD` in `pipeline_config.py`, because every later phase definition depends on it.

\newpage

# 5. Per-band light-curve Gaussian Process (step 3)

## 5.1 Motivation

Mangling and the two-dimensional SED need a **smooth, continuous** estimate of each band’s flux at arbitrary times—especially at spectroscopy epochs that may not coincide with photometric observations. A one-dimensional GP in log flux provides that estimate with uncertainties.

## 5.2 Model

For each filter $b$, training points are $\big(t_i,\, y_i,\, \sigma_i\big)$ with

$$
y_i = \log_{10} F_{b}(t_i).
$$

A Matérn $3/2$ kernel (via `george`) is the default family, possibly with band-specific length scales from a kernel-settings JSON:

$$
k(t, t') = \sigma_k^2\, k_{3/2}\!\left(\frac{|t-t'|}{\ell_b}\right).
$$

Hyperparameters are optimized by maximizing the marginal likelihood (or equivalently minimizing $-\ln\mathcal{L}$). Optional explosion anchors (`ANCHOR_T0_IN_LC_GP`) encourage the early light curve to approach a low flux near $t_0$.

## 5.3 Products

Two tables are written under `Outputs/<SN>/`:

\begin{tabular}{lp{10cm}}
\toprule
File & Role \\
\midrule
\texttt{fitted\_phot4mangling\_<SN>.dat} &
GP mean and error in $\log_{10}F$ evaluated at each spectroscopy MJD, plus in-range flags per band. This is the **mangling target**. \\
\texttt{fitted\_phot\_logspace\_<SN>.dat} &
GP light curve on a dense log-phase grid. This places photometry onto the **two-dimensional GP training grid**. \\
\bottomrule
\end{tabular}

Driver: `3_LCfit_KN_log.py`. Library: `helper_scripts/lc_gp_fit.py`.

\newpage

# 6. Spectroscopic prescaling (step 4)

## 6.1 Motivation

Instruments such as VLT/XSHOOTER deliver separate UVB, VIS, and NIR arms. Absolute calibrations can disagree where arms overlap or nearly abut. Mangling should not be forced to absorb those discontinuities as astrophysical SED features. Prescaling applies **multiplicative** corrections so arms share a common flux scale before the log-space mangling mask is fit.

## 6.2 Scale factors

Within a scale group (near-simultaneous arms listed in `<SN>_spec_scale_groups.json`), a reference arm is chosen (often the highest signal-to-noise arm, typically VIS). For a non-reference arm, on the wavelength overlap region,

$$
s = \mathrm{median}\left(\frac{F_{\mathrm{ref}}(\lambda)}{F_{\mathrm{arm}}(\lambda)}\right),
\qquad
F_{\mathrm{arm}}^{\mathrm{(scaled)}} = s\, F_{\mathrm{arm}}.
$$

When overlap is missing but the gap between arms is smaller than a configured maximum (default 400 angstroms), a **gap-seam** estimate uses median fluxes in edge windows of half-width about 50 angstroms (or a fixed number of edge pixels) to form an analogous ratio.

**Star mode** (default) scales every arm toward the anchor. **Chain mode** scales sequentially along `merge_order`. A **star-bridge fallback** scales a non-adjacent arm through an already-aligned intermediate arm when a direct overlap with the anchor is unavailable.

## 6.3 Output modes

- `scale_only` (default): write separate scaled files under `Inputs/Spectroscopy/2_spec_prescaled/<SN>/`.
- `merge_join`: after scaling, concatenate arms into one spectrum per group (optional linear bridge across small gaps).

Driver: `4_Scale_spectra_KN.py`. Library: `helper_scripts/spectra_pre_scale.py`.

Ungrouped spectra (for example single Magellan/LDSS3 exposures) keep scale factor $1$ and proceed to mangling as individuals.

\newpage

# 7. Log-space mangling (step 5)

## 7.1 Motivation

Even after dust correction and arm scaling, a spectrum’s synthetic photometry generally does not match the light-curve GP at the same epoch. Mangling multiplies the spectrum by a smooth wavelength-dependent factor so that broadband fluxes agree with photometry **without** destroying spectral features on scales smaller than the filter widths.

## 7.2 Synthetic photometry

For a linear-$F_\lambda$ spectrum and filter transmission $T(\lambda)$, trapezoidal integration (`band_flux_trapz`) computes a band-average flux of the form

$$
F_{\mathrm{band}}
=
\frac{\displaystyle\int T(\lambda)\, \lambda\, F_\lambda(\lambda)\, d\lambda}
{\displaystyle\int T(\lambda)\, \lambda\, d\lambda},
$$

with uncertainty propagated from $F_\lambda$ errors through the same quadrature. Effective wavelength $\lambda_{\mathrm{eff}}$ is used to place the constraint on the log-wavelength axis.

Only bands whose validity window (from `<SN>_band_mjd_ranges.json`) contains the spectrum MJD, and whose throughput overlaps the spectrum, contribute constraints.

## 7.3 Log offsets and the mangling GP

For each usable band $i$ at the epoch,

$$
\Delta_i
=
\log_{10} F^{\mathrm{GP}}_i
-
\log_{10} F^{\mathrm{syn}}_i,
$$

where $F^{\mathrm{GP}}_i$ comes from `fitted_phot4mangling` and $F^{\mathrm{syn}}_i$ is synthetic photometry of the **prescaled** spectrum. Uncertainties combine in quadrature in log space:

$$
\sigma_{\Delta,i}^2
=
\sigma_{\log F^{\mathrm{GP}},i}^2
+
\sigma_{\log F^{\mathrm{syn}},i}^2.
$$

A one-dimensional GP with a Matérn $3/2$ kernel is then fit to $\Delta_i$ versus $\log_{10}\lambda_{\mathrm{eff}}$ (after centering). The posterior mean $m(\lambda)$ evaluated on the spectrum’s wavelength grid is the **mangling mask**. The mangled spectrum is

$$
\log_{10} F_{\mathrm{man}}(\lambda)
=
\log_{10} F_{\mathrm{pre}}(\lambda)
+
m(\lambda),
$$

or equivalently $F_{\mathrm{man}} = F_{\mathrm{pre}} \times 10^{m(\lambda)}$ in linear flux.

## 7.4 Kernel lengthscale modes

Two modes are supported via `MANGLE_GP_KERNEL_MODE`:

\begin{tabular}{llp{7.5cm}}
\toprule
Mode & Lengthscale & Notes \\
\midrule
\texttt{fixed\_5} (default) & $\ell = 5$ in normalized $\log_{10}\lambda$ & Matches historical PyCoCo mangling; preferred for production. \\
\texttt{kernel\_divide\_scaled} & $\ell = K / \mathrm{median}(\log_{10}\lambda)$ & Experimental; can produce overly flexible masks if $K$ is poorly chosen. \\
\bottomrule
\end{tabular}

## 7.5 Bundle-aware mangling

When `MANGLE_BUNDLE_AWARE=True`, all arms in a scale group share one mask. Synthetic photometry for a filter may be computed on a **stitched** multi-arm spectrum (`MANGLE_BUNDLE_STITCH_SYNPHOT`) so that filters spanning arm boundaries are integrated consistently. The shared mask is then applied to each member’s native wavelength grid.

Driver: `5_Mangle_spectra_KN_log.py`. Library: `helper_scripts/mangle_spectra_log.py`.

\newpage

# 8. Two-dimensional Gaussian Process (step 6, inner fit)

## 8.1 Motivation

After mangling, each spectrum is a one-dimensional slice of the SED at a discrete phase. A two-dimensional GP interpolates (and carefully extrapolates) in the $(\log_{10}\lambda,\, \tau)$ plane so that the SED is defined at all wavelengths and phases needed for templates, including phases without spectroscopy.

## 8.2 Training data

Each iteration trains on:

- **Mangled spectroscopy** from the current iteration (dense in wavelength, sparse in time),
- **Photometry** from `fitted_phot_logspace`, placed on an extended log-phase grid (observed phases, early extrapolation columns, gap fills, late extension).

Coordinates are normalized (min–max or z-score via `USE_TWO_D_GP_ZSCORE_COORDS`) before fitting. The latent target inside the vendored engine is a scaled $\ln F_\lambda$; exports convert back to $\log_{10} F_\lambda$ or linear $F_\lambda$ as needed.

## 8.3 Kernel structure

The production configuration follows the collaborator “v5” design (see the vendored `twodim_gp` package). On each axis (wavelength $w$ and time $\tau$), an **additive** Matérn $5/2$ kernel captures short and long correlation lengths:

$$
k_{\mathrm{axis}}(\Delta x)
=
w_{\mathrm{short}}\, k_{5/2}(\Delta x;\, \ell_{\mathrm{short}})
+
(1 - w_{\mathrm{short}})\, k_{5/2}(\Delta x;\, \ell_{\mathrm{long}}).
$$

The full covariance is the product of the two axes, scaled by an overall amplitude:

$$
k\big((w,\tau),(w',\tau')\big)
=
A\; k_w(w-w')\; k_\tau(\tau-\tau').
$$

In `george`, metrics are stored as $\ell^2$.

## 8.4 Per-class jitter

Photometry and spectroscopy have different residual floors. The effective observation variance is

$$
\sigma_{\mathrm{eff}}^2(i)
=
\sigma_y^2(i)
+
\begin{cases}
\sigma_{\mathrm{phot}}^2 & i \in \mathrm{phot}, \\
\sigma_{\mathrm{spec}}^2 & i \in \mathrm{spec},
\end{cases}
$$

with lower bounds (defaults) $\sigma_{\mathrm{phot}} \ge 0.012$ and $\sigma_{\mathrm{spec}} \ge 0.005$. Without floors, the optimizer can drive short lengthscales to zigzag through near-simultaneous spectral scatter.

Point classification: a unique training phase with at least `phot_spec_threshold` (default 50) distinct wavelengths is labelled spectroscopy; otherwise photometry.

## 8.5 Mean function and optimization

A linear interpolant (`LinearNDInterpolator`) over prior points, with nearest-neighbour fallback outside the convex hull, provides the GP mean. Hyperparameters may be optimized with L-BFGS-B on a stratified subsample, then a final `gp.compute` on the full training set. Iteration warm-start reuses the previous iteration’s hyperparameters when enabled.

## 8.6 Early-time post-processing

For normalized log-phase below a cutoff (default $\tau < -4$), two post-processing constraints are applied to the posterior mean on the prediction grid:

1. **Monotonic early rise (tanh blend).** A linear extrapolation in time is matched in value and slope at the cutoff and blended with the raw GP mean using
   $$
   w(\tau) = \tfrac12\Big(1 + \tanh\big((\tau - \tau_{\mathrm{cut}})/s\big)\Big),
   $$
   so the join is $C^\infty$.
2. **Blue early spectrum.** At early phases, enforce non-increasing flux toward longer wavelengths via a cumulative minimum along wavelength (hot, blue continuum prior)—not a full blackbody.

These corrections define the exported surface key `mu` (as opposed to raw `mu_raw`).

\newpage

# 9. Iterative remangle loop (step 6, outer loop)

## 9.1 Motivation

A single mangling pass (step 5) uses only the native spectrum. Once a two-dimensional GP exists, a better estimate of the **unmangled** SED at each epoch is available across a wider wavelength range. Feeding that estimate back into mangling improves consistency between the surface and the photometry constraints. The outer loop repeats until synthetic photometry of **prescaled data plus the new mask** matches the light-curve GP within tolerance.

## 9.2 Loop structure

Driver: `run_iterative_gp_mangle` in `helper_scripts/iterative_gp_mangle.py`.

For iteration $K = 0, 1, \ldots$:

1. **Fit** the two-dimensional GP on `iter_K/mangled_spectra` (seeded from step 5 mangled spectra when $K=0$).
2. **Extract** the GP mean at each observed spectroscopy epoch. With default `ITER_MANGLE_USE_GP_WAVELENGTH_GRID=True`, extraction is sampled onto the GP’s log-uniform wavelength grid spanning the full training $\lambda$ range.
3. **Demangle.** If $m_{\mathrm{old}}(\lambda)$ is the mask currently associated with the mangled training spectrum,
   $$
   \log_{10} F_{\mathrm{dem}}(\lambda)
   =
   \log_{10} F_{\mathrm{GP,\,man}}(\lambda)
   -
   m_{\mathrm{old}}(\lambda),
   $$
   after interpolating the mask onto the extraction grid. This recovers a GP estimate of the spectrum **before** mangling.
4. **Remangle.** Recompute a new mask $m_{\mathrm{new}}$ by comparing synthetic photometry of the demangled GP SED to `fitted_phot4mangling` (bundle-aware when enabled). Apply $m_{\mathrm{new}}$ to the **original prescaled** spectra (not to the GP extract):
   $$
   \log_{10} F_{\mathrm{man}}^{(K+1)}
   =
   \log_{10} F_{\mathrm{pre}}
   +
   m_{\mathrm{new}}.
   $$
5. **Closure.** For each band used at each epoch, form
   $$
   \epsilon
   =
   \max_{b,\,\mathrm{epochs}}
   \left|
   \frac{F_{\mathrm{syn}}\!\big(F_{\mathrm{pre}}\times 10^{m_{\mathrm{new}}}\big)
   -
   F^{\mathrm{GP}}_b}{F^{\mathrm{GP}}_b}
   \right|.
   $$
   Stop if $\epsilon < \texttt{PHOT\_CONVERGENCE\_FRAC}$ (default $0.05$) or if $K+1$ reaches `ITER_GP_MANGLE_MAX_ITERS`.

## 9.3 What is and is not remangled

- Remangling updates masks on **observed spectroscopy epochs** only.
- Dense prediction phases, gap-fill columns, and early extrapolation columns train or decorate the GP surface but are **not** sent through demangle/remangle.
- Optional `full_gp` ASCII exports (`*_spec_extended*.txt`) are for QA and downstream notebooks; they are not the training input of the next iteration.

## 9.4 Diagnostic figures

Per-iteration QA under `twodim_iter/iter_KK/figs/` includes:

- `gp_surface/` — heatmaps and slices of $\mu$ and $\sigma$;
- `gp_vs_mangled/` — mangled input versus GP extract (GP interpolated onto the full prescaled $\lambda$ grid for fair comparison);
- `mangle_delta/`, `residuals/`, `phot_lc/` — mask changes, residual structure, and photometry closure.

\newpage

# 10. Exports and downstream QA (notebooks 7.5)

## 10.1 Final products

When iteration stops, products are copied under `Outputs/<SN>/twodim_iter/final/`, including `full_gp/` extended spectra in linear Å and linear $F_\lambda$. With `ITER_EXPORT_FINAL_SPEC_FOR_QA=True`, QA-oriented copies are also written under `FINAL_spectra_2dim/` with `_FINAL_spec` naming expected by older comparison notebooks.

## 10.2 Notebooks 7.5 and helpers

- `7.5_comparison_check_log.ipynb` integrates final spectra through filters and compares to the light-curve fit (`comparison_check_log_utils.py`, `comparison_trapz_lc.py`).
- `7.5_spectra.ipynb` / `7.5_alternate.ipynb` overlay spectra.
- `bolometric_luminosity.ipynb` constructs bolometric light curves from the SED surface.
- `rimangle_log_spectrum.py` preserves the convention that some on-disk extended spectra store $\log_{10} F$ while in-memory analysis uses linear $F$.

Set `USE_ITER_GP_MANGLE_FINAL=True` so path resolution prefers `twodim_iter/final/` over legacy branches.

\newpage

# 11. Configuration summary (production defaults)

\begin{tabular}{lll}
\toprule
Item & Default & Role \\
\midrule
\texttt{SNNAME\_DEFAULT} & AT2017gfo & Default object name \\
\texttt{USE\_PRESCALED\_SPECTRA} & True & Steps 5--6 read prescaled lists \\
\texttt{MANGLE\_BUNDLE\_AWARE} & True & Shared mangling mask per group \\
\texttt{MANGLE\_GP\_KERNEL\_MODE} & fixed\_5 & PyCoCo-parity mangling lengthscale \\
\texttt{ITER\_GP\_MANGLE\_MAX\_ITERS} & 20 & Outer-loop cap \\
\texttt{PHOT\_CONVERGENCE\_FRAC} & 0.05 & Photometry closure tolerance \\
\texttt{ITER\_MANGLE\_USE\_GP\_WAVELENGTH\_GRID} & True & Full-$\lambda$ remangle constraints \\
\texttt{ITER\_GP\_WARM\_START} & True & Hyperparameter warm start \\
\texttt{USE\_ITER\_GP\_MANGLE\_FINAL} & True & Notebooks 7.5 read iter final \\
2D kernels & Matérn 5/2 additive $w$ and $t$ & Via \texttt{GP\_INFERENCE\_KWARGS} \\
$\sigma_{\mathrm{phot}}$, $\sigma_{\mathrm{spec}}$ floors & 0.012, 0.005 & Binding jitter floors \\
\bottomrule
\end{tabular}

Edit `Codes/pipeline_config.py` for object-specific $t_0$, redshift, and experimental toggles. Prefer command-line flags for one-off experiments (mangling kernel mode, iteration cap, diagnostics).

\newpage

# Appendix A: File inventory

```
Codes/
  0.1_Smooth_spectra_KN.ipynb          spectral smoothing
  1_LC_DustCorrection_KN.ipynb         dust correction
  2_LC_modelRising_KN_fullfit_log.ipynb  Bazin / rising extrapolation
  3_LCfit_KN_log.py                    per-band LC GP driver
  4_Scale_spectra_KN.py                prescale driver
  5_Mangle_spectra_KN_log.py           first-pass mangling driver
  6_Iterative_GP_mangle_KN.py          iterative 2D GP + remangle driver
  clear_sn_run.py                      reset Outputs (and optional intermediates)
  pipeline_config.py                   shared constants and path helpers
  filter_plot_config.json              band colours / exclusions
  7.5_*.ipynb, bolometric_*.ipynb      downstream QA
  helper_scripts/
    lc_gp_fit.py, lc_gp_kernels.py     step 3
    lc_bazin_models.py, lc_bazin_fit.py  step 2 helpers
    spectra_pre_scale.py               step 4
    mangle_spectra_log.py              step 5 / remangle masks
    iterative_gp_mangle.py             step 6 outer loop
    iter_gp_grid.py, gp_surface_extract.py
    GP2dim_utils.py, GP2dim_utils_iter.py, twodim_grid_prep.py
    twodim_gp/                         vendored 2D GP engine + plots
    iter_plot_suite.py                 iteration QA figures
    gp_full_spectra_export.py, gp_final_spec_export.py
    comparison_check_log_utils.py, rimangle_log_spectrum.py
    docs/PHASE*.md, ITERATIVE_MANGLING_DATA_FLOW.md
Inputs/
  Photometry/{1,2,3}_LCs_*/            LC stages
  Spectroscopy/{1,2}_spec_*/           original / smoothed / prescaled
  Filters/GeneralFilters/              throughput curves
  SNe_Info/info.dat                    coordinates / types
  2DIM_priors/                         optional 2D priors
Outputs/<SN>/
  fitted_phot*.dat                     step 3
  mangled_spectra/                     step 5
  twodim_iter/                         step 6
  FINAL_spectra_2dim/                  QA export
docs/
  PIPELINE_USER_GUIDE.md               operator manual
  PIPELINE_WRITEUP.md                  this document
```

# Appendix B: References

- Vincenzi et al. 2019, PyCoCo templates: https://arxiv.org/abs/1908.05228
- Bazin et al. 2011, light-curve functional form: https://arxiv.org/abs/1109.0948
- george documentation: https://george.readthedocs.io/
- Collaborator two-dimensional GP configuration notes (additive Matérn 5/2, jitter floors, early-time tanh blend) are mirrored in `Codes/helper_scripts/twodim_gp/` and in the historical `ryan-final-gp/WRITEUP` handoff.
