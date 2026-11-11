# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

Changes before and including 0.1.6 were not well documented other than with git commits. As of [2026-03-20], not all changes have been included in this document.

Everything is subject to change, so I cannot guarantee that all functions and modules will act the same in the future.

## Unreleased — analytical TTA and fitting correctness

Based on the user-supplied archive for commit
`052add8340f7fb4a122ed6c8f48c6acaff8cb6f6`. The source version stays 0.1.6;
these changes have not been published as a release.

### Added

- Analytical normalized and dimensional triplet-population evaluators, with
  controlled positive-series truncation and no production ODE integration.
- `SensitizedTTAModel` using natural `[beta, gamma, eta]` parameters and
  `[annihilator, sensitizer]` basis columns.
- An optimizer-coordinate `KineticModel.parameter_bounds()` hook.
- Bounded SciPy least-squares global fitting, fixed kinetic parameters, and
  rank-aware local covariance calculations.
- Signed, shared-SVD spectral projection and a seeded global-fitting example.
- Regression tests, model/fitting documentation, a prioritized repository
  review, and a Python 3.13/3.14 CI configuration.
- An optional `io` dependency group for MATLAB v7.3's `h5py` reader.

### Changed defaults and compatibility

- Global fitting defaults to `solver="least_squares"`, `use_noise_as_sy=True`,
  and `lam=0.0`. A noise scalar/vector is used in the nonlinear objective.
- `solver="auto"` consistently selects least squares, independent of installed
  optional packages. Global ODRPACK usage now explicitly selects fixed-time OLS.
- `sx` and scope-uncertainty arguments raise an explicit unsupported-operation
  error for global variable projection. The previous callback ignored adjusted
  timestamps, so its apparent errors-in-time support was not reliable.
- `TTAKineticModel` retains log-lifetime/effective-rate coordinates but uses the
  analytical evaluator and anchors its initial condition at physical t=0.
- New TTA models reject negative/pretrigger times. Time units are explicit
  labels and must be converted by the caller along with rate values.
- Shape-changing NumPy operations and scalar/vector selections return ordinary
  arrays/scalars when the TA coordinate interpretation would otherwise be invalid.
- Trim includes its nearest selected endpoints and validates actual bounds.
- Default logarithmic downsampling now chooses the same observations after a
  consistent change of time units.
- Unsupported partial-overlap combination grids raise clear errors. Identical
  irregular grids remain supported. Historical `ta1 + ta2` averaging is retained.

### Corrected

- The long-time Bessel overflow/cancellation failure of direct formula evaluation.
- Legacy TTA dependence on the first requested sample and sorted integration grid.
- Normal-equation amplitude projection and rank-deficient covariance crashes.
- Reduced chi-square's per-observation weighting and total fitted parameter count.
- AIC/BIC likelihood normalization, covariance scaling, and report/result consumers.
- The mismatch between inner ridge projection and the outer objective.
- Suppressed model exceptions; failed fits no longer receive kinetic intervals.
- Coordinate-dependent noise/probe metadata after slicing, trimming, supported
  means, and combination. Unsupported uncertainty transformations remove stale
  metadata with a reason rather than preserving the first input's sigma.
- Recursive NumPy dispatch, caller-owned axis mutation/freezing, and a missing
  lazy data export.
- The one-component exponential/simplex case.
- Time-zero output length mismatch and the IRF helper's `hstack` call.
- The setuptools minimum needed by the declared PEP 621 project metadata.

### Remaining limits

This is a prototype update. It does not supply a calibrated IRF model, general
uncertainty propagation for arbitrary array arithmetic, complete model
serialization, or independent calibration of the initial triplet concentration.
The simulator's automatic pump-excitation approximation needs a separate
physical correction before it is used to infer absolute q/k2. See
[docs/repository-review.md](docs/repository-review.md).

Local validation used Python 3.12.14 with NumPy 2.3.5 and SciPy 1.17.0. The
project's declared Python >=3.13 support matrix and real optional ODR backend
still need execution in their corresponding environments.

---

Previous project entries are retained below as supplied.

## [0.1.6] - [Unreleased]

### Added

- CHANGELOG.md
  - A changelog to keep track of new, changed, fixed, and removed features

- fitting/solvers.py
  - Support for solvers other than odrpack

- utils
  - formatting.py
    - User-facing information formatters
  - performance.py
    - Performance measurements

- `Spectrum1D` base class for single-spectrum 1D data with wavelength axis and metadata.
- `FluorescenceSpectrum` and `AbsorptionSpectrum` spectrum handlers.
- `SpectrumCollection`, `FluorescenceCollection`, and `AbsorptionCollection` for multi-spectrum datasets.
- `load_absorption_collection()` parser for multi-series absorption CSV files with heterogeneous wavelength grids and mixed units.

### Changed

### Fixed

### Deprecated


## [0.1.5] - 2026-02-17

### Added

- simulation Module
  - absorption.py
    - Various models for absorption spectra
  - lineshapes.py
    - Probability distributions to emulate spectral broadening
  - noise.py
    - Functions used to emulate sources and distributions of noise
  - sources.py
    - Simulated properties of light sources
  - transient_absorption.py
    - Simulate transient absorption spectra with chosen spectral and temporal profiles

- data Module
  - io.py
    - File and data input/output
  - meta.py
    - Metadata handlers

- fitting Module
  - results.py
    - Results and functions to aid in interpretability or saving
  - reconstructions.py
    - Functions used to reconstruct fits from previous results

- kinetics Module
  - parameterization.py
    - Estimators for fitting functions

- preprocessing Module
  - downsampling.py
    - Downsampling schemes to reduce the size of data
  - svd_denoise.py
    - Various SVD-related denoising schemes
  - svd_ek.py
    - SVD reconstruction based on Epps and Krivitzky 2019

- utils Module
  - indexing.py
    - Utilities to find data easily

- visualization Module
  - colormaps.py
    - Functions to make and sample colormaps
  - plotting.py
    - Pre-defined plots

- TransientAbsorption.export_csv for exporting traces/spectra/slices to CSV (wide/long) with size guards.
- GlobalFitResults has several helper functions to create fit summaries and save fitting results

### Changed

- Metadata handling via MetaDict for attribute access (e.g., meta.noise_t0) and schema normalization.
- fit_global_kinetics now returns a GlobalFitResult object.

### Fixed

- combine: corrected concatenation axis for time merges.
- annotations: added `from __future__ import annotations` to files to fix type annotation errors.
- TransientAbsorption: updated list of _SUPPORTED_ARRAY_FUNCTIONS
- ImportErrors: Various functions and modules are imported from the files rather than from modules with exposed functions. Many imports called inside functions.
- Noise: Allow None as a value for several functions.
- Covariances: Covariances are now scaled by residual variance as expected.
- Guards: Several guards added to pre-existing and new functions to protect against unintended behavior.

### Deprecated

- None

---

## [0.1.4] - 2026-01-30

- Bug with PyPI, package wasn't uploaded

---

## [0.1.3] - 2026-01-30

- Bug with PyPI, package wasn't uploaded

---

## [0.1.2] - 2026-01-30

### Added

- kinetics Module
  - base.py
    - KineticModel
  - models.py

- data Module
  - spectrum_handlers.py
    - TransientAbsorption

- fitting Module
  - global_fit.py
  - initialization.py
  - projections.py
  - validation.py

### Changed

- analysis/ -> fitting/
- io/ -> data/

### Fixed

- Various import bugs

### Deprecated

- None

---

## [0.1.1] - 2025-02-03

### Added

- utils Module
  - typing.py

- analysis Module
  - fitting_functions.py
  - statistics.py
  
- visualization Module

### Changed

- io/printing_utils.py -> utils/printing_utils.py

### Fixed

- Package requirements

### Deprecated

- None

---

## [0.1.0] - 2025-01-06

### Added

Initial release.

- preprocessing Module
  - corrections.py
  - smoothing.py

- io Module
  - printing_utils.py
  - spectrum_handlers.py
