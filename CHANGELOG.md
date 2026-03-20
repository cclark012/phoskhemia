# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

Changes before and including 0.1.6 were not well documented other than with git commits. As of [2026-03-20], not all changes have been included in this document.

Everything is subject to change, so I cannot guarantee that all functions and modules will act the same in the future.

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
