# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

Changes before and including 0.1.5 were not well documented other than with git commits. As of [2026-03-02], not all changes have been included in this document.

## [0.1.5] - [Unreleased]

### Added

- simulation Module
  - absorption.py
  - lineshapes.py
  - noise.py
  - sources.py
  - transient_absorption.py

- data Module
  - io.py
  - meta.py

- fitting Module
  - results.py
  - solvers.py
  - reconstructions.py

- kinetics Module
  - parameterization.py

- preprocessing Module
  - downsampling.py
  - svd_denoise.py
  - svd_ek.py

- utils Module
  - indexing.py
  - performance.py

- visualization Module
  - colormaps.py

- TransientAbsorption.export_csv for exporting traces/spectra/slices to CSV (wide/long) with size guards.

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

- visualization/plotting.py

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
