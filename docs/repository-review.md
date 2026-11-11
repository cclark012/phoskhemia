# Repository review: analytical triplet kinetics and transient-absorption fitting

**Review date:** 19 October 2026  
**Starting point:** the supplied `phoskhemia-master.zip`, whose archive comment identifies commit `052add8340f7fb4a122ed6c8f48c6acaff8cb6f6`.  
**Scope:** the uploaded source and the implementation delivered with this review. This is not a review of a subsequently changed remote branch, and no experimental transient-absorption dataset was supplied for fitting.

## Assessment and implementation decision

The repository already has a useful separation between a kinetic model, the global fitter, and fitted spectra. Keeping `KineticModel.solve(times, beta)` as a time-by-species matrix lets the analytical sensitization model work with the existing data container and signed wavelength-dependent amplitudes. The main work needed for that path was reliable analytical evaluation, a functioning weighted optimizer, stable linear projection, consistent uncertainty accounting, and data/result contracts that preserve the physical axes. Those changes are included. See [kinetic models](../phoskhemia/kinetics/models.py), [analytical evaluation](../phoskhemia/kinetics/analytic.py), and [global fitting](../phoskhemia/fitting/global_fit.py).

A complete replacement of the array containers was deliberately deferred. Such a replacement would change indexing, NumPy dispatch, plotting, loading, arithmetic, and metadata handling at once, making numerical changes harder to distinguish from API changes. The delivered patch repairs specific invariants and returns ordinary NumPy objects when an operation no longer represents a time-by-wavelength dataset. It does **not** establish complete compatibility with every NumPy operation or automatic propagation of every metadata field. An ordinary container holding arrays and explicit coordinates remains a reasonable future direction if the current subclass requires repeated special cases; it should follow a written operation/metadata contract and migration tests.

### How to read the findings

- **Reproduced defect:** observed with a small deterministic example or regression test. The status says whether it is fixed here.
- **Confirmed remaining issue:** still present in the delivered source and supported by an explicit calculation or reproducible example.
- **Inspection concern:** the source leaves an important assumption or behavior insufficiently specified, but representative experimental files or a broader validation study were unavailable.
- **Design recommendation:** a proposed extension, not a claim that an existing supported operation is broken.

The analytical guide and validation report accompanying this change provide the derivation, parameter conventions, numerical comparisons, and detailed test results. This document focuses on repository behavior and extension priorities.

## Changes included in this implementation

| Area | Original finding | Delivered behavior and evidence |
| --- | --- | --- |
| Triplet kinetics | The existing TTA model used numerical integration and tied the initial condition to the requested integration interval. A fit starting after excitation could therefore change the modeled initial condition. | The new analytical evaluator uses the physical initial condition at `t=0` independently of the first requested sample. `SensitizedTTAModel` exposes natural `[beta, gamma, eta]`; `TTAKineticModel` preserves the legacy logarithmic parameter interface. [Analytical tests](../tests/test_analytic_kinetics.py). |
| Default global optimizer | The optional ODR backend was the default while the least-squares fallback was a stub. Measurement uncertainties did not consistently control the outer kinetic objective. | A working bounded SciPy least-squares backend is the default; per-wavelength standard deviations weight the outer residuals by default. Model bounds and explicitly fixed parameters are supported. [Fitter](../phoskhemia/fitting/global_fit.py), [solver tests](../tests/test_global_fit.py). |
| Spectral projection | Normal equations squared the condition number of the basis, and covariance calculation could still invert a singular matrix after a fallback solve. | A shared SVD projects all wavelengths. A deficient basis is reported, and individual unidentifiable amplitude uncertainties are not presented as finite estimates. A nonzero ridge penalty acts on the same raw amplitudes in the inner and outer objectives. [Projection code](../phoskhemia/fitting/projections.py). |
| Statistics and uncertainty | Reduced chi-square divided a total residual sum by a total uncertainty sum; global degrees of freedom omitted projected amplitudes. | Residuals are divided by their own standard deviations. Parameter counts include fitted amplitudes and account for fixed kinetic parameters. Nonpositive residual degrees of freedom remain visible. AIC/BIC include the known-noise Gaussian normalization; criteria are withheld if the fitted weighting does not match that likelihood. [Diagnostics](../phoskhemia/fitting/validation.py), [diagnostic tests](../tests/test_diagnostics.py). |
| Failure handling | Model exceptions could be hidden behind substitute objective values, and downstream confidence intervals were not consistently restricted to successful fits. | Model failures propagate with their cause. Nonconvergence is reported and kinetic covariance is withheld. Rank deficiencies are exposed. Local covariance remains an approximation, particularly near parameter boundaries. [Global fit tests](../tests/test_global_fit.py). |
| Result consumers | Validation helpers still expected an old dictionary/cache layout, and some referenced `TransientAbsorption` only through a type-checking import. Reconstruction helpers were partly stubs. | Comparisons, residual maps, and wavelength CV accept `GlobalFitResult`. Fixed-fit residuals, weighted residuals, and selected species sums are implemented. Reprojecting spectra at new wavelengths is explicitly distinguished from subtracting the stored fit. [Reconstructions](../phoskhemia/fitting/reconstructions.py), [tests](../tests/test_diagnostics.py). |
| Result reporting | LaTeX renderer arguments reached the wrong function; backslash escaping damaged inserted braces; optimizer parameters were omitted from dictionary output when covariance was absent. | Renderer options are routed correctly, escaping is performed once, and optimizer coordinates are retained. `cov_kinetics` implements existing covariance transformations and uses `expm1` for small variances. The obsolete report implementation was a triple-quoted string, **not an active duplicate function**, and has been removed. [Results](../phoskhemia/fitting/results.py). |
| Data axes and metadata | TA slices and shape-changing operations could retain incompatible axes or noise/probe arrays. Constructing a dataset could affect the caller's coordinate arrays. | TA and 1D constructors copy axes. Supported rectangular TA selections subset axes and known aligned metadata together. TA scalars, individual traces, paired indexing, reductions, and shape-changing operations use appropriate NumPy return types. [Container tests](../tests/test_data_invariants.py). |
| Combining datasets | Time-unit-sensitive comparisons and reconstruction of axes could mis-handle overlap or invent coordinates. Metadata could remain aligned to an old grid. | Identical irregular grids are preserved; supported partial overlaps retain measured coordinates. Irregular, contained, or noncontiguous shared-coordinate overlaps are rejected. Overlapping time intervals with no exact common sample also require explicit alignment. Noise is propagated for supported same-grid averages; unsupported noise/probe propagation is cleared with metadata explaining why. [Combine implementation](../phoskhemia/data/spectrum2d.py), [tests](../tests/test_data_invariants.py). |
| Downsampling | Noise updates used a stride rather than actual bin counts, some aggregation routes referenced an unset scaling variable, and logarithmic selection depended on the numerical time unit. | Actual bin counts are recorded. Independent, equal-size means have their noise scaled once. Unsupported constant-time noise estimates are cleared with a reason. Direct selection subsets aligned noise. Default logarithmic indices use dimensionless elapsed time; an explicitly supplied `eps` retains its historical time-unit meaning. [Downsampling](../phoskhemia/preprocessing/downsampling.py), [tests](../tests/test_data_invariants.py). |
| Correction helper mechanics | `find_time_zero` could return different numbers of times and data rows. `estimate_irf` failed at its `hstack` call. | Time-zero outputs retain matching observations and coordinates. The IRF stacking error and basic invalid-input cases are fixed. Their underlying physical estimation heuristics have **not** been validated by these repairs. [Corrections](../phoskhemia/preprocessing/corrections.py), [correction tests](../tests/test_corrections.py). |

`sample_kinetics` remains explicitly unimplemented. `GlobalFitResult.to_dict()` is a report snapshot, not complete serialization from which the kinetic model and fit can be reconstructed.

## Compatibility changes to account for when returning to the project

| Interface or convention | Behavior to use now |
| --- | --- |
| `fit_global_kinetics` defaults | `solver="least_squares"`, `use_noise_as_sy=True`, and `lam=0.0`. Previous fits can change because the objective and uncertainty accounting have been corrected. Choose an explicit nonzero penalty when wanted; its strength is defined for raw amplitude coefficients. |
| Timing uncertainty | The global objective uses fixed timestamps. Explicit `sx` or scope-uncertainty requests raise an unsupported-operation error. Metadata about the scope does not silently activate a different fit. The optional ODR adapter runs this global objective in OLS mode; it is not a joint errors-in-time implementation. The upstream [ODR API](https://hugomvale.github.io/odrpack-python/reference/) documents that OLS holds explanatory variables fixed. |
| Statistical outputs | Amplitudes count as fitted parameters. Undefined degrees-of-freedom statistics become NaN. Absolute AIC/BIC values differ from the old normalization. Compare models using the same observations, uncertainty convention, and preprocessing. Ridge information criteria and effective counts are approximate. |
| Parameter coordinates | The new model accepts natural rates and dimensionless `eta`; the legacy TTA model still accepts logarithms of its positive natural parameters. Its third parameter is an effective inverse-time rate, not automatically the physical bimolecular coefficient. Read the model's parameter descriptions before reusing an old initial guess. |
| Array-return behavior | A rectangular `ta[time_selection, wavelength_selection]` can remain a `TransientAbsorption`. A single trace, a reduction, a transpose, a reshape, or a paired advanced selection generally returns an ordinary array/scalar. Code relying on `.x`, `.y`, or `.meta` after such operations needs an explicit reconstruction with meaningful coordinates. |
| Addition | Historical `a + b` between two `TransientAbsorption` objects still combines them and **averages overlapping data**. `np.add(a, b)` performs numerical addition. This compatibility behavior is retained; prefer an explicit `combine` call when the purpose is to average measurements. |
| Unsupported merges | Some previously attempted partial overlaps now fail early. Align irregular grids explicitly before combining them rather than relying on the helper to infer an interpolation rule. |
| Noise shape | The global fitter currently accepts a scalar or a wavelength vector constant in time. The data container can carry per-observation noise through selection, but that does not mean the fitter supports it. Aggregation that clears `noise_t0` requires a new uncertainty choice before an uncertainty-aware fit; omission otherwise reaches the documented unity fallback. |

The NumPy return-type changes above apply to `TransientAbsorption`. `Spectrum1D` received constructor ownership fixes, but its general NumPy protocol behavior was not refactored in this patch.

The mathematical matrix conventions remain: data are **time by wavelength**, kinetic traces are **time by species**, and public fitted amplitudes are **wavelength by species**. The simulator's spectral input is species by wavelength, so transposition at that interface must be intentional.

## Remaining priorities

### 1. Correct the pump-to-excited-population model before using it for absolute calibration

**Confirmed remaining issue — high scientific priority.** In [the simulator](../phoskhemia/simulation/transient_absorption.py), `excitation_fraction` multiplies a per-molecule excitation probability by the total absorbed fraction of the sample:

\[
f_{\mathrm{exc}}=\Phi\,[1-\exp(-\sigma F_0)]\,[1-10^{-\varepsilon c\ell}].
\]

For a sufficiently thin sample, the incident fluence is essentially the same for every molecule. The per-molecule excitation probability then tends to a concentration-independent value. The extra sample-absorption factor instead makes the implemented fraction tend to zero with concentration. Multiplying that fraction by the concentration then produces an artificial approximately quadratic concentration dependence of the excited population in this limit.

For example, using `eps_pump=1e4`, a 532 nm pump, a 0.1 cm path, a 1 nJ pulse, and a 0.01 cm² beam area gives fractions approximately `2.355e-8`, `2.358e-11`, and `2.358e-14` for concentrations `1e-6`, `1e-9`, and `1e-12 M`. Their thousandfold changes follow sample concentration, although all three samples are optically thin.

A thin-sample model can use `-expm1(-sigma * F0) * quantum_yield` without the final absorption factor. A finite-optical-density, volume-average model needs a different calculation: in the weak-excitation limit, absorbed photons must be divided by the number of molecules in the illuminated volume. At higher excitation, depth-dependent depletion and the nonlinear annihilation kinetics require further care. Choose and document the intended physical approximation before modifying this helper. Its direct `pump.f_exc` branch can accept a separately established excited fraction without invoking the energy-derived formula.

### 2. Make parameter transformations a model contract

**Inspection concern and design recommendation — high extension priority.** The single `parameterization()` choice, `"log"` or `"linear"`, is too coarse for models that mix lifetimes, rates, offsets, and mixture coordinates. In [the exponential models](../phoskhemia/kinetics/models.py), some lifetimes use exponentials while mixture weights are obtained from [simplex coordinates](../phoskhemia/kinetics/parameterizations.py). Applying an exponential to every optimizer coordinate in the generic reporter does not itself describe the resulting mixture weights or their covariance.

The biexponential helper also uses `b = exp(beta[2])` and `1-b`; this does not constrain both coefficients to be nonnegative. That is acceptable only if a signed biexponential combination is intended. If the coefficients represent population fractions, their allowed domain should be expressed explicitly. This is a model-meaning question, not a reason to silently reinterpret existing saved parameter arrays.

A next API revision should give each model explicit conversion between optimizer coordinates and reported physical parameters, with units, bounds, and a transformation Jacobian. Individual parameters can have linear, logarithmic, or bounded transforms; simplex weights need a grouped transform. Both parameter reporting and covariance propagation should use that same contract. Add round-trip and finite-difference Jacobian tests, then migrate legacy models with documented parameter-name and serialization changes.

### 3. Keep identifiable kinetic combinations separate from absolute concentration claims

**Structural property of the supplied observation model — not a numerical defect.** For zero initial annihilator population,

\[
n(t)=\frac{P}{\beta}w(t;\beta,\gamma,\eta),
\qquad \eta=\frac{qP}{\beta^2}.
\]

When each species has a freely fitted spectral amplitude, the concentration scale is absorbed by that amplitude. Scaling `n` and `P` by a positive factor `c`, scaling `q` by `1/c`, and scaling the annihilator differential extinction coefficient by `1/c` preserves the signal. A corresponding unknown sensitizer excitation scale can be absorbed by its spectrum as well. The normalized fit therefore targets `beta`, `gamma`, and `eta`; it does not independently determine physical `q` and `P` from free spectra. More wavelengths can improve precision without removing this scale ambiguity.

Knowing the ground-state concentrations does not automatically provide the initially excited sensitizer concentration or the triplet differential extinction coefficients. Recovering `q = eta * beta**2 / P` needs an absolute scale anchor, such as calibrated excitation and transfer parameters or known differential extinction coefficients and optical path. Relative pump-fluence series can strengthen the kinetic test, but an unknown common absolute scale can remain. Separating `k_3S` from `k_TET*[A]` also requires information beyond their sum `beta`.

Practical identifiability is a separate question: weak annihilation, unresolved rise times, similar kinetic traces, missing early observations, and low signal-to-noise can make even the normalized parameters poorly determined. Use multiple initial guesses and profile likelihood or suitable resampling before interpreting a boundary estimate or a narrow local interval. The delivered covariance and rank diagnostics are local checks, not proof of physical identifiability.

### 4. Define time, units, noise, and preprocessing provenance together

**Confirmed remaining behavior plus design recommendations.** The data repairs preserve known metadata through specific operations; they are not a general uncertainty calculus. A current deterministic example is `2 * ta`: the signal doubles while `noise_t0=[1, 2]` remains `[1, 2]`, rather than `[2, 4]`. Coordinate-preserving NumPy operations can therefore still carry scientifically stale noise metadata. Similarly, a frozen result dataclass does not make its contained arrays and dictionaries immutable. [Container](../phoskhemia/data/spectrum2d.py), [metadata](../phoskhemia/data/meta.py), [results](../phoskhemia/fitting/results.py).

Specify a small schema before extending the processing chain:

- Distinguish time, wavelength, signal, concentration, path-length, and extinction-coefficient units. Keep the original time origin and any applied shift. A model's `time_unit` currently labels reports; it does not convert inputs or check them against dataset metadata.
- Store whether uncertainty is a standard deviation, a variance, or a weight, and whether it is scalar, per wavelength, per observation, or correlated. The existing fitter expects standard deviations. Do not infer this from array shape alone.
- State which operations propagate uncertainty, clear it, or require a supplied covariance model. Scalar calibration and averaging need explicit behavior; arbitrary NumPy dispatch should not imply that uncertainty has been propagated.
- Record bin membership or original sample times when averaging in time. The average of a curved kinetic function across a bin generally differs from its value at the bin's average time. Correct noise alone does not eliminate that approximation.
- Record assumptions about independent observations. Smoothing, interpolation, and SVD reconstruction can introduce correlations, so a smaller apparent residual variance does not imply more independent information.

Centralize unit conversion. With concentration units unchanged, seconds to nanoseconds multiplies times and lifetimes by `1e9`, divides `beta`, `gamma`, `P`, and physical `q` by `1e9`, and leaves `eta` unchanged. A concentration-unit conversion also changes `P`, `q`, and extinction coefficients. The conversions should occur once through an explicit operation, with invariance tests, rather than through ad hoc rescaling inside individual models.

The unequal-bin repair intentionally clears an invalid constant-time `noise_t0` and records why. It does **not** implement weighted global fitting of those unequal-bin means. Supporting time-dependent uncertainties or wavelength-dependent missing observations will require a weighted design matrix for each wavelength, appropriate observation counts, and a revised projection strategy; the current single shared SVD relies on time-constant weights within each wavelength.

### 5. Add an optional observation/calibration layer without changing the kinetic evaluator

**Design recommendation.** The present free signed amplitudes are appropriate for exploratory global TA fitting. A calibrated mode could explicitly represent

\[
\Delta\mathrm{OD}_\lambda(t)=\ell\left[
\Delta\varepsilon_{S,\lambda}S_0e^{-\beta t}
+\Delta\varepsilon_{A,\lambda}\frac{P}{\beta}w(t)
\right]+b_\lambda.
\]

This layer can distinguish calibrated spectra from freely fitted spectra, include an optional baseline, attach the path length, and later apply a measured instrument response or a fitted time offset. It should keep physical concentration scaling visible and preserve signed differential spectra: ground-state bleach makes nonnegative-amplitude constraints generally inappropriate for the measured TA signal.

Keep the analytical population function independent of this observation layer. Known calibration inputs, independent linear coefficients, and additional nonlinear parameters have different fitting and uncertainty roles. Identify each explicitly and count any new fitted baseline or response parameters in the diagnostics. Prefer species names at the calibration/simulation boundary so a positional interchange of sensitizer and annihilator spectra cannot silently pass a shape check.

The existing `find_time_zero` and `estimate_irf` remain **heuristics**. The former uses a 50-sample smoothing neighborhood and a ±20 search interval in the supplied time units; the latter has no time axis and works in sample coordinates. Mechanical fixes establish matching shapes and callable code, not correct recovery of an instrument response from noisy experimental kinetics. Validate an IRF/timing model against a reference measurement before using these helpers as calibration.

### 6. Build a small instrument-file fixture suite before expanding I/O

**Inspection concern and design recommendation.** [The MATLAB loader](../phoskhemia/data/io.py) assumes a specific matrix layout: a wavelength row, a probe row, and a time column preceding the signal. That layout should be an explicit format contract. Its MATLAB 7.3 fallback reads HDF5 objects directly into arrays; support for real MATLAB dimension ordering, references, cells, or structures needs representative fixtures rather than an assumption that every HDF5 file is a plain numeric matrix.

The absorption CSV loader also interprets paired columns, labels, units, and footers. Synthetic constructor tests and a callable lazy export do not validate real instrument exports. Add small anonymized fixtures with expected axes, values, units, probe metadata, and provenance. Useful cases include the actual TA MAT export, MATLAB 7.3 if used, one-wavelength/one-time boundary cases, ambiguous matrix keys, descending axes, missing/nonfinite values, paired absorption columns, and an unrecognized footer that must be preserved.

The delivered I/O-related fixes make the lazy `load_absorption_collection` export available and declare `h5py` in the optional `io` dependency extra. The implementation of `data/io.py`, including its metadata imports, is unchanged from the supplied archive. A tiny numeric `savemat`/`load_mat` **manual smoke check** succeeded on that existing path; no automated MAT or CSV fixture suite is included. That check is not general validation of MATLAB 7.3 or of all absorption-file variants. Keep future round-trip persistence distinct from the current report-only `to_dict()` method.

### 7. Establish a supported-version and performance baseline

**Verification limitation and design recommendation.** The execution environment for this work used Python **3.12.14**, NumPy **2.3.5**, and SciPy **1.17.0**. The supplied package metadata declares Python `>=3.13` and classifiers for 3.13/3.14. Source-level tests in this environment do not validate installation or behavior on those declared Python versions. The native optional `odrpack` backend was unavailable; its mocked interface test is not a native backend test. [Package metadata](../pyproject.toml), [regression modules](../tests/).

Run CI on the declared Python versions with a documented minimum dependency set and a current set. Include installation/import checks and representative fixture-based I/O tests. Test optional backends in a separate job rather than making their availability change the default optimizer. Reassess whether format/acceleration packages belong in core dependencies or explicit extras after their supported workflows are documented.

The analytical evaluator avoids ODE integration in the fit, but evaluating a matrix exponential at each distinct time still has a cost. Benchmark complete fits on representative time grids, wavelength counts, parameter bounds, and initial guesses before adding acceleration. Keep an independently checked implementation as a reference. Its truncation tolerance controls the analytical series tail, not all floating-point error or every parameter regime.

## Suggested order of further work

1. **For experimental interpretation:** retain reliable noise and timing information, establish the concentration/calibration assumptions, and test identifiability on representative raw datasets. Address the energy-derived excitation-fraction model before using it to infer absolute `P` or `q`.
2. **For API extension:** implement the parameter-transform contract and explicit units/noise schema, then add the optional observation model and support for masks or time-dependent noise where needed.
3. **For long-term maintenance:** add real-format fixtures and supported-version CI, then decide whether NumPy subclassing still serves the now-specified container contract. A later container migration can preserve the kinetic and fitting interfaces rather than rewriting their validated mathematics.

The delivered changes establish a tested path for the stated zero-initial-population, exponentially sensitized model with fixed timestamps and per-wavelength noise. They do not claim that every prototype feature, file format, preprocessing heuristic, or experimental inference problem in the repository is complete.
