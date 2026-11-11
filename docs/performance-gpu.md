# Analytical TTA performance and experimental GPU projection

## Summary

`SensitizedTTAModel(evaluation="direct")` evaluates one `scipy.linalg.expm` for **every distinct time** in each kinetic trial. An experimental 50,000-point grid therefore triggers about 50,000 small dense matrix exponentials for each optimizer evaluation (and additional evaluations for finite-difference derivatives). This is the dominant cost when the data matrix is modest. It is not an inherent cost of the exact Riccati solution.

Use `evaluation="accelerated"` to replace dense pointwise evaluation with analytical series samples, local Chebyshev interpolation, and an exact no-source Riccati tail. Interpolation is checked at independent nodes. It has no mathematically rigorous uniform bound; the neglected exponential source after `s_cut` does have the rigorous normalized absolute bound `exp(-s_cut)`. The legacy direct evaluator remains the numerical reference.

```python
from phoskhemia.kinetics import SensitizedTTAModel

model = SensitizedTTAModel(
    time_unit="ns",
    evaluation="accelerated",
    series_tolerance=1e-14,
    interpolation_tolerance=1e-11,
)

result = ta.fit_global_kinetics(
    kinetic_model=model,
    beta0=[1 / t3s + 1 / tr, 1 / t3a, eta_guess],
    noise=sigma_by_wavelength,
    lam=0.0,
    solver="least_squares",
)
```

Times, beta, gamma and rates must be consistently converted; `eta` is dimensionless and is unchanged between seconds and nanoseconds. Initial beta must include *both* intrinsic sensitizer decay and energy transfer. For normalized populations with initial sensitizer concentration one, the quadratic coefficient is `k2_physical*S0`, with inverse-time units. In that convention `eta=(k2_physical*S0)*K/beta**2` and `K=1/tr`.

### Evaluator benchmark

Measured with `OPENBLAS_NUM_THREADS=1`, 50,000 uniformly spaced times between 1 ns and 1 ms, `beta=0.006 ns^-1`, `gamma=4e-6 ns^-1`:

| eta | direct analytical evaluator | accelerated analytical sampling | speedup |
|---:|---:|---:|---:|
| 0.0001 | 5.98 s | 0.051 s | 118x |
| 10 | 10.42 s | 0.123 s | 85x |

These are single **basis evaluations** on the supplied host, not full global-fitting timings. Full fitting speedup will be lower if amplitude projection, data movement, uncertainty calculations or the optimizer dominates. Unit tests compare the accelerated interpolant with the direct evaluator across 36 `(nu,eta)` combinations, including `nu=0`, `nu=1`, and `eta=10`. Maximum observed absolute shape discrepancies in those tests are below `5e-10`; this is an observed test bound, not a universal guarantee.

## The scaling issue in the existing global fitter

The default global fit currently allocates `np.tile(times, n_wavelengths)`, a wavelength-major flattened copy of the TA matrix, full repeated per-observation noise weights, model predictions and residuals. At 50 million double-precision samples, each such array consumes 400 MB. Multiple live copies can consume several gigabytes and add memory bandwidth costs for each trial.

For fixed timestamps, `lam=0`, and constant-in-time wavelength noise, the exact profiled objective can instead be written:

```
C = Q @ R                   # reduced QR/SVD of the (n_times, n_species) basis
chi_squared = ||Y/sigma||_F**2 - ||(Q.T @ Y)/sigma||_F**2
```

The expensive term `||Y/sigma||_F**2` is computed only once. Only the two-by-wavelength cross-products `Q.T @ Y` are evaluated for each kinetic trial. This avoids materializing an equally large prediction and residual matrix and permits efficient GPU offloading of the data-dependent cross-products. Numerical subtraction can lose relative precision when the residual noise is extremely small compared with the signal; compare against an explicit residual calculation on representative datasets.

`phoskhemia.fitting.compact_objective` provides an **experimental** implementation with `device="cpu"` or `device="cuda"`. It uses a CPU basis evaluator, CPU SciPy optimization and CPU SVD of the narrow kinetic basis, optionally keeping the measured TA matrix and cross-products on a CUDA GPU via CuPy. The public result is `CompactFitResult`, not the full `GlobalFitResult`; covariance, optional penalties, x uncertainty, nonseparable noise and out-of-core GPU chunking are NOT supported.

```python
from phoskhemia.fitting.compact_objective import fit_compact_global
from phoskhemia.kinetics import SensitizedTTAModel

model = SensitizedTTAModel(time_unit="ns", evaluation="accelerated")

fit = fit_compact_global(
    times=ta.y,
    data=ta,
    sigma=sigma_by_wavelength,
    kinetic_model=model,
    initial_parameters=[0.006, 4e-6, 1e-4],
    parameter_scales=[0.006, 4e-6, 1e-4],
    bounds=([1e-3, 0.0, 0.0], [1e-1, 1e-3, 10.0]),
    device="cpu",  # "cuda" is an untested optional CuPy path
)

print(fit.parameters, fit.chi_squared)
```

The GPU path is supplied as a **prototype only**, because the execution environment did not expose a CUDA GPU or CuPy. Its correctness, asynchronous timing, memory behavior, and runtime performance must be measured on the target system. Keep CuPy an optional dependency, and validate against the CPU profile objective before relying on CUDA results.

### Compact projection reference benchmark

For 50,000 times and 150 wavelengths (7.5 million observations, 60 MB of
raw `float64` data) with two kinetic components, a single projected objective
calculation took approximately **0.008 s** with the compact CPU objective
versus **0.060 s** for the existing `project_spectra` plus explicit weighted
residual calculation. Initialization and the one-time weighted data-norm
computation took approximately **0.007 s**. The two weighted chi-square
values agreed to the printed precision (7,501,970.63). These timings exclude
kinetic evaluation and optimizer overhead, and do not measure the GPU path.

## GPU backend selection

- **CuPy**: best first option for this existing NumPy/SciPy-oriented repository. Keep `Y` resident on GPU for repeated projection cross-products; transfer the small `Q` per trial. Optional CuPy installation depends on compatible CUDA drivers and selected CUDA wheels.
- **PyTorch**: viable, especially for later autograd or coupled differentiable physics models. `torch.linalg.lstsq` on CUDA currently uses only the full-rank `gels` driver, so a direct drop-in replacement would lose the rank-revealing fallback of the existing CPU SVD approach. SVD or QR of the small basis on CPU avoids that restriction.
- **Gpufit**: intended primarily for many independent small nonlinear fits; the present model has globally shared kinetic parameters and separately projected amplitudes. Adding a custom model in Gpufit requires compiling C/CUDA code and model derivatives, and its Python fit interface uses `float32` arrays. It is not an attractive first choice for this fit architecture.

GPU acceleration is not guaranteed to improve a single small dataset. Host-to-device transfers and kernel launch overhead can dominate unless data stay resident across iterations. Consumer GPUs often have poor float64 throughput compared with their float32 performance. The cancellation in the compact objective must be checked, especially at high SNR. Full instrument-response convolution or time-varying noise would also change the optimal projection implementation.

## Work recommended before 50 million point production fits

1. Profile individual phases: analytical shape, amplitude projection/cross-products, solver, reconstruction and reporting. Separate one-time setup from repeated operations; include `nfev` and `njev`.
2. Validate the accelerated analytical evaluator across the intended parameter box. Compare select points to the direct analytical function, including the early rise and tail, and convert numerical errors into `delta absorbance / sigma`.
3. Benchmark the compact CPU objective with representative full-size data, including convergence and objective agreement; retain the existing fitter as the reference for uncertainty.
4. Add a tested CUDA backend on the actual NVIDIA GPU, making data residency and optional GPU chunking explicit. Benchmark with CUDA events or CuPy's profiler, not an unsynchronized wall-clock read.
5. Design a common result and uncertainty interface for scalar-objective optimizers. A Gaussian-noise Hessian or profiled-Jacobian covariance calculation requires separate implementation, plus checks for nonidentifiability and active bounds.

## Explicit limitations

- The fast method is approximate analytical interpolation with sample-based checks; it does **not** integrate the original nonlinear ODE.
- The source-free long-time expression is accurate only to a bounded neglected-source contribution, not a physically instantaneous source cutoff.
- The compact scalar objective uses weighted, fixed-time, unregularized least squares and assumes one sigma per wavelength.
- CUDA behavior and full 25–50-million-point end-to-end performance have not been tested here.
