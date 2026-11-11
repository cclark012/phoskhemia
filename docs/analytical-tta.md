# Analytical sensitization and mixed-order triplet decay

## The equation and its parameters

The physical model is

\[
\dot n=P e^{-\beta t}-\gamma n-qn^2,\qquad n(0)=0.
\]

Here `n` is the annihilator triplet concentration. With concentrations in M and
time in seconds:

| Symbol | Physical meaning | Units |
|---|---|---|
| `P` | `k_TET*[A]*S0` | M/s |
| `beta` | `k_3S + k_TET*[A]` | 1/s |
| `gamma` | `k_3A` | 1/s |
| `q` | Coefficient multiplying `n**2` in the stated population equation | 1/(M s) |
| `S0` | Initial sensitizer triplet concentration | M |

Some TTA conventions define an event-rate constant and put an additional factor
of two in the population loss term. This API takes the coefficient of `n**2`
exactly as written; any such factor belongs in `q`.

Define

\[
N_* = P/\beta,\quad s=\beta t,\quad \nu=\gamma/\beta,
\quad \eta=qP/\beta^2,\quad n=N_*w.
\]

Then

\[
\frac{dw}{ds}=e^{-s}-\nu w-\eta w^2,\qquad w(0)=0.
\]

`N_*` is the total injected annihilator concentration in the absence of its
losses, not necessarily the maximum population reached in the experiment.
The default fit returns the identifiable normalized kinetic shape `w`.

## Public API

```python
from phoskhemia.kinetics import (
    SensitizedTTAModel, TTAKineticModel,
    analytic_shape, triplet_population, series_order,
)

# Normalized shape; arbitrary array shape, unsorted times, duplicates allowed.
w = analytic_shape(times, beta=6e6, gamma=1e3, eta=8.333333333333333e-5)

# Dimensional concentration in M for these units.
n = triplet_population(times, P=3.0, beta=6e6, gamma=1e3, q=1e9)

# Model interface: times must be one-dimensional.
model = SensitizedTTAModel(time_unit="s")
C = model.solve(times, [6e6, 1e3, 8.333333333333333e-5])
# C[:, 0] = w(t), C[:, 1] = exp(-beta*t)
```

The low-level helpers preserve the input array shape, including scalar input
as a zero-dimensional array. The model requires a one-dimensional time axis.
All times must be finite, nonnegative and relative to excitation. Parameters
must be finite, with `beta>0` and `gamma, eta, P, q>=0` as applicable.
`gamma=0`, `eta=0`, and `P=0` have supported limits. A nonzero initial
annihilator population is not included in this model.

The initial condition is always at `t=0`. Starting a fit at 20 ns does not
reinitialize the population at 20 ns. Pretrigger data must be excluded or
handled by a separate causal observation/response model; this evaluator does
not silently clip negative times or move time zero.

## How this corresponds to transient absorption

Using an optical path length `ell`, or folding it into the spectral amplitudes,

\[
\Delta A(t,\lambda)=a_A(\lambda)w(t)+a_S(\lambda)e^{-\beta t},
\]

\[
a_A(\lambda)=\ell\,\Delta\varepsilon_A(\lambda)N_*,\qquad
 a_S(\lambda)=\ell\,\Delta\varepsilon_S(\lambda)S_0.
\]

This is the user's sensitizer-plus-annihilator signal model. The coefficients
may have either sign because ground-state bleach and excited-state absorption
contribute to the differential signal. The returned basis order is **[A, S]**,
matching the existing TTA model. Fitted amplitudes have shape
`(n_wavelengths, 2)`; reconstruction is `C @ amplitudes.T`.

Do not interpret an amplitude as an extinction coefficient without the path
length and concentration normalization. Knowing ground-state sample
concentrations does not by itself determine `S0`, the initially excited
triplet concentration.

Let `K = k_TET*[A]`. If `K` and `S0` are independently known,

\[
q=\eta\beta^2/(K S_0).
\]

If only `K` is known, the fit identifies `q*S0 = eta*beta**2/K`.
With freely fitted spectral amplitudes and no independent concentration scale,
`q` and `S0` cannot be separated. Multiple pump powers or concentrations can
help only when the appropriate shared calibration and kinetic relationships
are imposed. A joint multi-dataset fitter for those relationships is not part
of this update.

For a normalization by `N_*`, the effective inverse-time nonlinear coefficient
is `q_eff=q*N_*=eta*beta`. A normalization by `S0` instead gives `q*S0`.
Those coefficients differ by `K/beta` and coincide only when that ratio is one.

## Fitting an existing TransientAbsorption

```python
import numpy as np
from phoskhemia.kinetics import SensitizedTTAModel
from phoskhemia.fitting.reconstructions import (
    reconstruct_fit, reconstruct_residuals, reconstruct_weighted_residuals,
)

# ta.y is in seconds and already refers to excitation time zero.
# Restrict the selected interval as appropriate for the instrument response.
selected = ta[ta.y >= 0, :]
sigma_lambda = np.asarray(selected.meta["noise_t0"])

result = selected.fit_global_kinetics(
    SensitizedTTAModel(time_unit="s"),
    beta0=np.array([6e6, 1e3, 8.33e-5]),
    noise=sigma_lambda,
    bounds=([1e3, 0.0, 0.0], [1e8, 1e5, 10.0]),
    x_scale=np.array([6e6, 1e3, 1e-4]),
    maxit=200,
    ci_level=0.95,
)

print(result.kinetics)
print(result.kinetics_ci)
print(result.diagnostics)
fit = reconstruct_fit(result)
residuals = reconstruct_residuals(result, selected)
standardized = reconstruct_weighted_residuals(result, selected)
```

The bounds above are an example domain, not universal physical limits. Use
independent information about sensitizer decay, transfer, and lifetimes when
available. The model's default bounds allow any positive beta and nonnegative
gamma and eta; extremely large eta can exceed the series evaluator's practical
order limit and raises an explicit error.

Use `fix_beta` to hold a parameter exactly, including an exact zero:

```python
linear_limit = selected.fit_global_kinetics(
    SensitizedTTAModel(),
    beta0=[6e6, 1e3, 0.0],
    fix_beta=[False, False, True],
    noise=sigma_lambda,
    maxit=200,
)
```

Despite the legacy name `beta0`, this argument means the full optimizer
parameter vector. For the new model, its three entries are natural
`[beta, gamma, eta]`, not logarithms.

## The global objective and uncertainties

At every kinetic trial, one SVD of the time-by-species basis projects all
wavelength-dependent amplitudes. For positive wavelength standard deviations,

\[
\chi^2=\sum_{i,j}
\left[\frac{Y_{ij}-\sum_k C_{ik}(\theta)A_{jk}}{\sigma_j}\right]^2.
\]

A scalar wavelength weight cancels from that wavelength's unregularized
conditional amplitude solution. It still changes the importance of that
wavelength in the nonlinear objective; weighting only the amplitude projection
would leave the global fit effectively unweighted.

Defaults are `solver="least_squares"`, `use_noise_as_sy=True`, and `lam=0`.
`noise` accepts a positive scalar or vector of length `n_wavelengths`.
If omitted, metadata `noise_t0` is used, then unity. Matrix noise is rejected
explicitly by this global fitter, even though the data container can preserve
such metadata when slicing. Exclude invalid observations rather than passing
zero, infinite, or NaN sigmas.

For a full-rank two-species fit at L wavelengths with three free kinetic
parameters, the unregularized parameter count is `3+2*L`. Fixed kinetic
parameters are excluded, and projected amplitudes use the basis rank. These
counts enter reduced chi-square and residual-variance covariance scaling.
Counting only the nonlinear parameters understates how much was fitted.

`absolute_sigma=False` estimates a common residual-variance multiplier for the
supplied relative sigma pattern. `True` treats objective sigmas as known.
Kinetic covariance is a local linear approximation. Amplitude errors are
conditional on kinetics unless `propagate_kinetic_uncertainty=True`; that
option adds a linearized contribution from kinetic covariance. Boundary
estimates, strong parameter correlations, model mismatch, and correlated
noise need more than these local intervals. Check
`result._cache["solver_info"]` for convergence, active bounds, Jacobian rank,
and the reason a covariance was withheld.

A rank-deficient kinetic Jacobian or optimizer nonconvergence produces no
kinetic covariance/intervals. Rank-deficient individual amplitudes get NaN
uncertainties rather than a misleading finite error. Nonpositive residual
DoF produces undefined reduced chi-square. `AIC`, `AICc`, and `BIC` use the
full independent Gaussian likelihood; comparisons require the same measured
data and noise model. They are withheld if an explicitly unweighted fit does
not optimize the supplied heteroscedastic likelihood.

Optional `lam>0` minimizes the same raw-amplitude ridge penalty in both inner
and outer objectives. It makes results depend on basis normalization, uses
approximate effective parameter counts, and withholds kinetic covariance.
The default avoids imposing that extra physical/spectral assumption.
Robust losses are rejected until their inner amplitude optimization is
implemented consistently.

Timestamps are fixed. `solver="odrpack"` uses the optional package in OLS
mode; `sx` and oscilloscope uncertainty arguments are explicitly unsupported
by this global profile objective. A correct errors-in-time extension must
account for amplitudes depending on all observations and for physically shared
timestamps across wavelength channels. Simply letting an ODR callback adjust
independent flattened timestamps does not establish that model.

## Seconds, nanoseconds, and concentration units

For `t_ns = 1e9*t_s`, multiply beta, gamma, P, and q by `1e-9` when keeping
concentrations in M. Eta and the concentration scale P/beta remain unchanged.

```python
parameters_s = np.array([6e6, 1e3, 8.333333333333333e-5])
parameters_ns = parameters_s * [1e-9, 1e-9, 1.0]
C_ns = SensitizedTTAModel(time_unit="ns").solve(times_s * 1e9, parameters_ns)
```

`time_unit` labels reported units; it performs no implicit conversion.
Optimizer `x_scale`, explicit bounds, and any timing uncertainty or supplied
preprocessing epsilon must use the same coordinates as their inputs.

If concentration is changed from M to micromolar, multiply n and P by `1e6`
and divide q by `1e6`. Beta, gamma, nu and eta remain unchanged.

## Why the earlier Bessel expression failed numerically

For `q>0`, substituting `n=u_t/(q*u)` gives

\[
u_{tt}+\gamma u_t-qP e^{-\beta t}u=0,
\qquad u(0)=1,\quad u_t(0)=0.
\]

Thus, in the handbook's *second-order* notation, the first-derivative
coefficient is **+gamma**, the exponential coefficient is **-qP**, and
`lambda=-beta`. The minus sign in the exponential coefficient naturally leads
to real modified Bessel functions. Their argument is

\[
z=\frac{2\sqrt{qP}}{\beta}e^{-\beta t/2}.
\]

With beta=6e6/s, that argument becomes extraordinarily small within hundreds
of microseconds. Exponentially scaled `ive` and `kve` address large-argument
exponential growth/decay. They do not remove the small-argument power
singularity of `K_nu` or of the neighboring-order terms used in its derivative.
The concentration is finite while individual intermediate terms can overflow,
underflow, or cancel; plotting routines then omit NaNs.

See SciPy's [kve definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html)
and the NIST [modified-Bessel limiting forms](https://dlmf.nist.gov/10.30).
There is no Fourier-style normalization convention to fix this instability.

## The implemented analytical series

In dimensionless variables the same substitution gives

\[
u''+\nu u'=\eta e^{-s}u,\quad u(0)=1,\quad u'(0)=0,
\qquad w=\frac{u'}{\eta u}.
\]

Primes now refer to s. Write `u = sum_m eta**m f_m(s)` with `f_0=1` and

\[
f_m''+\nu f_m'=e^{-s}f_{m-1},\quad f_m(0)=f_m'(0)=0\quad(m\ge1).
\]

The Laplace transforms are rational:

\[
F_m(p)=\frac{1}{\prod_{j=0}^{m}(p+j)\prod_{j=0}^{m-1}(p+\nu+j)}.
\]

These positive analytical terms are evaluated together with a small matrix
exponential. The implementation groups all U states before the V states;
interleaving them activates a triangular exponential path that lost precision
in near-repeated-rate regression cases. The returned value is the ratio of
positive numerator and denominator sums. Each requested time is evaluated
directly; evaluation order, spacing, and duplicate timestamps do not change
the initial condition or require integrating between points.

For eta>0, truncation after order M obeys the uniform absolute bound

\[
|w-w_M|\le
\frac{\eta^M}{(M+1)!M!}
\frac{1}{1-\eta/[(M+1)(M+2)]},
\]

when the denominator is positive. `series_order(eta,tol)` selects a sufficient
order. A default `tol=1e-14` needs M=3 for the original q=1e9 example and M=18
at eta=10. The exact linear limit eta=0 uses `exprel` to cover gamma=beta and
gamma=0 continuously.

For a fixed finite expression throughout a bounded fit, choose an order at the
upper eta bound:

```python
eta_max = 10.0
model = SensitizedTTAModel(series_order=series_order(eta_max, 1e-14))
# Also constrain eta <= eta_max in the fitter's parameter bounds.
```

This is an analytical series evaluator with controlled truncation, evaluated
in floating-point arithmetic using SciPy's
[`expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
It is not an arbitrary-precision symbolic answer. The tolerance bounds series
truncation, **not total floating-point error**; independent validation found
up to about 1.7e-11 normalized absolute error at an extreme nearly zero-decay,
very-long-time corner. See [validation.md](validation.md) for domains, observed
errors, and the tested runtime. No late-time source cutoff or ODE fallback is
used in the production evaluator.

## Legacy TTA compatibility

The existing API is retained:

```python
legacy = TTAKineticModel()
C = legacy.solve(times_s, np.log([tau_r, tau_A, k2_effective]))
```

Its mapping is `beta=1/tau_r`, `gamma=1/tau_A`, and
`eta=k2_effective*tau_r`. This effective coefficient is `q*(P/beta)` in
inverse-time units, not automatically the physical bimolecular constant.
Legacy logarithmic coordinates cannot represent exact zero rates; use the new
linear model for those limits.

This update does not calibrate extinction coefficients, fit pump depletion,
account for singlet contamination, convolve an instrument response, or resolve
structural non-identifiability through numerical precision. Those are separate
scientific assumptions to assess with the experiment.
