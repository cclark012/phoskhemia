# PhosKhemia

PhosKhemia is a Python library for handling, processing, simulating, and fitting
spectroscopic data. Its main developed data workflow is transient absorption.
The repository is a prototype: coordinate handling, uncertainty assumptions,
and model parameter meanings should be explicit before using a result in a
scientific analysis.

## Install this checkout

The project declares Python 3.13 or newer. From the repository directory:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Optional MATLAB v7.3 support uses `python -m pip install -e '.[io]'`.
The optional ODRPACK backend uses `python -m pip install -e '.[odr]'`.
The default global fitter uses SciPy and does not require ODRPACK.

## Analytical sensitization and triplet annihilation

The new `SensitizedTTAModel` evaluates

\[
\frac{dn}{dt}=P e^{-\beta t}-\gamma n-qn^2,\qquad n(0)=0,
\]

through a positive analytical series and small matrix exponentials. It does
not integrate an ODE between observations. The fit parameters are the natural
values `[beta, gamma, eta]`, where `eta = q*P/beta**2`.

```python
import numpy as np
from phoskhemia.kinetics import SensitizedTTAModel

# ta is a TransientAbsorption with shape (time, wavelength).
# Its time axis is ta.y; the wavelength axis is ta.x.
# Choose data whose time axis is relative to excitation and is nonnegative.
model = SensitizedTTAModel(time_unit="s")
result = ta.fit_global_kinetics(
    model,
    beta0=np.array([6e6, 1e3, 8.33e-5]),
    noise=sigma_lambda,        # positive standard deviation per wavelength
    x_scale=np.array([6e6, 1e3, 1e-4]),
    maxit=200,
)

print(result.kinetics)
# result.traces:     (time, species), columns [annihilator, sensitizer]
# result.amplitudes: (wavelength, species), signed coefficients
fitted_signal = result.traces @ result.amplitudes.T
```

Run the self-contained, seeded example:

```bash
python examples/sensitized_tta_global_fit.py --plot docs/figures/tta-fit.png
```

See [the analytical model and fitting guide](docs/analytical-tta.md) for
normalization, physical parameter interpretation, time-unit conversion, fixed
parameters, numerical accuracy, and the limits of the observation model.
The [repository review](docs/repository-review.md) separates repaired bugs
from remaining scientific and architectural concerns.
[Validation notes](docs/validation.md) identify what was executed and what
still requires testing in the supported environments.

## Data and fitting conventions

- `TransientAbsorption` is two-dimensional: rows are times, columns are
  wavelengths. `.y` stores times and `.x` stores wavelengths.
- The default global objective is weighted least squares with fixed times.
  Noise is scalar or constant in time at each wavelength.
- `noise=None` looks for `ta.meta['noise_t0']`, then uses unity if unavailable.
  Meaningful reduced chi-square requires an appropriate measurement-noise scale.
- Spectral amplitudes are unconstrained in sign. They are projected by SVD at
  each kinetic trial, with no amplitude regularization by default.
- Ordinary rectangular slicing aligns known coordinate-dependent metadata.
  Operations that destroy the two-dimensional axis meaning return plain
  NumPy arrays rather than attaching stale axes.
- The historical `ta1 + ta2` operation still means combine/average.
  `np.add(ta1, ta2)` performs numerical addition. See the review before relying
  on this difference or on uncertainty metadata after arbitrary arithmetic.

## Compatibility and development status

The existing `TTAKineticModel` retains its logarithmic lifetime/effective-rate
API and now uses the analytical evaluator. The initial condition is anchored
at physical time zero, rather than the first requested observation.

Defaults and some return types have changed to repair incorrect behavior.
Read [CHANGELOG.md](CHANGELOG.md) before replacing an existing checkout.
Persistence is a report snapshot; complete model round-trip serialization,
calibrated IRF fitting, and joint errors-in-time fitting remain future work.

The source package version remains `0.1.6`; this is an unreleased working
update, not a published package release.

## Bugs and feedback

Please include a small reproducing example, data/axis shapes, parameter
coordinates and units, and the NumPy/SciPy versions when reporting a problem.

---

**AI usage disclosure:** This project has been developed with assistance from
AI workflow tools. Scientific assumptions and numerical results still require
review against the intended experiment. The tests and validation notes record
the checks included with this update.


### Experimental analytical-fit performance

For large TA time grids, `SensitizedTTAModel(evaluation="accelerated")`
uses an accuracy-checked interpolation of the analytical series rather than
performing one matrix exponential per time point. A compact weighted global
objective with optional CuPy support is available separately as an experimental
backend. See [`docs/performance-gpu.md`](docs/performance-gpu.md) for
limitations, examples, benchmark details, and the untested CUDA status.
