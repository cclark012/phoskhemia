"""Memory-light variable projection for time-independent wavelength noise.

This experimental objective avoids building a time-by-wavelength residual
matrix on every nonlinear iteration. The kinetic basis remains on the CPU;
optionally, the data and its cross-products reside on a CUDA GPU.

Only unregularized, fixed-time, Gaussian weighted least squares is supported.
The scalar objective is suitable for scipy.optimize.minimize. It does not
replace the existing uncertainty calculations or full GlobalFitResult API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from phoskhemia.kinetics.base import KineticModel


@dataclass(frozen=True)
class CompactFitResult:
    """Result of the experimental scalar-objective global fitting method.

    Attributes
    ----------
    parameters : NDArray[np.float64]
        Fitted kinetic parameters in the model's optimizer coordinates.
    amplitudes : NDArray[np.float64]
        Signed amplitude matrix with shape (n_wavelengths, n_species).
    chi_squared : float
        Sum of squared residuals divided by wavelength noise variances.
    success : bool
        Whether the scalar optimizer reported convergence.
    message : str
        Optimizer's termination explanation.
    evaluations : int
        Number of scalar function evaluations reported by the optimizer.
    """

    parameters: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    chi_squared: float
    success: bool
    message: str
    evaluations: int


class WeightedProjectionObjective:
    """Compute the exact profiled weighted sum of squares without residuals.

    Parameters
    ----------
    data : ArrayLike
        Transient-absorption matrix with shape (n_times, n_wavelengths).
    sigma : ArrayLike
        Positive standard deviations, one for each wavelength and constant
        across acquisition time.
    device : Literal["cpu", "cuda"], default="cpu"
        Data cross-products use NumPy or CuPy. The CUDA path is optional and
        requires compatible CUDA hardware and an installed CuPy distribution.

    Notes
    -----
    A reduced SVD of the kinetic basis supplies orthonormal columns Q.
    The unregularized least-squares objective then equals

        ||Y / sigma||_F^2 - ||(Q.T @ Y) / sigma||_F^2.

    Q has at most one column per independently identifiable species. The
    data remain on the requested device, while the kinetic basis is made
    on CPU. CUDA execution has not been tested on physical GPU hardware.
    """

    def __init__(
        self,
        data: ArrayLike,
        sigma: ArrayLike,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:

        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

        raw = np.asarray(data)
        noise = np.asarray(sigma, dtype=np.float64)

        if raw.ndim != 2 or not all(raw.shape):
            raise ValueError("data must have shape (n_times, n_wavelengths)")
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise ValueError("data must be real numeric values")
        if noise.ndim != 1 or noise.shape[0] != raw.shape[1]:
            raise ValueError("sigma must contain one value per wavelength")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0):
            raise ValueError("sigma must contain finite positive values")
        if not np.all(np.isfinite(raw)):
            raise ValueError("data must contain finite values")

        if device == "cuda":
            try:
                import cupy as xp
            except ImportError as exc:
                raise ImportError(
                    "CuPy is needed for device='cuda'. Install the wheel "
                    "matching your CUDA runtime."
                ) from exc
        else:
            xp = np

        self.device = device
        self.xp = xp
        self.data = xp.asarray(raw, dtype=xp.float64)
        self.sigma = xp.asarray(noise, dtype=xp.float64)
        self.n_times = raw.shape[0]
        self.n_wavelengths = raw.shape[1]

        # Contract the large data matrix once. No full-size weighted copy.
        columns_squared = xp.einsum("ij,ij->j", self.data, self.data)
        norm_squared = xp.sum(columns_squared / (self.sigma**2))
        self.norm_squared = float(norm_squared)

    def chi_squared(self, traces: ArrayLike) -> float:
        """Return profiled weighted residual sum of squares for given traces.

        Parameters
        ----------
        traces : ArrayLike
            Kinetic basis with shape (n_times, n_species).

        Returns
        -------
        float
            Minimized weighted sum of squared residuals over spectral
            amplitudes, up to floating-point cancellation in the difference.
        """
        basis = np.asarray(traces, dtype=np.float64)

        if basis.ndim != 2 or basis.shape[0] != self.n_times:
            raise ValueError("traces must have shape (n_times, n_species)")
        if basis.shape[1] == 0 or not np.all(np.isfinite(basis)):
            raise ValueError("traces must be finite and nonempty")

        left, singular, _ = np.linalg.svd(basis, full_matrices=False)
        threshold = np.finfo(np.float64).eps * max(basis.shape) * singular[0]
        rank = int(np.count_nonzero(singular > threshold))

        if rank == 0:
            return self.norm_squared

        orthogonal = self.xp.asarray(left[:, :rank])
        overlap = orthogonal.T @ self.data
        explained = self.xp.sum((overlap / self.sigma[None, :]) ** 2)
        value = self.norm_squared - float(explained)

        # The true objective is nonnegative. A negative result larger than
        # roundoff signals catastrophic cancellation or another error.
        if value < -1e-11 * self.norm_squared:
            raise FloatingPointError("Profiled objective is unexpectedly negative")

        return max(value, 0.0)


def fit_compact_global(
    times: ArrayLike,
    data: ArrayLike,
    sigma: ArrayLike,
    kinetic_model: KineticModel,
    initial_parameters: ArrayLike,
    *,
    parameter_scales: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    max_iterations: int = 100,
) -> CompactFitResult:
    """Fit shared kinetics by minimizing a scalar profiled objective.

    Parameters
    ----------
    times : ArrayLike
        Times in the reciprocal units of the supplied rate parameters.
    data : ArrayLike
        Array with shape (n_times, n_wavelengths).
    sigma : ArrayLike
        One positive noise standard deviation per wavelength.
    kinetic_model : KineticModel
        Model returning a kinetic basis with shape (n_times, n_species).
    initial_parameters : ArrayLike
        Starting values in the model's optimizer coordinates.
    parameter_scales : ArrayLike
        Positive scalings for dimensionless optimization coordinates.
    bounds : tuple[ArrayLike, ArrayLike] | None, default=None
        Optimizer-coordinate lower and upper bounds. Defaults to model bounds.
    device : Literal["cpu", "cuda"], default="cpu"
        Array backend used for profile cross-products.
    max_iterations : int, default=100
        Upper limit of quasi-Newton optimizer iterations.

    Returns
    -------
    CompactFitResult
        Fitted kinetic parameters and amplitude spectra. Covariances are not
        estimated by this prototype; use the full fitting interface for CIs.
    """
    t = np.asarray(times, dtype=np.float64)
    guess = np.asarray(initial_parameters, dtype=np.float64)
    scales = np.asarray(parameter_scales, dtype=np.float64)

    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a finite nonempty vector")
    if guess.shape != (kinetic_model.n_params(),):
        raise ValueError("initial_parameters must match the model's parameter count")
    if scales.shape != guess.shape or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("parameter_scales must be finite positive values")
    if not np.all(np.isfinite(guess)):
        raise ValueError("initial_parameters must be finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    lower, upper = kinetic_model.parameter_bounds() if bounds is None else bounds
    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), guess.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), guess.shape)

    if np.any(lower >= upper) or np.any(guess < lower) or np.any(guess > upper):
        raise ValueError("invalid bounds or initial parameters")

    objective = WeightedProjectionObjective(data, sigma, device=device)
    if t.size != objective.n_times:
        raise ValueError("data and times must agree on the number of rows")

    def scalar_objective(scaled_parameters: NDArray[np.float64]) -> float:
        parameters = scaled_parameters * scales
        traces = kinetic_model.solve(t, parameters)
        return objective.chi_squared(traces)

    optimizer_result = minimize(
        scalar_objective,
        guess / scales,
        method="L-BFGS-B",
        jac="3-point",
        bounds=list(zip(lower / scales, upper / scales)),
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )

    parameters = optimizer_result.x * scales
    traces = np.asarray(kinetic_model.solve(t, parameters), dtype=np.float64)
    amplitudes = np.linalg.lstsq(traces, np.asarray(data), rcond=None)[0].T

    return CompactFitResult(
        parameters=parameters,
        amplitudes=amplitudes,
        chi_squared=float(optimizer_result.fun),
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        evaluations=int(optimizer_result.nfev),
    )
