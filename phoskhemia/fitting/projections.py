"""Stable variable projection of signed spectra onto a kinetic basis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel


@dataclass(frozen=True)
class AmplitudeProjection:
    """A shared-SVD projection for all wavelengths.

    Coefficients are species by wavelength; public fit results transpose them
    to retain the repository's wavelength-by-species layout.
    """
    coefficients: NDArray[np.floating]
    residuals: NDArray[np.floating]
    singular_values: NDArray[np.floating]
    right_vectors: NDArray[np.floating]
    rank: int
    effective_parameters: float
    noise: NDArray[np.floating]
    lam: float


def project_spectra(traces, data, noise=None, lam=0.0) -> AmplitudeProjection:
    """Minimize sum(((Y-C@A)/sigma)**2) + lam*sum(A**2).

    Noise is constant in time at each wavelength. A single SVD of C is reused
    across wavelengths; normal equations are never formed. With lam=0 the
    scalar wavelength weights cancel in the conditional amplitude estimates,
    but they still affect the outer kinetic objective.
    """
    C, Y = np.asarray(traces, dtype=float), np.asarray(data, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2 or C.shape[0] == 0 or C.shape[1] == 0:
        raise ValueError("traces must be nonempty with shape (n_times, n_species)")
    if Y.ndim != 2 or Y.shape[0] != C.shape[0] or Y.shape[1] == 0:
        raise ValueError("data must have shape (n_times, n_wavelengths)")
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(Y)):
        raise ValueError("traces and data must be finite")
    sigma = np.ones(Y.shape[1]) if noise is None else np.asarray(noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(Y.shape[1], float(sigma))
    if sigma.shape != (Y.shape[1],) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("noise must be a positive finite scalar or wavelength vector")
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    cutoff = np.finfo(float).eps * max(C.shape) * (s[0] if s.size else 0)
    keep = s > cutoff
    rank = int(np.count_nonzero(keep))
    if lam == 0:
        factors = np.zeros_like(s)
        factors[keep] = 1.0 / s[keep]
        coeffs = Vt.T @ (factors[:, None] * (U.T @ Y))
        effective = float(rank * Y.shape[1])
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = s[:, None] ** 2 + lam * sigma[None, :] ** 2
            factors = np.divide(s[:, None], denominator, out=np.zeros_like(denominator), where=denominator > 0)
            coeffs = Vt.T @ (factors * (U.T @ Y))
            effective = float(np.sum(np.divide(s[:, None] ** 2, denominator, out=np.zeros_like(denominator), where=denominator > 0)))
    residuals = Y - C @ coeffs
    if not np.all(np.isfinite(coeffs)) or not np.all(np.isfinite(residuals)):
        raise FloatingPointError("amplitude projection produced nonfinite values")
    return AmplitudeProjection(coeffs, residuals, s, Vt, rank, effective, sigma.copy(), lam)


def project_amplitudes(traces, data, noise=None, lam=0.0, *, absolute_sigma=False):
    """Project one wavelength and return coefficients, covariance, residuals.

    Covariance is conditional on the kinetic parameters. For unregularized,
    full-column-rank fits it is sigma**2*(C.T C)^(-1), optionally multiplied by
    the local reduced residual variance when absolute_sigma=False. A deficient
    basis has non-identifiable individual amplitudes and returns NaN covariance.
    Ridge covariance uses the frequentist sandwich expression, conditional on
    the chosen penalty; it does not include regularization bias.
    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 1:
        raise ValueError("data must be one-dimensional for one wavelength")
    projection = project_spectra(traces, y[:, None], noise, lam)
    coeffs, residuals = projection.coefficients[:, 0], projection.residuals[:, 0]
    n_species = coeffs.size
    cov = np.full((n_species, n_species), np.nan)
    dof = y.size - projection.effective_parameters
    if projection.rank == n_species and (absolute_sigma or dof > 0):
        s, Vt, sigma = projection.singular_values, projection.right_vectors, projection.noise[0]
        gain = 1.0 / s if lam == 0 else s / (s * s + lam * sigma * sigma)
        inverse = Vt.T * gain
        cov = sigma * sigma * (inverse @ inverse.T)
        if not absolute_sigma:
            cov *= float(np.sum((residuals / sigma) ** 2) / dof)
    return coeffs, cov, residuals


def propagate_kinetic_covariance(
    kinetic_model: KineticModel, times, beta, coeffs, cov_beta, data, noise, lam,
    eps=1e-6,
):
    """First-order propagation of kinetic covariance into fitted amplitudes.

    Derivatives are with respect to the model's optimizer coordinates. Steps
    are relative to each nonzero coordinate, respect model bounds, and avoid
    the previous unit-dependent floor of 1 for small rates in nanoseconds.
    """
    times, beta = np.asarray(times, dtype=float), np.asarray(beta, dtype=float)
    coeffs, cov_beta = np.asarray(coeffs, dtype=float), np.asarray(cov_beta, dtype=float)
    if cov_beta.shape != (beta.size, beta.size):
        raise ValueError("cov_beta shape mismatch")
    if not np.all(np.isfinite(cov_beta)):
        return np.full((coeffs.size, coeffs.size), np.nan)
    lower, upper = kinetic_model.parameter_bounds()
    jac = np.zeros((coeffs.size, beta.size))
    for k in range(beta.size):
        if not np.any(cov_beta[k]):
            continue
        scale = abs(beta[k])
        if scale == 0:
            scale = max(float(np.sqrt(max(cov_beta[k, k], 0))), np.finfo(float).eps)
        step = max(eps * scale, abs(np.spacing(beta[k])))
        plus, minus = min(step, upper[k] - beta[k]), min(step, beta[k] - lower[k])
        if plus <= 0 and minus <= 0:
            continue
        b = beta.copy()
        if plus > 0:
            b[k] += plus
            cplus = project_amplitudes(kinetic_model.solve(times, b), data, noise, lam)[0]
        else:
            cplus = coeffs
        if minus > 0:
            b[k] = beta[k] - minus
            cminus = project_amplitudes(kinetic_model.solve(times, b), data, noise, lam)[0]
        else:
            cminus = coeffs
        jac[:, k] = (cplus - cminus) / (max(plus, 0) + max(minus, 0))
    return jac @ cov_beta @ jac.T
