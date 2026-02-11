from __future__ import annotations

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import odr

from phoskhemia.kinetics.base import KineticModel
from phoskhemia.fitting.projections import project_amplitudes, propagate_kinetic_covariance
from phoskhemia.fitting.validation import compute_diagnostics
from phoskhemia.fitting.results import GlobalFitResult, FitCache
if TYPE_CHECKING:
    from phoskhemia.data.spectrum_handlers import TransientAbsorption


def _z_from_ci_level(ci_level: float) -> float:
    """
    Convert two-sided CI level to z for a standard normal.
    Example: ci_level=0.95 -> z=1.959963984540054
    """

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1).")

    # Prefer SciPy if available (project already uses SciPy elsewhere); fall back otherwise.
    try:
        from scipy.special import ndtri  # inverse standard normal CDF
        return float(ndtri(0.5 + 0.5 * ci_level))
    except Exception:
        # Fallback: approximate inverse error function not in stdlib; require SciPy.
        raise ImportError("ci_level requires SciPy (scipy.special.ndtri).")

def _ci_level_from_sigma(ci_sigma: float) -> float:
    """
    Convert z/sigma to two-sided CI level.
    Example: ci_sigma=2 -> ci_level=0.9544997361036416
    """

    if not ci_sigma >= 0:
        raise ValueError("ci_sigma must be positive.")

    try:
        from scipy.special import ndtr
        return float(2 * ndtr(ci_sigma) - 1)
    except Exception:
        raise ImportError("ci_sigma requires SciPy (scipy.special.ndtr).")

def transform_params_and_cis(
        beta: np.ndarray,
        cov_beta: np.ndarray | None,
        *,
        parameterization: str,
        ci_sigma: float | None = None,
        ci_level: float | None = None,
    ) -> tuple[
        NDArray[np.floating], 
        NDArray[np.floating] | None, 
        NDArray[np.floating] | None
    ]:
    """
    Transform fitted parameters and compute confidence intervals.

    Parameters
    ----------
    beta : ndarray
        Fitted parameters (log or linear)
    cov_beta : ndarray or None
        Covariance of fitted parameters (in beta-space)
    parameterization : {"log", "linear"}
        Parameterization used by the model
    ci_sigma : float or None
        z-score / # sigmas for CI half-width. If explicitly provided, overrides ci_level.
    ci_level : float or None
        Two-sided confidence level in (0, 1), converted to z.

    Returns
    -------
    params : NDArray[np.floating]
        Parameters in natural space
    ci_low : NDArray[np.floating] | None
        Lower confidence bound
    ci_high : NDArray[np.floating] | None
        Upper confidence bound
    """

    beta: NDArray[np.floating] = np.asarray(beta, dtype=float)

    sigma: None = None
    if cov_beta is not None:
        cov_beta: NDArray[np.floating] = np.asarray(cov_beta, dtype=float)
        sigma: NDArray[np.floating] = np.sqrt(np.diag(cov_beta))

    # Determine z
    if ci_sigma is not None:
        z: float | None = float(ci_sigma)
    elif ci_level is not None:
        z = _z_from_ci_level(ci_level)
    else:
        z = 1.0  # default: 1σ

    if parameterization == "log":
        params: NDArray[np.floating] = np.exp(beta)
        if sigma is None:
            return params, None, None
        ci_low: NDArray[np.floating] = np.exp(beta - z * sigma)
        ci_high: NDArray[np.floating] = np.exp(beta + z * sigma)
        return params, ci_low, ci_high

    if parameterization == "linear":
        params: NDArray[np.floating] = beta.copy()
        if sigma is None:
            return params, None, None
        ci_low: NDArray[np.floating] = params - z * sigma
        ci_high: NDArray[np.floating] = params + z * sigma
        return params, ci_low, ci_high

    warnings.warn(
        f"Unknown parameterization '{parameterization}'. "
        "Assuming linear parameters without CIs."
    )
    return beta.copy(), None, None


def fit_global_kinetics(
        arr: TransientAbsorption,
        kinetic_model: KineticModel,
        beta0: NDArray[np.floating],
        *,
        noise: NDArray[np.floating] | float | int | None = None,
        lam: float = 1e-12,
        propagate_kinetic_uncertainty: bool = False,
        ci_sigma: float | None = None,
        ci_level: float | None = None,
        debug: bool = False,
    ) -> GlobalFitResult:
    """
    Perform a global kinetic fit using variable projection.

    Parameters
    ----------
    kinetic_model : KineticModel
        Kinetic model instance
    beta0 : ndarray
        Initial guesses for kinetic parameters (log-space)
    noise : ndarray or None
        Per-wavelength noise σ(λ)
    lam : float
        Tikhonov regularization strength
    propagate_kinetic_uncertainty : bool
        Propagate kinetic covariance into amplitude uncertainties
    debug : bool
        Print diagnostic messages on failures

    Returns
    -------
    result : GlobalFitResult
        Structured fit result
    """

    # Input preparation and validation
    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    times: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(arr.x, dtype=float)
    beta0: NDArray[np.floating] = np.asarray(beta0, dtype=float)

    n_times: int
    n_wl: int
    n_times, n_wl = data.shape

    if beta0.size != kinetic_model.n_params():
        raise ValueError("beta0 length does not match kinetic model")

    # Default to unity noise if none provided
    if noise is None:
        noise: NDArray[np.floating] = np.ones(n_wl, dtype=float)
    elif isinstance(noise, float | int):
        noise: NDArray[np.floating] = noise * np.ones(n_wl, dtype=float)
    else:
        noise: NDArray[np.floating] = np.asarray(noise, dtype=float)
        if noise.size != n_wl:
            raise ValueError("noise length must match number of wavelengths")

    # Flatten observed data (wavelength-major) for fitting
    y_obs: NDArray[np.floating] = data.T.ravel()
    t_flat: NDArray[np.floating] = np.tile(times, n_wl)

    # ODR model with separable kinetics and amplitudes
    def odr_model(beta, t):
        try:
            # Compute the kinetic model
            traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
            fit: NDArray[np.floating] = np.empty((n_times, n_wl))

            # Compute the wavelength-dependent amplitudes separately.
            for i in range(n_wl):
                coeffs: NDArray[np.floating]
                coeffs, _, _ = project_amplitudes(
                    traces,
                    data[:, i],
                    noise[i],
                    lam,
                )
                fit[:, i] = traces @ coeffs

            return fit.T.ravel()

        except Exception as exc:
            if debug:
                print("ODR model failure:", exc)
            return np.full_like(y_obs, 1e30)

    # Run ODR
    odr_data: odr.RealData = odr.RealData(t_flat, y_obs)
    odr_mod: odr.Model = odr.Model(odr_model)
    odr_out: odr.Output = odr.ODR(odr_data, odr_mod, beta0=beta0).run()

    # Post-fit reconstruction
    beta: NDArray[np.floating] = odr_out.beta
    traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
    n_species: int = traces.shape[1]

    amplitudes: NDArray[np.floating] = np.empty((n_wl, n_species))
    amp_errors: NDArray[np.floating] = np.empty_like(amplitudes)
    # Scale covariance based on residual variance
    cov_beta: NDArray[np.floating] = odr_out.cov_beta * odr_out.res_var

    # Calculate amplitudes and errors
    for i in range(n_wl):
        coeffs: NDArray[np.floating]
        cov: NDArray[np.floating]
        coeffs, cov, _ = project_amplitudes(
            traces,
            data[:, i],
            noise[i],
            lam,
        )

        if propagate_kinetic_uncertainty:
            cov += propagate_kinetic_covariance(
                kinetic_model,
                times,
                beta,
                coeffs,
                cov_beta,
                data[:, i],
                noise[i],
                lam,
            )

        amplitudes[i] = coeffs
        amp_errors[i] = np.sqrt(np.diag(cov))

    # Reconstruct fitted signal for validation
    fit: NDArray[np.floating] = traces @ amplitudes.T   # (n_times, n_wl)
    y_fit: NDArray[np.floating] = fit.T.ravel()
    noise_flat: NDArray[np.floating] = np.repeat(noise, n_times)
    param_type: str = kinetic_model.parameterization()

    # Determine desired confidence interval (1σ or ≈68%)
    if ci_sigma is not None:
        ci_sigma = float(ci_sigma)
        ci_level = _ci_level_from_sigma(ci_sigma)
    elif ci_level is not None:
        ci_level = float(ci_level)
        ci_sigma = _z_from_ci_level(ci_level)
    else:
        ci_sigma = 1.0  # default: 1σ
        ci_level = _ci_level_from_sigma(ci_sigma)

    params, ci_low, ci_high = transform_params_and_cis(
        beta=odr_out.beta,
        cov_beta=cov_beta,
        parameterization=param_type,
        ci_sigma=ci_sigma,
    )

    # Compute statistical tests (AIC, BIC, χ², etc.)
    diagnostics: dict[str, float] = compute_diagnostics(
        y_obs=y_obs,
        y_fit=y_fit,
        noise=noise_flat,
        n_params=kinetic_model.n_params(),
    )

    raw_names: str | list[str] = kinetic_model.param_names()
    names: list[str] = (
        [raw_names] 
        if isinstance(raw_names, str) 
        else list(raw_names)
    )
    raw_species: str | list[str] = kinetic_model.species_names()
    species: list[str] = (
        [raw_species]
        if isinstance(raw_species, str)
        else list(raw_species)
    )
    # Package result
    kinetics: dict[str, float] = {
        name: float(val)
        for name, val in zip(
            names,
            params,
        )
    }
    kinetics_ci: dict[str, tuple[float, float]] | None = None
    if ci_low is not None:
        kinetics_ci = {
            name: (float(low), float(high))
            for name, low, high in zip(
                names,
                ci_low,
                ci_high,
            )
        }

    result: GlobalFitResult = GlobalFitResult(
        kinetics=kinetics,
        kinetics_ci=kinetics_ci,
        amplitudes=amplitudes,
        amplitude_errors=amp_errors,
        species=species,
        wavelengths=wl,
        times=times,
        diagnostics=diagnostics,
        backend={"odr": odr_out},
        _cache = {
            "lam": lam,
            "noise": noise,
            "kinetic_model": kinetic_model,
            "beta": odr_out.beta,
            "cov_beta": cov_beta,
            "ci_sigma": ci_sigma,
            "ci_level": ci_level,
            "parameterization": param_type,
            "traces": traces,
        }
    )

    return result



