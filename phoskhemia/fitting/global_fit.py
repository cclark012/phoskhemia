from typing import Any
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import odr

from phoskhemia.kinetics.base import KineticModel
from phoskhemia.fitting.projections import project_amplitudes, propagate_kinetic_covariance
from phoskhemia.fitting.validation import compute_diagnostics
from phoskhemia.data.spectrum_handlers import TransientAbsorption
from phoskhemia.fitting.results import GlobalFitResult

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

def transform_params_and_cis(
        beta: np.ndarray,
        cov_beta: np.ndarray | None,
        *,
        parameterization: str,
        ci_sigma: float | None = 1.0,      # explicit z-score / # sigmas (default: 1)
        ci_level: float | None = None,     # two-sided level, e.g. 0.95
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

    if cov_beta is not None:
        cov_beta: NDArray[np.floating] | float = np.asarray(cov_beta, dtype=float)
        sigma: NDArray[np.floating] | float = np.sqrt(np.diag(cov_beta))
    else:
        sigma: None = None

    # Determine z
    if ci_sigma is None:
        z: float = _z_from_ci_level(ci_level) if ci_level is not None else None
    else:
        z: float = float(ci_sigma)

    if parameterization == "log":
        params: NDArray[np.floating] | float = np.exp(beta)

        if sigma is None or z is None:
            return params, None, None

        ci_low: NDArray[np.floating] | float = np.exp(beta - z * sigma)
        ci_high: NDArray[np.floating] | float = np.exp(beta + z * sigma)
        return params, ci_low, ci_high

    elif parameterization == "linear":
        params: NDArray[np.floating] = beta.copy()

        if sigma is None or z is None:
            return params, None, None

        ci_low: NDArray[np.floating] | float = params - z * sigma
        ci_high: NDArray[np.floating] | float = params + z * sigma
        return params, ci_low, ci_high

    else:
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
        noise: NDArray[np.floating] | None = None,
        lam: float = 1e-12,
        propagate_kinetic_uncertainty: bool = False,
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

    # Input preparation
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

    # Flatten observed data (wavelength-major)
    y_obs: NDArray[np.floating] = data.T.ravel()
    t_flat: NDArray[np.floating] = np.tile(times, n_wl)

    # ODR model (kinetics only)
    def odr_model(beta, t):
        try:
            traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
            fit: NDArray[np.floating] = np.empty((n_times, n_wl))

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
                odr_out.cov_beta,
                data[:, i],
                noise[i],
                lam,
            )

        amplitudes[i] = coeffs
        amp_errors[i] = np.sqrt(np.diag(cov))

    # Diagnostics
    # Reconstruct fitted signal for diagnostics
    fit: NDArray[np.floating] = traces @ amplitudes.T   # (n_times, n_wl)
    y_fit: NDArray[np.floating] = fit.T.ravel()
    noise_flat: NDArray[np.floating] = np.repeat(noise, n_times)
    param_type: str = kinetic_model.parameterization()
    params, ci_low, ci_high = transform_params_and_cis(
        beta=odr_out.beta,
        cov_beta=odr_out.cov_beta,
        parameterization=param_type
    )

    diagnostics: dict[str, float] = compute_diagnostics(
        y_obs=y_obs,
        y_fit=y_fit,
        noise=noise_flat,
        n_params=kinetic_model.n_params(),
    )

    # Package result
    kinetics: dict[str, float] = {
        name: float(val)
        for name, val in zip(
            kinetic_model.param_names(),
            params,
        )
    }
    kinetics_ci = None
    if ci_low is not None:
        kinetics_ci: dict[str, float] = {
            name + "_ci": (float(low), float(high))
            for name, low, high in zip(
                kinetic_model.param_names(),
                ci_low,
                ci_high,
            )
        }

    result: GlobalFitResult = GlobalFitResult(
        kinetics=kinetics,
        kinetics_ci=kinetics_ci,
        amplitudes=amplitudes,
        amplitude_errors=amp_errors,
        species=kinetic_model.species_names(),
        wavelengths=wl,
        diagnostics=diagnostics,
        backend={"odr": odr_out},
        _cache={
            "kinetic_model": kinetic_model,
            "beta": beta,
            "times": times,
            "traces": traces,
        },
    )

    return result



