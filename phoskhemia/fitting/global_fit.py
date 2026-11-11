from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
from phoskhemia.fitting.solvers import (
    SolverResult,
    ODRPackSolver,
    LeastSquaresSolver,
    Solver,
    get_solver,
    weights_from_sigma,
)

from phoskhemia.kinetics.base import KineticModel
from phoskhemia.fitting.projections import (
    project_amplitudes, project_spectra, propagate_kinetic_covariance,
)
from phoskhemia.fitting.validation import compute_diagnostics
from phoskhemia.fitting.results import GlobalFitResult
if TYPE_CHECKING:
    from phoskhemia.data.spectrum2d import TransientAbsorption


def _z_from_ci_level(ci_level: float) -> float:
    """
    Convert two-sided CI level to z for a standard normal.
    Example: ci_level=0.95 -> z=1.959963984540054
    """

    if not np.isfinite(ci_level) or not (0.0 < ci_level < 1.0):
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

    if not np.isfinite(ci_sigma) or ci_sigma < 0:
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

def normalize_ci(
        ci_sigma: float | None=None,
        ci_level: float | None=None,
    ) -> tuple[float, float]:

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
    
    return ci_sigma, ci_level

def _basis(model, times, beta):
    raw = model.solve(times, beta)
    if np.iscomplexobj(raw):
        raise ValueError("kinetic_model.solve must return real basis functions")
    traces = np.asarray(raw, dtype=float)
    if traces.ndim == 1:
        traces = traces[:, None]
    if traces.ndim != 2 or traces.shape[0] != times.size or traces.shape[1] == 0:
        raise ValueError("kinetic_model.solve must return (n_times, n_species)")
    if not np.all(np.isfinite(traces)):
        raise ValueError("kinetic_model.solve returned nonfinite values")
    return traces


def package_result(
    sol, kinetic_model, times, wl, data, y_obs, noise, lam,
    ci_sigma, ci_level, propagate_kinetic_uncertainty, *,
    objective_noise=None, absolute_sigma=False,
) -> GlobalFitResult:
    """Reconstruct a fit and report total parameter counts and uncertainty."""
    objective_noise = noise if objective_noise is None else objective_noise
    beta = np.asarray(sol.beta, dtype=float)
    traces = _basis(kinetic_model, times, beta)
    projection = project_spectra(traces, data, objective_noise, lam)
    amplitudes = projection.coefficients.T
    n_free = int(sol.info.get("n_free_kinetics", kinetic_model.n_params()))
    n_parameters = n_free + projection.effective_parameters
    dof = data.size - n_parameters
    weighted_rss = float(np.sum((projection.residuals / objective_noise) ** 2))
    variance_scale = 1.0 if absolute_sigma else (weighted_rss / dof if dof > 0 else np.nan)
    cov_beta = sol.cov_beta
    amp_errors = np.empty_like(amplitudes)
    for i, coeffs in enumerate(amplitudes):
        _, cov, _ = project_amplitudes(
            traces, data[:, i], objective_noise[i], lam, absolute_sigma=True,
        )
        cov *= variance_scale
        if propagate_kinetic_uncertainty and cov_beta is not None:
            cov += propagate_kinetic_covariance(
                kinetic_model, times, beta, coeffs, cov_beta, data[:, i],
                objective_noise[i], lam,
            )
        amp_errors[i] = np.sqrt(np.maximum(np.diag(cov), 0))

    y_fit = (traces @ amplitudes.T).T.ravel()
    diagnostics = compute_diagnostics(
        y_obs=y_obs, y_fit=y_fit,
        noise=np.repeat(noise, times.size), n_params=n_parameters,
    )
    diagnostics.update({
        "n_kinetic_parameters": float(n_free),
        "n_amplitude_parameters": projection.effective_parameters,
        "amplitude_rank": float(projection.rank),
        "penalty": float(lam * np.sum(amplitudes ** 2)),
    })
    relative_weights = noise / objective_noise
    likelihood_matches_objective = bool(np.all(relative_weights == relative_weights[0]))
    if not likelihood_matches_objective:
        # Chi-square remains a descriptive check using supplied sigmas, but
        # an unweighted optimum is not the heteroscedastic maximum likelihood
        # estimate required for these model-selection criteria.
        for criterion in ("AIC", "AICc", "BIC"):
            diagnostics[criterion] = float("nan")
    param_type = kinetic_model.parameterization()
    ci_sigma, ci_level = normalize_ci(ci_sigma, ci_level)
    params, ci_low, ci_high = transform_params_and_cis(
        beta, cov_beta, parameterization=param_type, ci_sigma=ci_sigma,
    )
    raw_names, raw_species = kinetic_model.param_names(), kinetic_model.species_names()
    names = [raw_names] if isinstance(raw_names, str) else list(raw_names)
    species = [raw_species] if isinstance(raw_species, str) else list(raw_species)
    if len(names) != beta.size or len(set(names)) != len(names):
        raise ValueError("model parameter names must be unique and match n_params")
    if len(species) != traces.shape[1]:
        raise ValueError("species_names must match the number of kinetic basis columns")
    kinetics_ci = None if ci_low is None else {
        name: (float(low), float(high))
        for name, low, high in zip(names, ci_low, ci_high)
    }
    return GlobalFitResult(
        kinetics={name: float(value) for name, value in zip(names, params)},
        kinetics_ci=kinetics_ci,
        amplitudes=amplitudes,
        amplitude_errors=amp_errors,
        species=species,
        wavelengths=wl.copy(),
        times=times.copy(),
        diagnostics=diagnostics,
        backend={str(sol.info.get("backend", "unknown")): sol.info.get("raw", sol.info)},
        _cache={
            "lam": lam, "noise": noise.copy(), "objective_noise": objective_noise.copy(),
            "kinetic_model": kinetic_model, "beta": beta.copy(), "cov_beta": cov_beta,
            "ci_sigma": ci_sigma, "ci_level": ci_level, "parameterization": param_type,
            "traces": traces, "solver_info": sol.info, "absolute_sigma": absolute_sigma,
            "amplitude_rank": projection.rank,
            "amplitude_errors_include_kinetics": bool(propagate_kinetic_uncertainty and cov_beta is not None),
            "time_error_model": "fixed", "regularized_diagnostics_approximate": bool(lam),
            "likelihood_matches_objective": likelihood_matches_objective,
        },
    )


def fit_global_kinetics(
    arr: TransientAbsorption, kinetic_model: KineticModel, beta0,
    *, noise=None, lam: float = 0.0, propagate_kinetic_uncertainty: bool = False,
    ci_sigma: float | None = None, ci_level: float | None = None,
    debug: bool = False, solver: str | Solver = "least_squares", maxit: int = 50,
    use_noise_as_sy: bool = True, sx=None, scope_ppm=None, scope_jitter=None,
    absolute_sigma: bool = False, **kwargs,
) -> GlobalFitResult:
    """Fit shared kinetic parameters and wavelength-dependent signed amplitudes.

    ``arr`` has shape (time, wavelength). ``beta0`` is in the coordinates
    accepted by ``kinetic_model.solve``: natural [beta, gamma, eta] for
    SensitizedTTAModel; logarithmic coordinates for legacy log models.

    A bounded SciPy least-squares optimizer minimizes residuals divided by the
    per-wavelength standard deviations, projecting all amplitudes by SVD at
    every kinetic trial. ``noise`` is a positive scalar or wavelength vector;
    if omitted, metadata ``noise_t0`` is used, then unity as a last fallback.
    ``use_noise_as_sy=False`` explicitly selects an unweighted objective.

    Model bounds apply unless ``bounds=(lower, upper)`` is supplied. Both
    bounds and boolean ``fix_beta`` refer to optimizer coordinates. ``maxit``
    limits function evaluations. SciPy options such as x_scale, ftol, xtol,
    and gtol can be supplied via keyword arguments. Only quadratic loss is
    supported because amplitudes are projected with quadratic least squares.

    With ``absolute_sigma=False``, covariance is scaled by the fitted residual
    variance using degrees of freedom that include projected amplitudes.
    True treats the supplied objective sigmas as known. Confidence intervals
    are local linear approximations; boundary estimates and weakly identified
    parameters require profile or bootstrap analysis. Rank-deficient kinetic
    fits return no kinetic covariance.

    ``lam>0`` adds a penalty on raw amplitudes to both inner and outer
    objectives; effective parameter counts are then approximate and kinetic
    covariance is withheld. The default is unregularized.

    Timestamps are fixed. The optional odrpack backend is available in OLS
    mode only. Errors-in-time fitting is not implemented for this profiled
    multichannel objective; supplying sx or scope uncertainty raises an
    explicit error instead of silently ignoring adjusted timestamps. Metadata
    scope settings do not automatically change the fit objective.

    ``debug`` is retained for compatibility. Model exceptions always propagate
    with their actual cause; they are never replaced by constant residuals.
    """
    if sx is not None or scope_ppm is not None or scope_jitter is not None:
        raise NotImplementedError(
            "Global variable projection currently fits fixed timestamps. "
            "sx/scope uncertainty requires a joint errors-in-time model."
        )
    if kwargs.get("weight_x") is not None:
        raise NotImplementedError("Global fitting currently requires fixed timestamps")
    kwargs.pop("weight_x", None)
    if kwargs.get("fix_x") is not None and not np.all(kwargs["fix_x"]):
        raise NotImplementedError("Global fitting currently requires fixed timestamps")
    kwargs.pop("fix_x", None)
    if "task" in kwargs and kwargs.pop("task") != "OLS":
        raise NotImplementedError("The global odrpack backend supports task='OLS' only")
    if kwargs.get("loss", "linear") != "linear":
        raise ValueError("Global variable projection currently supports loss='linear' only")
    if np.iscomplexobj(arr) or np.iscomplexobj(beta0):
        raise ValueError("data and beta0 must be real")
    data = np.asarray(arr, dtype=float)
    times, wl = np.asarray(arr.y, dtype=float), np.asarray(arr.x, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError("arr must have nonempty shape (n_times, n_wavelengths)")
    if times.shape != (data.shape[0],) or wl.shape != (data.shape[1],):
        raise ValueError("time and wavelength axes must match the data shape")
    if not all(np.all(np.isfinite(a)) for a in (data, times, wl)):
        raise ValueError("data and axes must be finite; exclude invalid observations before fitting")
    if beta0.shape != (kinetic_model.n_params(),) or not np.all(np.isfinite(beta0)):
        raise ValueError("beta0 must be a finite vector matching kinetic_model.n_params()")
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    normalize_ci(ci_sigma, ci_level)  # validate before optimization
    if noise is None:
        meta = getattr(arr, "meta", None)
        noise = meta.get("noise_t0") if meta is not None else None
    if noise is None:
        noise = np.ones(wl.size)
    if np.iscomplexobj(noise):
        raise ValueError("noise must be real standard deviations")
    noise = np.asarray(noise, dtype=float)
    if noise.ndim == 0:
        noise = np.full(wl.size, float(noise))
    if noise.shape != (wl.size,):
        raise ValueError("noise must be scalar or shape (n_wavelengths,); time-dependent noise is not supported")
    if np.any(~np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("noise standard deviations must be finite and strictly positive")
    objective_noise = noise.copy() if use_noise_as_sy else np.ones_like(noise)
    initial_traces = _basis(kinetic_model, times, beta0)
    n_species = initial_traces.shape[1]
    kwargs.setdefault("bounds", kinetic_model.parameter_bounds())
    solver_obj = get_solver(solver.casefold().strip()) if isinstance(solver, str) else solver

    y_obs, t_flat = data.T.ravel(), np.tile(times, wl.size)
    objective_y = y_obs
    objective_t = t_flat
    weight_y = weights_from_sigma(np.repeat(objective_noise, times.size))
    if lam:
        if not isinstance(solver_obj, LeastSquaresSolver):
            raise NotImplementedError("Penalized global fits currently require solver='least_squares'")
        # Pseudo-observations impose the same raw-amplitude penalty optimized
        # by the inner SVD solve. They do not count as measured observations.
        n_penalty = n_species * wl.size
        objective_y = np.concatenate([y_obs, np.zeros(n_penalty)])
        objective_t = np.concatenate([t_flat, np.zeros(n_penalty)])
        weight_y = np.concatenate([weight_y, np.ones(n_penalty)])

    def projection_at(beta):
        traces = _basis(kinetic_model, times, beta)
        if traces.shape[1] != n_species:
            raise ValueError("the number of kinetic basis columns must remain constant")
        return traces, project_spectra(traces, data, objective_noise, lam)

    def f(t, beta):
        if np.shape(t) != objective_t.shape or not np.array_equal(t, objective_t):
            raise NotImplementedError("This global objective requires fixed timestamps")
        traces, projection = projection_at(beta)
        fit = (traces @ projection.coefficients).T.ravel()
        if lam:
            fit = np.concatenate([fit, np.sqrt(lam) * projection.coefficients.T.ravel()])
        return fit

    solver_options = dict(kwargs)
    if isinstance(solver_obj, LeastSquaresSolver):
        solver_options.update(
            absolute_sigma=absolute_sigma,
            n_projected=lambda beta: projection_at(beta)[1].effective_parameters,
            n_observations=data.size,
            covariance=(lam == 0),
        )
    elif isinstance(solver_obj, ODRPackSolver):
        solver_options.update(task="OLS", fix_x=np.ones_like(t_flat, dtype=bool))
    elif absolute_sigma:
        raise NotImplementedError("Custom solvers must handle their own absolute-sigma covariance convention")
    sol = solver_obj.fit(
        f=f, xdata=objective_t, ydata=objective_y, beta0=beta0,
        weight_y=weight_y, maxit=maxit, **solver_options,
    )
    if "n_free_kinetics" not in sol.info:
        fixed = kwargs.get("fix_beta")
        n_free = beta0.size if fixed is None else beta0.size - np.count_nonzero(fixed)
        sol = SolverResult(sol.beta, sol.cov_beta, sol.sd_beta, sol.res_var,
                           dict(sol.info, n_free_kinetics=int(n_free)))
    if isinstance(solver_obj, ODRPackSolver):
        # Native ODRPACK does not know that spectra were profiled out. Correct
        # its covariance scaling for those additional fitted parameters.
        projection = projection_at(sol.beta)[1]
        fixed = np.asarray(kwargs.get("fix_beta", np.zeros(beta0.size, dtype=bool)))
        n_free = int(beta0.size - np.count_nonzero(fixed))
        dof = data.size - n_free - projection.effective_parameters
        rss = float(np.sum((projection.residuals / objective_noise) ** 2))
        res_var = rss / dof if dof > 0 else float("nan")
        cov = None
        if sol.cov_beta is not None and sol.res_var is not None and sol.res_var > 0 and dof > 0:
            cov = sol.cov_beta / sol.res_var * (1.0 if absolute_sigma else res_var)
        info = dict(sol.info, n_free_kinetics=n_free, n_projected=projection.effective_parameters,
                    dof=dof, n_observations=data.size, absolute_sigma=absolute_sigma, task="OLS")
        sol = SolverResult(sol.beta, cov, None if cov is None else np.sqrt(np.maximum(np.diag(cov), 0)), res_var, info)
    if not sol.info.get("success", True):
        warnings.warn(f"Global fit did not converge: {sol.info.get('stopreason', '')}", RuntimeWarning, stacklevel=2)
        sol = SolverResult(sol.beta, None, None, sol.res_var, dict(
            sol.info, covariance_unavailable_reason="Optimizer did not converge; confidence intervals are unavailable.",
        ))
    return package_result(
        sol, kinetic_model, times, wl, data, y_obs, noise, lam,
        ci_sigma, ci_level, propagate_kinetic_uncertainty,
        objective_noise=objective_noise, absolute_sigma=absolute_sigma,
    )
