from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from phoskhemia.fitting.projections import project_spectra
from phoskhemia.fitting.results import GlobalFitResult
from phoskhemia.kinetics.base import KineticModel
from phoskhemia.utils.typing import ArrayFloatAny
if TYPE_CHECKING:
    from phoskhemia.data.spectrum2d import TransientAbsorption


def _validated_data(data, fit):
    """Require finite observations and predictions on exactly the same grid."""
    data = np.asarray(data, dtype=float)
    fit = np.asarray(fit, dtype=float)
    if data.shape != fit.shape:
        raise ValueError("data and fit must have the same shape")
    if data.size == 0:
        raise ValueError("data and fit must be nonempty")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(fit)):
        raise ValueError("data and fit must contain only finite values")
    return data, fit


def _validated_noise(noise, shape):
    """Broadcast positive standard deviations without broadcasting the data."""
    noise = np.asarray(noise, dtype=float)
    if not np.all(np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("noise standard deviations must be finite and strictly positive")
    try:
        return np.broadcast_to(noise, shape)
    except ValueError as exc:
        raise ValueError("noise must be scalar or broadcast to the data shape") from exc


def _parameter_count(value):
    """Allow an explicitly supplied effective parameter count for penalized fits."""
    count = np.asarray(value, dtype=float)
    if count.ndim != 0 or not np.isfinite(count) or count < 0:
        raise ValueError("parameter count must be a finite, nonnegative scalar")
    return float(count)


def _result_cache(result):
    """Read current results, retaining support for older in-memory result dicts."""
    if isinstance(result, GlobalFitResult):
        return result._cache
    if isinstance(result, dict):
        return result["_cache"]
    raise TypeError("fit_result must be a GlobalFitResult or a legacy result dict")


def r_squared(
        data: ArrayFloatAny, 
        fit: ArrayFloatAny
    ) -> float:
    """
    Computes the R-squared value for a set of data and their 
    associated fit. While not strictly valid for nonlinear
    fits, the R-squared can still give a general idea of
    how a fit has performed.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of values fit to the data.

    Returns:
        float: The R-squared value. Usually between 0 and 1,
            but can be negative when the sample mean is a better
            overall predictor than the supplied fit.
    """

    residuals: ArrayFloatAny = data - fit
    mean: float = np.mean(data)
    sum_sqrs_resids: float = np.sum(np.square(residuals))
    sum_sqrs_total: float = np.sum(np.square(data - mean))
    r_sqrd: float = 1 - (sum_sqrs_resids / sum_sqrs_total)
    
    return r_sqrd

def adjusted_r_squared(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
        num_variables: int,
    ) -> float:
    """
    Computes the adjusted R-squared (also called 
    R-bar squared) for a set of data and the associated
    fit. As the normal R-squared increases when more
    explanatory variables are added to the fitting model,
    it does not allow for strictly meaningful comparisons.
    Adjusting the R-squared value by the degrees of freedom
    restricts the statistic to only increase when the added
    variables are actually meaningful.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of values fit to the data.
        num_variables (int): Number of explanatory 
            variables (excluding intercept).

    Returns:
        float: Adjusted R-squared value.
    """

    r_sqrd: float = r_squared(data, fit)
    num_samples: int = (
        len(data) if np.ndim(data) == 1 
        else (np.shape(data)[0] * np.shape(data)[1])
    )

    dof_resids: int = (num_samples - num_variables - 1)
    dof_total: int = (num_samples - 1)

    rbar_sqrd: float = (1 - (1 - r_sqrd) * (dof_total / dof_resids))

    return rbar_sqrd

def root_mean_squared_error(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
    ) -> float:
    """
    Calculates the root-mean squared error of the residuals 
    between the provided data and fit.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.

    Returns:
        float: RMSE value.
    """

    residuals: ArrayFloatAny = data - fit
    rmse: float = np.sqrt(np.mean(np.square(residuals)))

    return rmse

def lack_of_fit(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
    ) -> float:
    """
    Generates a lack-of-fit statistic for the 
    provided data and associated fit.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.

    Returns:
        float: Lack-of-Fit parameter. 
    """

    residuals: ArrayFloatAny = data - fit
    lof: float = np.sqrt(
        np.sum(np.square(residuals)) / np.sum(np.square(data))
    )

    return lof

def reduced_chi_square(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
        num_variables: int | float,
        errors: ArrayFloatAny | float | None=None
    ) -> float:
    """
    Return ``sum(((data - fit) / errors)**2) / (N - p)``.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Predictions with exactly the same shape as data.
        num_variables (int | float): Total fitted parameter count, including
            projected amplitudes. An effective count can be supplied for a
            penalized fit, in which case this is an approximate diagnostic.
        errors (ArrayFloatAny | float | None): Noise standard deviations,
            either scalar or broadcastable to data. For a time-by-wavelength
            matrix, a wavelength vector is broadcast over time. If omitted,
            the residual standard deviation is used for backward compatibility;
            that self-normalized statistic is not an independent goodness-of-fit
            check.

    Returns:
        float: Reduced chi-square, or NaN when the degrees of freedom are
            nonpositive or the inferred residual standard deviation is zero.

    With independent Gaussian errors and a correctly specified locally linear
    model, the expected reduced chi-square is approximately one, with standard
    deviation ``sqrt(2 / (N - p))``. Correlated noise and strong constraints
    invalidate this simple degrees-of-freedom interpretation.
    """
    data, fit = _validated_data(data, fit)
    residuals = data - fit
    dof = data.size - _parameter_count(num_variables)
    if errors is None:
        errors = float(np.std(residuals))
        if errors == 0:
            return float("nan")
    sigma = _validated_noise(errors, data.shape)
    if dof <= 0:
        return float("nan")
    return float(np.sum(np.square(residuals / sigma)) / dof)

def bayesian_information_criterion(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
        num_variables: int
    ) -> float:
    """
    Calculates the Bayesian information criterion for a given model.
    The residuals are assumed independent, normally distributed, 
    and that the derivative of the log likelihood with respect 
    to the true variance is zero. The value in and of itself is not
    very informative and only becomes useful when comparing to other
    models of the same data. In the case of model comparison, the smallest
    value is the best model.

    This legacy helper profiles a single unknown common variance and omits
    likelihood constants. For known, wavelength-dependent noise, use
    ``compute_diagnostics`` instead; the two conventions are not interchangeable.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.
        num_variables (int): Total estimated likelihood parameters, including
            the common variance when that variance is estimated from the data.

    Returns:
        float: Bayesian Information Criterion parameter.
    """

    residuals: ArrayFloatAny = data - fit
    num_elements: int = (
        len(data) if np.ndim(data) == 1 
        else data.shape[0] * data.shape[1]
    )

    BIClikelihood: float = (
        num_variables * np.log(num_elements) 
        + num_elements * np.log(np.sum(np.square(residuals)) / num_elements)
    )

    return BIClikelihood

def akaike_information_criterion(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
        num_variables: int
    ) -> float:
    """
    Calculates the Akaike information criterion for a given model. A
    correction for small sample sizes is also included which approaches
    zero as the number of samples becomes larger. The assumptions and 
    use cases are similar to the Bayesion information criterion. 
    See bayesian_information_criterion for more information.

    This returns the small-sample-corrected AIC (AICc), profiles a single
    unknown common variance, and omits likelihood constants. For known,
    wavelength-dependent noise, use ``compute_diagnostics`` instead.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.
        num_variables (int): Total estimated likelihood parameters, including
            the common variance when that variance is estimated from the data.

    Returns:
        float: Akaike information criterion parameter.
    """

    residuals: ArrayFloatAny = data - fit
    num_elements: int = (
        len(data) if np.ndim(data) == 1 
        else data.shape[0] * data.shape[1]
    )
    
    AIClikelihood: float = (
        2 * num_variables 
        + num_elements * np.log(np.sum(np.square(residuals)) / num_elements)
    )
    
    correction: float = (
        (2 * np.square(num_variables) + 2 * num_variables) 
        / (num_elements - num_variables - 1)
    )

    return AIClikelihood + correction

def compute_diagnostics(
        *,
        y_obs: NDArray[np.floating],
        y_fit: NDArray[np.floating],
        noise: NDArray[np.floating] | float,
        n_params: int | float,
    ) -> dict[str, float]:
    """
    Compute diagnostics for independent Gaussian errors with known noise.

    Parameters
    ----------
    y_obs, y_fit : ndarray
        Finite, nonempty observations and predictions with identical shapes.
    noise : ndarray or float
        Positive standard deviations, scalar or broadcastable to y_obs.
    n_params : int or float
        Total fitted parameter count, including the fitted spectral amplitudes.
        An effective count may be supplied for ridge regression, but likelihood
        criteria and reduced chi-square then have only an approximate meaning.

    Returns
    -------
    diagnostics : dict
        Chi-square, descriptive R-squared, full Gaussian log likelihood,
        AIC, AICc, BIC, observation/parameter counts, and residual ``DoF``.
        Undefined reduced chi-square or small-sample corrections are NaN.

    Notes
    -----
    The likelihood includes the normalization ``sum(log(2*pi*noise**2))``.
    Comparisons require the same observations and compatible likelihood models.
    The usual AICc expression is a small-sample heuristic for general nonlinear
    fits. Uncertainty in estimated noise and correlation introduced by smoothing
    or other preprocessing are not modeled here.
    """
    y_obs, y_fit = _validated_data(y_obs, y_fit)
    noise = _validated_noise(noise, y_obs.shape)

    N = y_obs.size
    p = _parameter_count(n_params)
    dof = N - p

    # Residuals
    resid = y_obs - y_fit
    wresid = resid / noise

    # Chi-squared
    chi2: float = float(np.sum(wresid * wresid))
    chi2_red: float = chi2 / dof if dof > 0 else float("nan")

    # R^2 (unweighted, descriptive only)
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
    R2: float = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Full known-noise Gaussian likelihood; taking logs before squaring avoids
    # underflow of tiny but representable experimental standard deviations.
    minus_two_log_likelihood = chi2 + float(
        np.sum(np.log(2 * np.pi) + 2 * np.log(noise))
    )
    AIC: float = minus_two_log_likelihood + 2 * p
    if N > p + 1:
        AICc: float = AIC + (2 * p * (p + 1)) / (N - p - 1)
    else:
        AICc: float = float("nan")

    BIC: float = minus_two_log_likelihood + p * np.log(N)

    return {
        "chi2": chi2,
        "chi2_red": chi2_red,
        "R2": R2,
        "AIC": AIC,
        "AICc": AICc,
        "BIC": BIC,
        "DoF": dof,
        "n_obs": N,
        "n_params": p,
        "log_likelihood": -0.5 * minus_two_log_likelihood,
    }

def compute_residual_maps(
        arr: TransientAbsorption,
        fit_result: GlobalFitResult | dict,
        *,
        noise: NDArray[np.floating] | float | None = None,
        lam: float = 0.0,
    ) -> dict[str, TransientAbsorption | None]:
    """
    Reproject spectra at fixed fitted kinetics and return residual maps.

    Amplitudes are estimated again from ``arr``, so this supports held-out
    wavelengths in cross-validation. To subtract the original fitted signal
    without re-estimating amplitudes, use ``reconstruct_residuals`` instead.

    Parameters
    ----------
    arr : TransientAbsorption
        Dataset to evaluate (train or test)
    fit_result : GlobalFitResult or dict
        Result returned by fit_global_kinetics
    noise : ndarray, float or None
        Per-wavelength noise σ(λ)
    lam : float
        Tikhonov regularization strength

    Returns
    -------
    residuals : dict
        {
            "raw": TransientAbsorption,
            "weighted": TransientAbsorption | None,
        }
    """

    from phoskhemia.data.spectrum2d import TransientAbsorption

    cache = _result_cache(fit_result)
    kinetic_model = cache.get("kinetic_model", cache.get("model"))
    if kinetic_model is None:
        raise ValueError("fit_result has no cached kinetic model")
    beta = cache["beta"]

    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    times: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(arr.x, dtype=float)

    if data.ndim != 2 or data.shape != (times.size, wl.size):
        raise ValueError("data must have shape (n_times, n_wavelengths)")
    _validated_data(data, data)

    n_wl: int = data.shape[1]

    has_noise: bool = noise is not None
    if noise is None:
        noise = np.ones(n_wl, dtype=float)
    else:
        noise = _validated_noise(noise, (n_wl,))

    # Solve kinetics for THESE times
    traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
    traces = np.asarray(traces, dtype=float)
    if traces.ndim == 1:
        traces = traces[:, None]
    if traces.ndim != 2 or traces.shape[0] != times.size:
        raise ValueError("kinetic traces must have shape (n_times, n_species)")

    projection = project_spectra(traces, data, noise, lam)
    raw_residuals = projection.residuals
    raw: TransientAbsorption = TransientAbsorption(raw_residuals, x=wl, y=times)

    if has_noise:
        weighted_residuals: NDArray[np.floating] = raw_residuals / noise[None, :]
        weighted: TransientAbsorption = TransientAbsorption(weighted_residuals, x=wl, y=times)
    else:
        weighted: TransientAbsorption = None

    return {
        "raw": raw,
        "weighted": weighted,
    }

def compare_models(results, criterion="AICc"):
    """
    Compare multiple fit results.

    Parameters
    ----------
    results : dict[str, GlobalFitResult | dict]
        Mapping name -> fit_result
    criterion : {"AIC", "AICc", "BIC"}
        The criterion to compare between models

    Returns
    -------
    comparison : list of dict
        Sorted by best model
    """

    if criterion not in {"AIC", "AICc", "BIC"}:
        raise ValueError("criterion must be 'AIC', 'AICc', or 'BIC'")
    table = []
    for name, res in results.items():
        diag = res.diagnostics if isinstance(res, GlobalFitResult) else res["diagnostics"]
        table.append({
            "model": name,
            "AIC": diag["AIC"],
            "AICc": diag["AICc"],
            "BIC": diag.get("BIC", float("nan")),
            "chi2_red": diag["chi2_red"],
        })

    key = criterion
    table.sort(key=lambda d: d[key] if np.isfinite(d[key]) else np.inf)
    if not table:
        return table

    # Undefined criteria remain NaN rather than winning a model comparison.
    best = table[0][key]
    for row in table:
        row[f"Δ{criterion}"] = (
            row[key] - best if np.isfinite(row[key]) and np.isfinite(best)
            else float("nan")
        )

    return table

def cross_validate_wavelengths(
        arr: TransientAbsorption,
        kinetic_model: KineticModel,
        beta0: NDArray[np.floating],
        *,
        noise: NDArray[np.floating] | float,
        n_folds: int = 5,
        lam: float = 0.0,
        debug: bool = False,
        **fit_kwargs,
    ) -> dict:
    """
    Perform wavelength-block cross-validation for a global kinetic model.

    Parameters
    ----------
    arr : TransientAbsorption
        Full dataset
    kinetic_model : KineticModel
        Kinetic model to validate
    beta0 : ndarray
        Initial kinetic parameter guesses
    noise : ndarray
        Per-wavelength noise σ(λ)
    n_folds : int
        Number of wavelength folds
    lam : float
        Tikhonov regularization strength
    debug : bool
        Print diagnostics on failures
    **fit_kwargs
        Additional options forwarded to ``fit_global_kinetics`` (for example
        ``solver`` and ``maxit``). Kinetics are learned from training wavelengths;
        each held-out wavelength still has its own fitted spectral amplitudes.

    Returns
    -------
    result : dict
        {
            "chi2_red_folds": list[float],
            "chi2_red_mean": float,
            "chi2_red_std": float,
            "n_folds": int,
        }
    """

    from phoskhemia.data.spectrum2d import TransientAbsorption

    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    times: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(arr.x, dtype=float)
    beta0: NDArray[np.floating] = np.asarray(beta0, dtype=float)
    
    if beta0.size != kinetic_model.n_params():
        raise ValueError("beta0 length mismatch")

    if data.ndim != 2 or data.shape != (times.size, wl.size):
        raise ValueError("data must have shape (n_times, n_wavelengths)")
    _validated_data(data, data)
    n_wl: int = data.shape[1]
    noise = _validated_noise(noise, (n_wl,))

    if n_folds < 2 or n_folds > n_wl:
        raise ValueError("invalid number of folds")

    # Build wavelength indices
    indices: NDArray[np.integer] = np.arange(n_wl)
    folds: list[NDArray[np.integer]] = np.array_split(indices, n_folds)

    chi2_red_folds: list[float] = []

    for k, test_idx in enumerate(folds):
        # Set the train/test mask
        train_mask: NDArray[np.bool_] = np.ones(n_wl, dtype=bool)
        train_mask[test_idx] = False

        # Mask wavelength block for training set
        train = TransientAbsorption(
            data[:, train_mask],
            x=wl[train_mask],
            y=times,
        )
        # Mask everything except the wavelength block for the test set
        test = TransientAbsorption(
            data[:, ~train_mask],
            x=wl[~train_mask],
            y=times,
        )

        noise_train: NDArray[np.floating] = noise[train_mask]
        noise_test: NDArray[np.floating] = noise[~train_mask]

        # Fit on training wavelengths only
        fit_res = train.fit_global_kinetics(
            kinetic_model,
            beta0,
            noise=noise_train,
            lam=lam,
            propagate_kinetic_uncertainty=False,
            debug=debug,
            **fit_kwargs,
        )

        cache = _result_cache(fit_res)
        traces = np.asarray(kinetic_model.solve(times, cache["beta"]), dtype=float)
        if traces.ndim == 1:
            traces = traces[:, None]
        projection = project_spectra(traces, np.asarray(test), noise_test, lam)
        weighted_res = projection.residuals / noise_test[None, :]
        N_test = weighted_res.size
        # Kinetics were learned on the training set. The held-out data determine
        # only these new spectral amplitudes. The effective count equals rank
        # times wavelengths for lam=0; reduced chi-square is approximate with a
        # ridge penalty because shrinkage introduces bias.
        dof = N_test - projection.effective_parameters

        chi2: float = float(np.sum(weighted_res * weighted_res))
        chi2_red: float = chi2 / dof if dof > 0 else float("nan")

        chi2_red_folds.append(chi2_red)

        if debug:
            print(f"CV fold {k}: chi2_red = {chi2_red:.3f}")

    chi2_red_folds: NDArray[np.floating] = np.asarray(chi2_red_folds)

    return {
        "chi2_red_folds": chi2_red_folds.tolist(),
        "chi2_red_mean": float(chi2_red_folds.mean()),
        "chi2_red_std": float(chi2_red_folds.std(ddof=1)),
        "n_folds": n_folds,
    }

def cv_rank_models(
        cv_results: dict[str, dict],
        *,
        alpha: float = 1.0,
        tol: float = 0.05,
    ) -> list[dict]:
    """
    Rank models based on cross-validated reduced x².

    Parameters
    ----------
    cv_results : dict
        Mapping model_name -> cross_validate_wavelengths result
    alpha : float
        Weight for fold-to-fold variability penalty
    tol : float
        Indifference threshold for Δscore

    Returns
    -------
    ranking : list of dict
        Each entry contains:
            - model
            - chi2_cv_mean
            - chi2_cv_std
            - score
            - Δscore
            - indistinguishable
    """

    rows: list[dict] = []

    for model_name, res in cv_results.items():
        chi2_folds: NDArray[np.floating] = np.asarray(res["chi2_red_folds"], dtype=float)

        mu: float = float(chi2_folds.mean())
        sigma: float = float(chi2_folds.std())

        score: float = mu + alpha * sigma

        rows.append({
            "model": model_name,
            "chi2_cv_mean": mu,
            "chi2_cv_std": sigma,
            "score": score,
        })

    # Sort by composite score
    rows.sort(key=lambda r: r["score"])

    # Compute Δscore and equivalence flag
    if not rows:
        return rows
    best_score: float = rows[0]["score"]
    for r in rows:
        r["Δscore"] = r["score"] - best_score
        r["indistinguishable"] = r["Δscore"] < tol

    return rows


