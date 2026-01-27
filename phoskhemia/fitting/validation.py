import numpy as np
from numpy.typing import NDArray

from phoskhemia.data.spectrum_handlers import MyArray
from phoskhemia.fitting.projections import project_amplitudes
from phoskhemia.utils.typing import ArrayFloatAny

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
        num_variables: int,
        errors: ArrayFloatAny | float | None=None
    ) -> float:
    """
    Calculates an estimate of the reduced chi-squared, x², goodness 
    of fit for a provided set of data, fit values, number of variables, 
    and the measurement error. If the model used to fit the data is nonlinear,
    this statistic is a rough estimate as the degrees of freedom, K, can be anywhere
    between 0 and N - 1. If the normalized residuals, Rₙ = [yₙ - f(xₙ, θ)] / σₙ, 
    are representative of the true model and the residuals are distributed normally, 
    then the distribution has mean μ = 0 and variance σ² = 1 and x² is the sum of 
    K variates with probability distribution 
    P(x²; K) = [1 / 2ᴷ𝄍² ∙ Γ(K / 2)] ∙ (x²)ᴷ𝄍²⁻¹ ∙ e⁻ˣᒾ𝄍² and expectation value 
    ⟨x²⟩ = ⎰x² ∙ P(x²; K) dx² = K with a variance of 2K. The expectation value 
    of the reduced x² is then ≈1, with an approximate standard deviation of √2/K.

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.
        num_variables (int): Number of variables used in the fit.
        errors (ArrayFloatAny | float | None, optional): Estimated or known 
            error(s) in data. Defaults to None, in which case the standard 
            deviation of the residuals is used.

    Returns:
        float: Reduced chi-square value.
    """

    residuals: ArrayFloatAny = data - fit
    dof: int = (
        len(data) - num_variables if np.ndim(data) == 1 
        else (data.shape[0] * data.shape[1]) - num_variables
    )
    sumsq_resids: float = np.sum(np.square(residuals))
    sumsq_errors: float = (
        np.sum(np.square(errors)) if errors is not None 
        else np.sum(np.square(np.std(residuals)))
    )
    chi_square: float = (sumsq_resids / sumsq_errors) / dof

    return chi_square

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

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.
        num_variables (int): Number of variables used in model fitting.

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

    Args:
        data (ArrayFloatAny): Array of data values.
        fit (ArrayFloatAny): Array of fit values.
        num_variables (int): Number of variables used in fitting.

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
    y_obs,
    y_fit,
    noise,
    n_params,
):
    """
    Compute statistical diagnostics for global fit.

    Parameters
    ----------
    y_obs : ndarray, shape (N,)
        Observed flattened data
    y_fit : ndarray, shape (N,)
        Fitted flattened data
    noise : ndarray, shape (N,)
        Noise standard deviation per point
    n_params : int
        Number of nonlinear fitted parameters

    Returns
    -------
    diagnostics : dict
    """

    resid = y_obs - y_fit
    wresid = resid / noise

    N = y_obs.size
    p = n_params
    dof = max(N - p, 1)

    chi2 = np.sum(wresid**2)
    chi2_red = chi2 / dof

    # R^2 (unweighted, descriptive only)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # AIC and AICc
    AIC = 2 * p + chi2
    if N > p + 1:
        AICc = AIC + (2 * p * (p + 1)) / (N - p - 1)
    else:
        AICc = np.nan

    return {
        "chi2": chi2,
        "chi2_red": chi2_red,
        "R2": R2,
        "AIC": AIC,
        "AICc": AICc,
        "dof": dof,
    }

def compute_residual_maps(
        arr: MyArray,
        fit_result: dict,
        *,
        noise: NDArray[np.floating] | None = None,
        lam: float = 1e-12,
    ) -> dict[str, MyArray]:
    """
    Compute raw and weighted residual maps for a global kinetic fit.

    Parameters
    ----------
    arr : MyArray
        Dataset to evaluate (train or test)
    fit_result : dict
        Result returned by fit_global_kinetics
    noise : ndarray or None
        Per-wavelength noise σ(λ)
    lam : float
        Tikhonov regularization strength

    Returns
    -------
    residuals : dict
        {
            "raw": MyArray,
            "weighted": MyArray | None,
        }
    """

    if not isinstance(fit_result, dict):
        raise TypeError("fit_result must be a dict from fit_global_kinetics")

    if "_cache" not in fit_result:
        raise KeyError("fit_result is missing '_cache'")

    cache = fit_result["_cache"]
    kinetic_model = cache["model"]
    beta = cache["beta"]

    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    times: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(arr.x, dtype=float)

    if data.shape[0] != times.shape[0]:
        raise ValueError("data and times length mismatch")

    n_wl: int = data.shape[1]

    has_noise: bool = noise is not None
    if noise is None:
        noise = np.ones(n_wl, dtype=float)
    else:
        noise = np.asarray(noise, dtype=float)
        if noise.size != n_wl:
            raise ValueError("noise length must match number of wavelengths")

    # Solve kinetics for THESE times
    traces: NDArray[np.floating] = kinetic_model.solve(times, beta)

    fit: NDArray[np.floating] = np.empty_like(data)

    # Re-project amplitudes wavelength-by-wavelength
    for i in range(n_wl):
        coeffs: NDArray[np.floating]
        coeffs, _, _ = project_amplitudes(
            traces,
            data[:, i],
            noise[i],
            lam,
        )
        fit[:, i] = traces @ coeffs

    raw_residuals: NDArray[np.floating] = data - fit
    raw: MyArray = MyArray(raw_residuals, x=wl, y=times)

    if has_noise:
        weighted_residuals: NDArray[np.floating] = raw_residuals / noise[None, :]
        weighted: MyArray = MyArray(weighted_residuals, x=wl, y=times)
    else:
        weighted: MyArray = None

    return {
        "raw": raw,
        "weighted": weighted,
    }

def compare_models(results, criterion="AICc"):
    """
    Compare multiple fit results.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping name -> fit_result
    criterion : {"AIC", "AICc"}

    Returns
    -------
    comparison : list of dict
        Sorted by best model
    """

    table = []
    for name, res in results.items():
        diag = res["diagnostics"]
        table.append({
            "model": name,
            "AIC": diag["AIC"],
            "AICc": diag["AICc"],
            "chi2_red": diag["chi2_red"],
        })

    key = criterion
    table.sort(key=lambda d: d[key])

    # Compute ΔAIC
    best = table[0][key]
    for row in table:
        row[f"Δ{criterion}"] = row[key] - best

    return table

def cross_validate_wavelengths(
    arr,
    kinetic_model,
    beta0,
    *,
    noise,
    n_folds=5,
    lam=1e-12,
):
    wl = arr.x
    n_wl = len(wl)
    fold_size = n_wl // n_folds

    scores = []

    for k in range(n_folds):
        start = k * fold_size
        stop = (k + 1) * fold_size

        mask = np.ones(n_wl, dtype=bool)
        mask[start:stop] = False

        # Array with fold cut out.
        train = MyArray(
            arr[:, mask],
            x=wl[mask],
            y=arr.y,
        )
        # Array with only the fold.
        test = MyArray(
            arr[:, ~mask],
            x=wl[~mask],
            y=arr.y,
        )

        res = train.fit_global_kinetics(
            kinetic_model,
            beta0,
            noise=noise[mask],
            lam=lam,
        )

        resid = compute_residual_maps(
            test,
            res,
            noise=noise[~mask],
        )

        wres = np.asarray(resid["weighted"])

        N_test = wres.size
        p_nl = kinetic_model.n_params()
        dof = max(N_test - p_nl, 1)

        chi2 = np.sum(wres * wres)
        chi2_red = chi2 / dof

        scores.append(chi2_red)

    return {
        "chi2_red_mean": np.mean(scores),
        "chi2_red_std": np.std(scores),
        "chi2_red_folds": scores,
        "n_folds": n_folds,
    }

def rank_models_by_cv(cv_results, alpha=1.0, tol=0.05):
    """
    Rank models based on CV reduced chi2.

    Parameters
    ----------
    cv_results : dict[str, dict]
        model_name -> result of cross_validate_wavelengths
    alpha : float
        Weight for stability penalty
    tol : float
        Indifference threshold

    Returns
    -------
    ranking : list of dict
        Sorted model ranking
    """

    rows = []

    for name, res in cv_results.items():
        chi2s = np.asarray(res["chi2_red_folds"])
        mu = chi2s.mean()
        sigma = chi2s.std()

        score = mu + alpha * sigma

        rows.append({
            "model": name,
            "chi2_cv_mean": mu,
            "chi2_cv_std": sigma,
            "score": score,
        })

    rows.sort(key=lambda r: r["score"])

    # Compute Δscore
    best_score = rows[0]["score"]
    for r in rows:
        r["Δscore"] = r["score"] - best_score
        r["indistinguishable"] = r["Δscore"] < tol

    return rows


