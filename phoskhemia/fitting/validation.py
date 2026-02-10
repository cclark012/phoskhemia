from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from phoskhemia.fitting.projections import project_amplitudes
from phoskhemia.kinetics.base import KineticModel
from phoskhemia.utils.typing import ArrayFloatAny
if TYPE_CHECKING:
    from phoskhemia.data.spectrum_handlers import TransientAbsorption

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
        *,
        y_obs: NDArray[np.floating],
        y_fit: NDArray[np.floating],
        noise: NDArray[np.floating],
        n_params: int,
    ) -> dict[str, float]:
    """
    Compute statistical diagnostics for a global kinetic fit.

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
        {
            "chi2": float,
            "chi2_red": float,
            "R2": float,
            "AIC": float,
            "AICc": float,
            "dof": int,
        }
    """

    y_obs = np.asarray(y_obs, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if y_obs.shape != y_fit.shape:
        raise ValueError("y_obs and y_fit must have the same shape")

    if noise.shape != y_obs.shape:
        raise ValueError("noise must match y_obs shape")

    if np.any(noise <= 0):
        raise ValueError("noise must be strictly positive for diagnostics")

    N: int = y_obs.size
    p: int = int(n_params)
    dof: int = max(N - p, 1)

    # Residuals
    resid = y_obs - y_fit
    wresid = resid / noise

    # Chi-squared
    chi2: float = float(np.sum(wresid * wresid))
    chi2_red: float = chi2 / dof

    # R^2 (unweighted, descriptive only)
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
    R2: float = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Information criteria (Gaussian likelihood)
    AIC: float = chi2 + 2 * p
    if N > p + 1:
        AICc: float = AIC + (2 * p * (p + 1)) / (N - p - 1)
    else:
        AICc: float = float("nan")

    BIC: float = chi2 + p * np.log(N)

    return {
        "chi2": chi2,
        "chi2_red": chi2_red,
        "R2": R2,
        "AIC": AIC,
        "AICc": AICc,
        "BIC": BIC,
        "dof": dof,
    }

def compute_residual_maps(
        arr: TransientAbsorption,
        fit_result: dict,
        *,
        noise: NDArray[np.floating] | None = None,
        lam: float = 1e-12,
    ) -> dict[str, TransientAbsorption]:
    """
    Compute raw and weighted residual maps for a global kinetic fit.

    Parameters
    ----------
    arr : TransientAbsorption
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
            "raw": TransientAbsorption,
            "weighted": TransientAbsorption | None,
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
    results : dict[str, dict]
        Mapping name -> fit_result
    criterion : {"AIC", "AICc"}
        The criterion to compare between models

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
        arr: TransientAbsorption,
        kinetic_model: KineticModel,
        beta0: NDArray[np.floating],
        *,
        noise: NDArray[np.floating],
        n_folds: int = 5,
        lam: float = 1e-12,
        debug: bool = False,
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

    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    times: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(arr.x, dtype=float)
    beta0: NDArray[np.floating] = np.asarray(beta0, dtype=float)
    noise: NDArray[np.floating] = np.asarray(noise, dtype=float)
    
    if beta0.size != kinetic_model.n_params():
        raise ValueError("beta0 length mismatch")

    n_wl: int = data.shape[1]

    if noise.size != n_wl:
        raise ValueError("noise length must match number of wavelengths")

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
        fit_res: dict = train.fit_global_kinetics(
            kinetic_model,
            beta0,
            noise=noise_train,
            lam=lam,
            propagate_kinetic_uncertainty=False,
            debug=debug,
        )

        # Compute residuals on test wavelengths
        resid: dict[str, TransientAbsorption] = compute_residual_maps(
            test,
            fit_res,
            noise=noise_test,
            lam=lam,
        )

        if resid["weighted"] is None:
            raise RuntimeError("weighted residuals required for CV")

        weighted_res: NDArray[np.floating] = np.asarray(resid["weighted"])

        N_test: int = weighted_res.size
        n_params: int = kinetic_model.n_params()
        dof: int = max(N_test - n_params, 1)

        chi2: float = float(np.sum(weighted_res * weighted_res))
        chi2_red: float = chi2 / dof

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
    best_score: float = rows[0]["score"]
    for r in rows:
        r["Δscore"] = r["score"] - best_score
        r["indistinguishable"] = r["Δscore"] < tol

    return rows


