import numpy as np
from utils.typing import ArrayFloatAny

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
