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

    residuals = data - fit
    mean = np.mean(data)
    sum_sqrs_resids = np.sum(np.square(residuals))
    sum_sqrs_total = np.sum(np.square(data - mean))
    r_sqrd = 1 - (sum_sqrs_resids / sum_sqrs_total)
    
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

    r_sqrd = r_squared(data, fit)
    num_samples = (
        len(data) if np.ndim(data) == 1 
        else (np.shape(data)[0] * np.shape(data)[1])
    )

    dof_resids = (num_samples - num_variables - 1)
    dof_total = (num_samples - 1)

    rbar_sqrd = (1 - (1 - r_sqrd) * (dof_total / dof_resids))

    return rbar_sqrd

def root_mean_squared_error(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
    ) -> float:

    residuals = data - fit
    rmse = np.sqrt(np.mean(np.square(residuals)))

    return rmse

def lack_of_fit(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
    ) -> float:

    residuals = data - fit
    lof = np.sqrt(
        np.sum(np.square(residuals)) / np.sum(np.square(data))
    )

    return lof

def chi_square(
        data: ArrayFloatAny,
        fit: ArrayFloatAny,
        num_variables: int,
        errors: ArrayFloatAny | float | None=None
    ) -> float:

    residuals = data - fit
    dof = (
        len(data) - num_variables if np.ndim(data) == 1 
        else (data.shape[0] * data.shape[1]) - num_variables
    )

    #np.sum([np.sum((np.square(residuals[:, i]) / np.square(noise[i]))) / dof for i in range(len(noise))])
