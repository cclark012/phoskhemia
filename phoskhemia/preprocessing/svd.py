from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

def minmax_eigenvalues(
        lam: float
    ) -> tuple[float, float]:
    """
    Calculate the min and max eigenvalues for a Marchenko-Pastur distribution.

    Based on equation 118 from (1).
    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.

    Parameters
    ----------
    lam : float
        Aspect ratio to compute the eigenvalue distribution (rows / columns).

    Returns
    -------
    tuple[float, float]
        Minimum and maximum eigenvalues.
    """

    min_eigenvalue: float = (1 - np.sqrt(lam)) ** 2
    max_eigenvalue: float = (1 + np.sqrt(lam)) ** 2

    return min_eigenvalue, max_eigenvalue

def marchenko_pastur_pdf(
        x: NDArray[np.floating], 
        lam: float
    ) -> NDArray[np.floating]:
    """
    Marchenko-Pastur probability density function for a chosen lam value. 
    lam is usually the ratio of rows to columns.
    Based on equations 117 and 118 from (1).
    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.

    Parameters
    ----------
    x : NDArray[np.floating]
        Range of values for which to calculate the PDF.
    lam : float
        Aspect ratio (rows / columns).

    Returns
    -------
    NDArray[np.floating]
        The values of the PDF over x.
    """

    prefactor: float = (1 / (2 * np.pi * lam))
    # Min and max eigenvalues for given lam.
    lamplus: float = (1 + np.sqrt(lam)) ** 2
    lamminus: float = (1 - np.sqrt(lam)) ** 2
    lamprod: NDArray[np.floating] = (lamplus - x) * (x - lamminus)
    # Filter invalid values
    lamprod[lamprod < 0] = 0

    distribution: NDArray[np.floating] = (
        prefactor * (np.sqrt(lamprod) / x)
    )

    return distribution

def marchenko_pastur_cdf(
        x: NDArray[np.floating],
        lam: float,
    ) -> NDArray[np.floating]:
    """
    Computes the Marchenko-Pastur cumulative density function for a chosen lam value. 
    
    Uses the analytic integral of the PDF. Specific values of x and lam 
    that cause issues (square root of negative, division by zero) 
    are parsed during evaluation. If a general x array is passed, 
    there may be sharp cut-on and cut-off points seen in the 
    final cdf; these are where the minimum and maximum eigenvalues occur.
    a and b are the smallest and largest eigenvalues in the distribution,
    respectively.
    P(y) = (1 / 2πy) ∙ {2√(ab} ∙ [tan⁻¹((a(b - z) / b(z - a))¹ᐟ²) - π / 2]
        + ((a + b) / 2) ∙ [tan⁻¹((z - ½(a + b)) / ((b - z)(z - a))¹ᐟ²) + π / 2]
        + ((b - z)(z - a))¹ᐟ²}

    Based on equation 119 in (1).
    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.

    Parameters
    ----------
    x : NDArray[np.floating]
        Range of values for which to calculate the CDF.
    lam : float
        Aspect ratio (rows / columns).

    Returns
    -------
    NDArray[np.floating]
        The values of the CDF over x.
    """

    lamplus: float
    lamminus: float
    lamminus, lamplus = minmax_eigenvalues(lam)
    lamprod: NDArray[np.floating] = (lamplus - x) * (x - lamminus)
    # Filter invalid values
    lamprod[lamprod < 0] = 0
    sqrt_ab: NDArray[np.floating] = np.sqrt(lamprod)

    # Make sure there is no dividing by zero 
    diff_prod: NDArray[np.floating] = (lamplus * (x - lamminus))
    fraction1: NDArray[np.floating] = np.zeros_like(x)
    fraction1[diff_prod > 0] = (
        (lamminus * (lamplus - x[diff_prod > 0])) / diff_prod[diff_prod > 0]
    )
    # Remove all negative values (if any)
    fraction1[fraction1 < 0] = 0

    inv_tangent1: NDArray[np.floating] = (
        2 * np.sqrt(lamplus * lamminus) 
        * (np.arctan(np.sqrt(fraction1)) - (np.pi / 2))
    )

    # Guard for divide by zero and domain errors
    condition: NDArray[np.bool_] = sqrt_ab > 0
    inv_tangent2: NDArray[np.floating] = np.zeros_like(x)
    inv_tangent2[condition] = (
        ((lamplus + lamminus) / 2) * (
            (np.arctan((x[condition] - ((lamplus + lamminus) / 2)) 
                / sqrt_ab[condition])) + (np.pi / 2)
        )
    )

    cdf: NDArray[np.floating] = (1 / (2 * np.pi * lam)) * (
        inv_tangent1 + inv_tangent2 + sqrt_ab
    )
    # Remove any negative values that may have crept in
    cdf[cdf < 0] = 0

    return cdf

def lambda_k(
        lam: float,
        num_singulars: int,
    ) -> NDArray[np.floating]:
    """
    Evaluates the Marchenko-Pastur cdf between min and max eigenvalues with even spacing.
    
    Based off of equations (117) - (120) in (1).
    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.


    Parameters
    ----------
    lam : float
        Aspect ratio (rows / columns)
    num_singulars : int
        Number of singular values to evaluate for.

    Returns
    -------
    NDArray[np.floating]
        The interpolated eigenvalue distribution.
    """

    lamminus: float
    lamplus: float
    lamminus, lamplus = minmax_eigenvalues(lam=lam)

    # Range over which to interpolate.
    eigen_range: NDArray[np.floating] = (
        np.linspace(lamminus, lamplus, num_singulars * 2)
    )

    cdf: NDArray[np.floating] = (
        marchenko_pastur_cdf(x=eigen_range, lam=lam)
    )

    # Even spacing across the range of values.
    interval: NDArray[np.floating] = (
        1 - (np.arange(0, num_singulars, 1, dtype=int) / (num_singulars - 1))
    )

    # Interpolate to find evenly spaced eigenvalues.
    eigens: NDArray[np.floating] = (
        PchipInterpolator(
            cdf[cdf.nonzero()], eigen_range[cdf.nonzero()]
        )(interval)
    )

    return eigens

def estimate_mse(
        unit_error: NDArray[np.floating],
        singular_values: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    
    ϵ' - Measurement Error
    T - Number of rows of matrix A
    ᵴ - Noisy singular values.
    ŝ - "Unit" Marchenko-Pastur distribution
    Given a tail-start index k, the tail of noisy singular values ᵴₗ (l=k,...,T) 
    are fit to a Marchenko-Pastur distribution ϵ'ŝₗ via least squares. The
    mean square error between log₁₀(ϵ'ŝₗ) and log₁₀(ᵴₗ) is:
    L = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ϵ') + log₁₀(ŝₗ) - log₁₀(ᵴₗ))
    The ϵ' yielding the minimum value for L is found by solving
    dL / d(log₁₀(ϵ')) = 0 for ϵ', which gives:
    log₁₀(ϵ') = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ᵴₗ) - log₁₀(ŝₗ)).
    For each k, log₁₀(ϵ') can be used to find the best fit ϵ'₍ₖ₎ and then used
    to find the associated error L₍ₖ₎. 
    Based on equations 26 and 27 in (1).
    
    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    unit_error : NDArray[np.floating]
        The unit error, i.e. a linear scaling parameter 
        for the Marchenko-Pastur distribution.
    singular_values : NDArray[np.floating]
        The singular values from SVD.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        The arrays of mean square error of the fit and the fit noise level.
        Both arrays are of shape (K,) with K being the number of singular values.
    """

    num_singulars: int = len(singular_values)
    log_noise: NDArray[np.floating] = (
        np.array(list(map(lambda k: (
            (1 / (num_singulars - k)) * np.sum(
                np.log10(singular_values[k:]) - np.log10(unit_error[k:])
            )), np.arange(num_singulars))))
    )

    mean_square_error: NDArray[np.floating] = (
        np.array(list(map(lambda k: (
            (1 / (num_singulars - k)) * np.sum(np.square(
                log_noise[k] + np.log10(unit_error[k:]) - np.log10(singular_values[k:])
            ))), np.arange(num_singulars))))
    )

    return mean_square_error, log_noise


def calculate_error_estimates(
        n_cols: int, 
        svd_vals: NDArray[np.floating], 
        unit_error: NDArray[np.floating], 
    ) -> tuple[float, float, float]:
    """
    Construct estimates of error in data.
    
    ϵ' - Fit Measurement Error
    ϵ" - Preliminary Estimate of Measurement Error
    ϵ - True Error, generally ϵ > ϵ" and ϵ < ϵ'
    ε̄ - Estimate of the true measurement error
    T - Number of rows of matrix A
    ᵴ - Noisy singular values
    ŝ - "Unit" Marchenko-Pastur distribution
    kₑ - Critical index of best fit
    kₑ' - Critical index of estimate
    (27) log₁₀(ϵ') = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ᵴₗ) - log₁₀(ŝₗ))
    (26) L = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ϵ') + log₁₀(ŝₗ) - log₁₀(ᵴₗ))
    (29) ϵ" = (ϵ'(kₑ) ∙ ŝ(kₑ)) / √D
    (30) kₑ' = minₖ ᵴₖ < ϵ'√D
    (31) ε̄ = min{ϵ', ϵ" + (ϵ' - ϵ") ∙ [(kₑ - kₑ') / (floor(0.8T) - kₑ')]}
    
    Following the procedures outlined in (1), log(noise) and MSE are constructed
    using equations 27 and 26. The unit Marchenko-Pastur distribution is 
    evaluated for each tail index. The best index (lowest MSE) is found and 
    used to construct a preliminary estimate of the measurement error using
    equation 29. Equation 30 is used to find the critical index for the 
    fit measurement error ϵ'. Finally, the measurement error is estimated using 
    equation 31.

    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    n_cols : int
        Number of columns in original array.
    svd_vals : NDArray[np.floating]
        Array of singular values.
    unit_error : NDArray[np.floating]
        Array of unit error singular values, ŝₖ = √Dλₖ.

    Returns
    -------
    tuple[ float, float, float ]
        Fit measurement error, preliminary estimate of measurement 
        error, and the estimate for the true measurement error.
    """

    # Mean squared error and loss function for Marchenko-Pastur distribution of ϵ'*sₖ.
    mean_square_error: NDArray[np.floating]
    log_noise: NDArray[np.floating]
    mean_square_error, log_noise = estimate_mse(unit_error, svd_vals)

    # Minimum of loss function.
    best_index: int = (
        np.argmin(mean_square_error[:int(np.floor(len(svd_vals) * 0.8))])
    )

    # ϵ' and ϵ" at loss function minimum.
    error_prime: float = np.power(10., log_noise[best_index])
    error_doubleprime: float = (
        (error_prime * unit_error[best_index]) / np.sqrt(n_cols)
    )

    # Index where singular values fall below noise threshold.
    other_best_index: int = (
        np.min((svd_vals < error_prime * np.sqrt(n_cols)).nonzero())
    )

    # Estimate measurement error from the weighted average of ϵ' and ϵ".
    measurement_error: float = (np.min((
        error_prime, (
            error_doubleprime + (error_prime - error_doubleprime) 
            * ((best_index - other_best_index) / (len(svd_vals) - other_best_index))
        )
    )))

    return (error_prime, error_doubleprime, measurement_error)

def calculate_threshold_indices(
        svd_vals: NDArray[np.floating], 
        measurement_error: float, 
        shape: tuple[int, int]
    ) -> tuple[int, int, int]:
    """
    Finds the critical mode indices for various noise levels.
    
    Based on equations 29 in (1) and 20 in (2). The RMSE(ṽₖ) ≈ ε̄ / ᵴₖ,
    so the indices kf, k2, and ke correspond to noise 
    RMSE(ṽₖ) ≈ 1 / √TD, 1 / (√D + √fT), and 1 / √D.
    kf = minₖ ᵴₖ < ε̄√TD
    k2 = minₖ ᵴₖ < ε̄(√D + √fT)
    ke = minₖ ᵴₖ < ε̄√D

    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.
    
    (2) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    svd_vals : NDArray[np.floating]
        Array of singular values.
    measurement_error : float
        Estimate of the true measurement error.
    shape : tuple[int, int]
        Shape of original array.

    Returns
    -------
    tuple[int, int, int]
        The indices where the RMSE(ṽₖ) ≈ 1 / √TD, 1 / (√D + √T), and 1 / √D.
    """

    T: int = shape[0]
    D: int = shape[1]
    # First mode that fails the test of ᵴₖ > ε̄√DT.
    kf: int = np.min((svd_vals < measurement_error * np.sqrt(D * T)).nonzero())
    # Rough estimate of minimum-loss reconstruction rank.
    k2: int = np.min((svd_vals < measurement_error * (np.sqrt(D) + np.sqrt(T))).nonzero())
    # Minimum index for where singular values and Marchenko-Pastur overlay one another.
    ke: int = np.min((svd_vals < measurement_error * np.sqrt(D)).nonzero())

    return kf, k2, ke

def calculate_mode_rmse(
        svd_vals: NDArray[np.floating], 
        measurement_error: NDArray[np.floating], 
        shape: tuple[int, int]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculates the root mean square error for the left and right singular vectors.
    
    Based on equations 8 - 20 in (1) and 43 - 44 in (2). The expectation values
    for the root mean square error (RMSE) was derived from perturbation theory
    to be defined as:
    (1) RMSE(ũₖ) = [(1 / T) ∙ Σᵢ₌₁ᵀ (Ũᵢₖ - Uᵢₖ)²]¹ᐟ²
    (2) RMSE(ṽₖ) = [(1 / D) ∙ Σᵢ₌₁ᴰ (Ṽᵢₖ - Vᵢₖ)²]¹ᐟ²
    With the expectation values:
    (3) ⟨RMSE(ũₖ)⟩ = ⟨[(1 / T) ∙ Σᵢ₌₁ᵀ (Ũᵢₖ - Uᵢₖ)²]¹ᐟ²⟩ 
    (4) = ϵ⟨[(1 / T) ∙ Σᵢ₌₁ᵀ (Wᵢₘ⁽¹⁾ ∙ Uᵢₘ)²]¹ᐟ² + 𝒪(ϵ²)
    (4) cannot be simplified into a practical form, but the root mean square
    standard deviation (RMS) does have an analytical form:
    (5) RMS(σᵤ) = [(1 / T) ∙ Σᵢ₌₁ᵀ σ(Ũᵢₖ)²]¹ᐟ² 
    (6) = ϵ[(1 / T) ∙ Σᵢ₌₁ᵀ ⟨(Wᵢₘ⁽¹⁾ ∙ Uᵢₘ)²⟩]¹ᐟ² + 𝒪(ϵ²)
    (7) = σ(Ũᵢₖ) = ϵ√w [Σₘ₌₁ᵀ ((λₘ + λₖ) / (λₘ - λₖ)²) ∙ Uᵢₘ²]¹ᐟ² + 𝒪(ϵ²) for m ≠ k.
    The combination of (7) and (5) along with the unit-norm property 
    Σᵢ₌₁ᵀ Uᵢₘ² = 1 yields equation (8):
    (8) RMS(σᵤ) = (ϵ / sₖ) ∙ [(w / T) ∙ Σₘ₌₁ᵀ λₖ(λₘ + λₖ) / (λₘ - λₖ)²]¹ᐟ² + 𝒪(ϵ²) for m ≠ k.
    While RMS(σᵤ) is the square root of an average and ⟨RMSE(ũₖ)⟩ is the average 
    of the square root, they are expected to be approximately equal.
    (9) ⟨RMSE(ũₖ)⟩ ≈ RMS(σᵤ)
    An analogous development can be shown for the right singular vectors. Using 
    equation (10), equation (11) was developed.
    (10) σ(Ṽᵢₖ) = (ϵ / sₖ) ∙ [1 - wVᵢₖ² + w ∙ Σₘ₌₁ᵀ (λₘ(3λₖ - λₘ) / (λₘ - λₖ)²) ∙ Vᵢₘ²]¹ᐟ² + 𝒪(ϵ²) for m ≠ k.
    (11) RMS(σᵥ) = (ϵ / sₖ) ∙ [((D - w) / D) + (w / D) ∙ Σₘ₌₁ᵀ (λₘ(3λₖ - λₘ) / (λₘ - λₖ)²)]¹ᐟ² + 𝒪(ϵ²) for m ≠ k.
    (12) ⟨RMSE(ṽₖ)⟩ ≈ RMS(σᵥ)
    In a practical implementation, RMS is found with:
    (13) RMS(σᵤ) ≈ min{(2/T)¹ᐟ², RMS(σᵤ)}
    (14) RMS(σᵥ) ≈ min{(2/D)¹ᐟ², RMS(σᵥ)}
    Using equations (8) and (11) inside equations (13) and (14), respectively.


    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.
    
    (2) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    svd_vals : NDArray[np.floating]
        Array of singular values.
    measurement_error : float
        Estimate of the true measurement error.
    shape : tuple[int, int]
        Shape of the original array.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        The estimated root mean squared error for the left and right singular vectors.
    """

    u_rmse: NDArray[np.floating] = (
        (measurement_error / svd_vals) * np.sqrt([((1 / shape[0])
            * np.sum([((svd_vals[j] ** 2) * (m + (svd_vals[j] ** 2))) / np.square(m - (svd_vals[j] ** 2))
            for m in np.square(np.delete(svd_vals, j))])) for j in range(len(svd_vals))
        ])
    )
    u_rmse[u_rmse >= np.sqrt(2 / shape[0])] = np.sqrt(2 / shape[0])

    vh_rmse: NDArray[np.floating] = (
        (measurement_error / svd_vals) * np.sqrt([
            (((shape[1] - 1) / shape[1]) + (1 / shape[1])
            * np.sum([(m * (3 * (svd_vals[j] ** 2) - m)) / np.square(m - (svd_vals[j] ** 2))
            for m in np.square(np.delete(svd_vals, j))])) for j in range(len(svd_vals))
        ])
    )
    vh_rmse[vh_rmse >= np.sqrt(2 / shape[1])] = np.sqrt(2 / shape[1])

    return u_rmse, vh_rmse

def calculate_minimum_loss_rank(
        vh_rmse: NDArray[np.floating], 
        shape: tuple[int, int],
        threshold: float = 0.05
    ) -> int:
    """
    Generates an estimate for the rank of minimum-loss reconstruction.
    This is based off of equation 14 in (1):
    tₖ = log(RMSE(ṽₖ)) - log(√(2/D)) / log(RMSE(ṽ₁)) - log(√(2/D))
    tₖ quantifies the "cleanliness" of a mode, with t₁ = 1 and tₖ = 0 for 
    modes at the noise ceiling (RMSE(ṽₖ) = √(2/D)). Modes sufficiently below
    the noise ceiling (large tₖ) are deemed clean enough for reconstruction. 

    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    vh_rmse : NDArray[np.floating]
        Array of root mean square errors for the right singular vectors.
    shape : tuple[int, int]
        Shape of the original array.
    threshold : float, optional
        Threshold value determining the cutoff criteria for estimating the
        rank of the minimum-loss reconstruction. tₖ = 1 for the cleanest modes
        and tₖ = 0 for modes at the noise floor. By default 0.05.
    
    Returns
    -------
    int
        Rank of the minimum-loss reconstruction.
    """

    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    t_k: NDArray[np.floating] = (
        (np.log(vh_rmse) - np.log(np.sqrt(2 / shape[0]))) 
        / (np.log(vh_rmse[0]) - np.log(np.sqrt(2 / shape[0])))
    )
    r_min: int = np.argmax((t_k > threshold).nonzero()) + 1

    return r_min

def calculate_clean_singular_values(
        svd_vals: NDArray[np.floating], 
        unit_error: NDArray[np.floating], 
        error_prime: float, 
        measurement_error: float, 
        shape: tuple[int, int],
        method: Literal['e15', 'ek18'] = 'e15'
    ) -> NDArray[np.floating]:
    """
    Generates an estimate of the clean singular values.
    
    Based on equations 16 and 19 from (1). In both cases, a critical index
    is chosen such that the singular values fall below a certain noise
    threshold. Observation and perturbation theory predicts ᵴₖ ≈ sₖ² + (ϵ'ŝₖ)² 
    and ᵴₖ ≈ sₖ + ½(ε̄²D / sₖ), respectively, giving the basis for each reconstruction.
    The E15 reconstruction chooses the critical index kₜ such that ᵴₖ < ϵ'ŝₖ. The 
    clean singular values are then reconstructed using equation 16:
    (16) s̄ₖ = √[ᵴₖ² - (ϵ'ŝₖ)²] for k < kₜ, 0 otherwise.
    The EK18 reconstruction chooses the critical index kₜ such that 
    ᵴₖ < max{ε̄√2D, ε̄(√D + √fT)}. The clean singular values are then 
    reconstructed using equation 19:
    (19) s̄ₖ = ½(ᵴₖ + √[ᵴₖ² - 2ε̄²D]) for k < kₜ, 0 otherwise.

    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.

    Parameters
    ----------
    svd_vals : NDArray[np.floating]
        Array of singular values.
    unit_error : NDArray[np.floating]
        Array of unit error singular values.
    error_prime : float
        Best fit error estimate.
    measurement_error : float
        Estimate of true measurement error.
    shape : tuple[int, int]
        Shape of the original array.
    method : Literal['e15', 'ek18'], optional
        Method used to estimate clean singular values. By default 'e15'

    Returns
    -------
    NDArray[np.floating]
        Array of clean singular values.
    """
    
    if method == 'e15':
        # Estimate clean singular values from the assumption that ᵴₖ² ≈ sₖ² + (ϵ'*ŝₖ)².
        e15_critical_index: int = (
            np.min((svd_vals < unit_error * error_prime).nonzero())
        )

        e15_clean_svd_vals: NDArray[np.floating] = np.zeros_like(svd_vals)
        e15_clean_svd_vals[:e15_critical_index] = (
            np.sqrt(np.square(svd_vals[:e15_critical_index]) 
                    - np.square(unit_error[:e15_critical_index] * error_prime))
        )

        return e15_clean_svd_vals

    elif method == 'ek18':
        # Estimate clean singular values from the assumption that ᵴₖ ≈ sₖ + ½(ε̄²D / sₖ).
        ek18_critical_index: int = np.min((
            svd_vals < np.max((measurement_error * np.sqrt(2 * shape[1]), 
            measurement_error * (np.sqrt(shape[0]) + np.sqrt(shape[1]))))
        ).nonzero())

        ek18_clean_svd_vals: NDArray[np.floating] = np.zeros_like(svd_vals)
        ek18_clean_svd_vals[:ek18_critical_index] = (
            0.5 * (svd_vals[:ek18_critical_index] 
                + (np.sqrt(np.square(svd_vals[:ek18_critical_index]) 
                - 2 * np.square(measurement_error) * shape[1])
            ))
        )

        return ek18_clean_svd_vals

    else:
        raise ValueError("method must be either 'e15' or 'ek18'")


def calculate_canonical_angles(
        u_rmse: NDArray[np.floating], 
        vh_rmse: NDArray[np.floating], 
        r_min: int, 
        shape: tuple[int, int]
    ) -> NDArray[np.floating]:
    """
    Calculates the canonical angles between noisy and clean singular vectors.
    Based on equations (21) - (23) in (1).

    (1) Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.


    Parameters
    ----------
    u_rmse : NDArray[np.floating]
        Array of RMSE estimates for the left singular vectors.
    vh_rmse : NDArray[np.floating]
        Array of RMSE estimates for the right singular vectors.
    r_min : int
        Rank of minimum loss reconstruction.
    shape : tuple[int, int]
        Shape of the original array.

    Returns
    -------
    NDArray[np.floating]
        _description_
    """
    cos_phi_k: NDArray[np.floating] = 1 - shape[0] * np.square(u_rmse)
    cos_theta_k: NDArray[np.floating] = 1 - shape[1] * np.square(vh_rmse)
    cl_estimate: NDArray[np.floating] = cos_phi_k * cos_theta_k
    cl_estimate[r_min:] = 0
    return cl_estimate
