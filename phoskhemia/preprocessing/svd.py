"""
SVD denoising and minimum-loss reconstruction (Epps & Krivitzky).

Implements:
- MP tail fit for estimating measurement error (epsilon-bar)
- Mode RMSE / cleanliness metric t_k
- Minimum-loss rank selection
- Clean singular value estimation:
- E15 (Mode Corruption, 2019)
- EK18 (Noise Filtering, 2019)

Conventions
-----------
Given data matrix A (T×D), we compute A = U diag(s) V^T with K=min(T,D).
We use lambda_k = s_k^2.

Public API
----------
- svd_denoise(...): orchestrator returning A_hat (+ details optionally)
- svd_rank(...): rank estimate
- svd_components(...): (C(t), S(lambda)) for initialization

Notes
-----
This implementation assumes i.i.d. additive noise in the matrix elements after any
optional centering/whitening. Whitening by σ(λ) is supported via weights.

Background
----------
Definitions
===========

Variables
^^^^^^^^^
    ϵ' - Fit Measurement Error
    ϵ" - Preliminary Estimate of Measurement Error
    ϵ - True Error, generally ϵ > ϵ" and ϵ < ϵ'
    ε̄ - Estimate of the true measurement error
    T - Number of rows of matrix A
    D - Number of columns of matrix A
    K - Number of singular values
    ᵴ - Noisy singular values
    ŝ - "Unit" Marchenko-Pastur distribution
    kₑ - Critical index of best fit
    kₑ' - Critical index of estimate

Equations
^^^^^^^^^
Generates an estimate for the rank of minimum-loss reconstruction.
This is based off of equation 14 in (1):
tₖ = log(RMSE(ṽₖ)) - log(√(2/D)) / log(RMSE(ṽ₁)) - log(√(2/D))
tₖ quantifies the "cleanliness" of a mode, with t₁ = 1 and tₖ = 0 for 
modes at the noise ceiling (RMSE(ṽₖ) = √(2/D)). Modes sufficiently below
the noise ceiling (large tₖ) are deemed clean enough for reconstruction. 

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

Construct estimates of error in data.



(27) log₁₀(ϵ') = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ᵴₗ) - log₁₀(ŝₗ))
(26) L = (1 / (T + 1 - k)) * Σₗ₌ₖᵀ (log₁₀(ϵ') + log₁₀(ŝₗ) - log₁₀(ᵴₗ))²
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

P(y) = (1 / 2πy) ∙ {2√(ab} ∙ [tan⁻¹((a(b - z) / b(z - a))¹ᐟ²) - π / 2]
    + ((a + b) / 2) ∙ [tan⁻¹((z - ½(a + b)) / ((b - z)(z - a))¹ᐟ²) + π / 2]
    + ((b - z)(z - a))¹ᐟ²}

References
----------
[1] Epps, B.P.; Krivitzky, E.M. Mode Corruption. Exp Fluids 2019.
[2] Epps, B.P.; Krivitzky, E.M. Noise Filtering. Exp Fluids 2019.
"""

from typing import Literal, Any
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.linalg import svd

def marchenko_pastur_pdf(
        x: NDArray[np.floating], 
        lam: float
    ) -> NDArray[np.floating]:
    """
    Marchenko-Pastur probability density function for a chosen lam value. 
    lam is usually the ratio of rows to columns.
    Based on equations 117 and 118 from [1]_.

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

    References
    ----------
    .. [1] Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.
    """

    x: NDArray[np.floating] = np.asarray(x).reshape(-1)
    x[x <= 0] = np.min(x[x > 0])

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


    Based on equation 119 in [1]_.

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

    References
    ----------
    .. [1] Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.
    """

    lamminus: float = (1 - np.sqrt(lam)) ** 2
    lamplus: float = (1 + np.sqrt(lam)) ** 2
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
    cdf = np.clip(cdf, 0, 1)

    return cdf



def svd_reconstruction(
        arr: NDArray[np.floating],
        method: Literal['e15', 'ek18'] = 'e15',
        value_rotation: Literal['include', 'exclude', 'auto'] = 'exclude',
        threshold: float = 0.05,
        return_details: bool = False
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], dict[str, Any]]:
    """
    Calculates and constructs the optimal reconstruction of an array by SVD.

    Parameters
    ----------
    arr : NDArray[np.floating]
        Array to reconstruct.
    method : Literal['e15', 'ek18'], optional
        Method to use to reconstruct the array, by default 'e15'.
    value_rotation : Literal['include', 'exclude', 'auto'], optional
        Whether to include mixing factors to scale singular values, by default 'exclude'.
    threshold : float, optional
        Threshold value determining the cutoff criteria for estimating the
        rank of the minimum-loss reconstruction. tₖ = 1 for the cleanest modes
        and tₖ = 0 for modes at the noise floor. By default 0.05.
    return_details : bool, optional

    Returns
    -------
    NDArray[np.floating] | tuple[NDArray[np.floating], dict[str, Any]]
        The optimally reconstructed array.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_

    References
    ----------
    .. [1] Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Mode Corruption. Exp Fluids 2019, 60 (8), 121. 
        https://doi.org/10.1007/s00348-019-2761-y.
    
    .. [2] Epps, B. P.; Krivitzky, E. M. 
        Singular Value Decomposition of Noisy Data: 
        Noise Filtering. Exp Fluids 2019, 60 (8), 126. 
        https://doi.org/10.1007/s00348-019-2768-4.
    """

    arr: NDArray[np.floating] = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("arr must be two-dimensional")

    if method not in ['e15', 'ek18']:
        raise ValueError("method must be either 'e15' or 'ek18'")

    if value_rotation not in ['include', 'exclude', 'auto']:
        raise ValueError("value_rotation must be one of 'include', 'exclude', or 'auto'")

    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    shape: tuple[int, int] = arr.shape
    T: int # rows
    D: int # cols
    T, D = shape

    transposed: bool = False
    if T > D:
        transposed = True
        arr = arr.T
        T, D = D, T

    U: NDArray[np.floating] # MxK
    S: NDArray[np.floating] # (K,)
    V: NDArray[np.floating] # KxN
    U, S, V = svd(arr, full_matrices=False, lapack_driver='gesvd')

    K: int = len(S)

    # Use aspect ratio for Marchenko-Pastur distribution of eigenvalues.
    lam: float = T / D
    lamminus: float = (1 - np.sqrt(lam)) ** 2
    lamplus: float = (1 + np.sqrt(lam)) ** 2

    # Range over which to interpolate.
    eigen_range: NDArray[np.floating] = (
        np.linspace(lamminus, lamplus, K * 2)
    )

    # Marchenko-Pastur cumulative eigenvalue distribution.
    cdf: NDArray[np.floating] = (
        marchenko_pastur_cdf(x=eigen_range, lam=lam)
    )

    # Even spacing across the range of values.
    interval: NDArray[np.floating] = (
        1 - (np.arange(0, K, 1, dtype=int) / (K - 1))
    )
    mask = np.diff(cdf, prepend=0) > 0

    # Interpolate to find evenly spaced eigenvalues.
    eigens: NDArray[np.floating] = (
        PchipInterpolator(
            cdf[mask], eigen_range[mask]
        )(interval)
    )

    # Find 'unit error' singular values, ŝₖ = √Dλₖ.
    unit_s: NDArray[np.floating] = np.sqrt((D * eigens))
    
    tiny = np.finfo(float).smallest_normal
    S: NDArray[np.floating] = np.maximum(S, tiny)
    unit_s: NDArray[np.floating] = np.maximum(unit_s, tiny)

    # Mean square error and loss function for fit to Marchenko-Pastur distribution, ϵ'ŝₗ.
    k: NDArray[np.int64] = np.arange(0, K, 1, dtype=np.int64)
    k_max: int = int(np.floor(0.8 * K))
    idx = (K - 1) - k

    # Precompute dₗ = log₁₀(ᵴₗ) - log₁₀(ŝₗ).
    a: NDArray[np.floating] = np.log10(S)
    b: NDArray[np.floating] = np.log10(unit_s)
    diff: NDArray[np.floating] = a - b

    # Cumulative sum over reversed array allows us to index into precomputed Σₗ₌ₖᵀ values.
    diff_rev: NDArray[np.floating] = diff[::-1]
    c1_rev: NDArray[np.floating] = np.cumsum(diff_rev)
    c2_rev: NDArray[np.floating] = np.cumsum(diff_rev * diff_rev)
    tail_sum1: NDArray[np.floating] = c1_rev[idx]
    tail_sum2: NDArray[np.floating] = c2_rev[idx]

    # Factor of T + 1 - k
    n: NDArray[np.floating] = (K - np.arange(K)).astype(float)

    # log₁₀(ϵ') = (1 / n) * Σₗ₌ₖᵀ (log₁₀(ᵴₗ) - log₁₀(ŝₗ)) = μₖ
    log_e: NDArray[np.floating] = tail_sum1 / n
    # MSEₖ = (1 / nₖ) ∙ Σₗ₌ₖᵀ (log₁₀(ϵ') + b - a)² = (1 / nₖ) ∙ Σₗ₌ₖᵀ (μₖ - dₗ)²
    var: NDArray[np.floating] = (tail_sum2 / n) - log_e * log_e
    mse: NDArray[np.floating] = np.maximum(var, 0.0)

    # Index where the best fit is achieved for ϵ'ŝₗ.
    fit_min: int = np.argmin(mse[:k_max])

    # Fit measurement error, ϵ'.
    e_prime: float = np.power(10., log_e[fit_min])
    # Preliminary estimate of measurement error, ϵ".
    e_dprime: float = (e_prime * unit_s[fit_min]) / np.sqrt(D)
    
    # Index of first value to fall below ϵ'√D.
    min_cond: NDArray[np.bool_] = (S < e_prime * np.sqrt(D))
    est_min: int = int(np.argmax(min_cond)) if min_cond.any() else K

    # Estimate the true measurement error ε̄.
    m_error: float = np.min((e_prime, e_dprime + (e_prime - e_dprime) * ((fit_min - est_min) / (np.floor(0.8*T) - (est_min + 1)))))

    # Calculate root mean square error for the left and right singular vectors.
    S_safe: NDArray[np.floating] = np.maximum(S, np.finfo(float).smallest_normal)
    lamv: NDArray[np.floating] = S ** 2
    lam_i: NDArray[np.floating] = lamv[:, None]
    lam_j: NDArray[np.floating] = lamv[None, :]
    cdf_mask: NDArray[np.bool_] = ~np.eye(K, dtype=bool)
    den: NDArray[np.floating] = (lam_j - lam_i) ** 2
    den = np.where(cdf_mask, den, np.inf) # avoid divide by zero on diagonal

    num_u: NDArray[np.floating] = lam_i * (lam_j + lam_i)
    sum_u: NDArray[np.floating] = np.sum(num_u / den, axis=1)
    u_rmse: NDArray[np.floating] = (m_error / S_safe) * np.sqrt((1.0 / T) * sum_u)
    u_rmse = np.minimum(u_rmse, np.sqrt(2.0 / T))

    num_v: NDArray[np.floating] = lam_j * (3.0 * lam_i - lam_j)
    sum_v: NDArray[np.floating] = np.sum(num_v / den, axis=1)
    v_rmse: NDArray[np.floating] = (m_error / S_safe) * np.sqrt(((D - 1.0) / D) + (1.0 / D) * sum_v)
    v_rmse = np.minimum(v_rmse, np.sqrt(2.0 / D))


    # Rank of minimum loss reconstruction.
    t_k: NDArray[np.floating] = (
        (np.log(v_rmse) - np.log(np.sqrt(2 / D))) 
        / (np.log(v_rmse[0]) - np.log(np.sqrt(2 / D)))
    )
    r_cond: NDArray[np.bool_] = (t_k > threshold)
    r_min: int = int(np.argmax(~r_cond))
    r_min = r_min if r_cond.any() and (~r_cond).any() else (K if r_cond.all() else 1)

    # Critical index and form of clean singular values differs depending on method.
    if method == 'e15':
        c_cond: NDArray[np.bool_] = (S < unit_s * e_prime)
        c_index: int = int(np.argmax(c_cond)) if c_cond.any() else K
        clean_s: NDArray[np.floating] = np.zeros_like(S, dtype=float)
        clean_s[:c_index] = np.sqrt(np.square(S[:c_index]) - np.square(unit_s[:c_index] * e_prime))
    
    elif method == 'ek18':
        c_cond: NDArray[np.bool_] = (S < np.max((m_error * np.sqrt(2 * D), m_error * (np.sqrt(T) + np.sqrt(D)))))
        c_index: int = int(np.argmax(c_cond)) if c_cond.any() else K
        clean_s: NDArray[np.floating] = np.zeros_like(S, dtype=float)
        clean_s[:c_index] = 0.5 * (S[:c_index] + (np.sqrt(np.square(S[:c_index]) - 2 * np.square(m_error) * D)))

    if value_rotation in ['include', 'auto']:
        # Construct an estimate of the canonical angle product.
        cos_phi_k: NDArray[np.floating] = 1 - T * np.square(u_rmse)
        cos_theta_k: NDArray[np.floating] = 1 - D * np.square(v_rmse)
        cl_est: NDArray[np.floating] = cos_phi_k * cos_theta_k
        cl_est[r_min:] = 0

        # Calculate approximate loss function, minimum, and merit of reconstruction.
        approx_loss_cl: NDArray[np.floating] = np.array([(m_error ** 2) * D * i + np.sum(np.square(clean_s[i:] * cl_est[i:])) for i in np.arange(K)])
        merit_cl: float = ((m_error ** 2) * T * D - approx_loss_cl[r_min]) / ((m_error ** 2) * T * D)
        if value_rotation == 'include':
            loss: float = approx_loss_cl[r_min]
            r_loss: int = r_min
            merit: float = merit_cl
            reconstruction: NDArray[np.floating] = U[:, :r_min] @ np.diag((clean_s[:r_min] * cl_est[:r_min])) @ V[:r_min, :]

    if value_rotation in ['exclude', 'auto']:
        # Calculate approximate loss function, minimum, and merit of reconstruction.
        approx_loss: NDArray[np.floating] = np.array([(m_error ** 2) * D * i + np.sum(np.square(clean_s[i:])) for i in np.arange(K)])
        merit: float = ((m_error ** 2) * T * D - approx_loss[r_min]) / ((m_error ** 2) * T * D)
        if value_rotation == 'exclude':
            r_loss: int = r_min
            loss: float = approx_loss[r_min]
            reconstruction: NDArray[np.floating] = U[:, :r_min] @ np.diag(clean_s[:r_min]) @ V[:r_min, :]

    if value_rotation == 'auto':
        loss_cl_rmin: float = (m_error ** 2) * D * r_min + np.sum(np.square(clean_s[r_min:] * cl_est[r_min:]))
        loss_rmin: float = (m_error ** 2) * D * r_min + np.sum(np.square(clean_s[r_min:]))
        merit_cl_rmin: float = ((m_error ** 2) * T * D - loss_cl_rmin) / ((m_error ** 2) * T * D)
        merit_rmin: float = ((m_error ** 2) * T * D - loss_rmin) / ((m_error ** 2) * T * D)
        r_loss: int = np.argmin(approx_loss)
        r_loss_cl: int = np.argmin(approx_loss_cl)

        # 'Best' is chosen to be the method that has the highest figure of merit.
        best: int = np.argmax([merit, merit_rmin, merit_cl, merit_cl_rmin])
        r_loss: int = [r_loss, r_min, r_loss_cl, r_min][best]
        merit: float = [merit, merit_rmin, merit_cl, merit_cl_rmin][best]
        loss: float = [approx_loss[r_loss], loss_rmin, approx_loss_cl[r_loss_cl], loss_cl_rmin][best]
        reconstruction: NDArray[np.floating] = (
            U[:, :r_loss] @ np.diag(clean_s[:r_loss]) @ V[:r_loss, :] 
            if best < 2 else U[:, :r_loss] @ np.diag((clean_s[:r_loss] * cl_est[:r_loss])) @ V[:r_loss, :]
        )

    if transposed:
        arr = arr.T
        reconstruction = reconstruction.T

    if return_details:
        resid_std: float = np.std((arr - reconstruction))
        # First mode that fails the test of ᵴₖ > ε̄√DT.
        kf: int = np.argmax((S < m_error * np.sqrt(D * T)))
        # Rough estimate of minimum-loss reconstruction rank.
        k2: int = np.argmax((S < m_error * (np.sqrt(D) + np.sqrt(T))))
        # Minimum index for where singular values and Marchenko-Pastur overlay one another.
        ke: int = np.argmax((S < m_error * np.sqrt(D)))

        info: dict[str, Any] = {
            "e_prime": e_prime,
            "e_dprime": e_dprime,
            "error": m_error,
            "rank": r_loss,
            "merit": merit,
            "loss": loss,
            "threshold": threshold,
            "shape": shape,
            "transposed": transposed,
            "c_index": c_index,
            "method": method,
            "resid_std": resid_std,
            "kf": kf,
            "k2": k2,
            "ke": ke,
            "fit_min": fit_min,
            "est_min": est_min,
            "value_rotation": value_rotation
        }
        return reconstruction, info

    else:
        return reconstruction

if __name__ == "__main__":
    m_error = 1.e-3






