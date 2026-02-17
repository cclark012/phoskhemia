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
from __future__ import annotations

from typing import Literal, Any, TypedDict, Sequence, NotRequired
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.linalg import svd

class Candidate(TypedDict):
    name: Literal['include', 'exclude']
    r: int
    loss: float
    merit: float
    use_rotation: bool

class Selection(TypedDict):
    value_rotation: Literal["exclude", "include", "auto"]
    best: Candidate
    candidates: list[Candidate]

class SVDReconstructionInfo(TypedDict):
    method: Literal["e15", "ek18"]
    value_rotation: Literal["exclude", "include", "auto"]
    shape: tuple[int, int]
    transposed: bool

    e_prime: float
    e_dprime: float
    error: float

    r_min: int
    selection: Selection

    resid_std: NotRequired[float]
    fit_min: NotRequired[int]
    est_min: NotRequired[int]

def _build_candidates(
        *,
        approx_loss_ex: NDArray[np.floating],
        approx_loss_in: NDArray[np.floating] | None,
        r_min: int,
        denom: float,
    ) -> list[Candidate]:
    """
    Build reconstruction candidates (exclude/include × {argmin(loss), r_min}).

    approx_loss_* are length-K arrays indexed by r-1 (0-based).
    r_min is a rank in [1, K].
    denom is (m_error^2)*T*D (or other positive normalizer) used for merit.
    """

    K: int = int(approx_loss_ex.size)
    if K <= 0:
        raise ValueError("loss curve must be non-empty")
    if approx_loss_in is not None and int(approx_loss_in.size) != K:
        raise ValueError("approx_loss_in must have the same length as approx_loss_ex")

    def clamp_rank(r: int) -> int:
        return int(np.clip(int(r), 1, K))

    def merit_from_loss(loss: float) -> float:
        if denom <= 0.0 or not np.isfinite(denom):
            return float("nan")
        return float((denom - loss) / denom)

    def make(
            name: Literal["exclude", "include"], 
            r: int, 
            loss_curve: NDArray[np.floating], 
            use_rotation: bool
        ) -> Candidate:

        r: int = clamp_rank(r)
        loss: float = float(loss_curve[r - 1])
        return {
            "name": name,
            "r": r,
            "loss": loss,
            "merit": merit_from_loss(loss),
            "use_rotation": use_rotation,
        }

    cands: list[Candidate] = []

    # exclude
    r_loss_ex: int = int(np.argmin(approx_loss_ex)) + 1
    cands.append(make("exclude", r_loss_ex, approx_loss_ex, use_rotation=False))
    cands.append(make("exclude", r_min, approx_loss_ex, use_rotation=False))

    # include (optional)
    if approx_loss_in is not None:
        r_loss_in = int(np.argmin(approx_loss_in)) + 1
        cands.append(make("include", r_loss_in, approx_loss_in, use_rotation=True))
        cands.append(make("include", r_min, approx_loss_in, use_rotation=True))

    return cands

def _select_candidate(
        *,
        candidates: Sequence[Candidate],
        value_rotation: Literal["exclude", "include", "auto"],
        key: Literal["merit", "loss"] = "merit",
    ) -> Candidate:
    """
    Select the best candidate.

    key="merit": maximize merit (preferred).
    key="loss": minimize loss.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")

    if value_rotation != "auto":
        candidates = [c for c in candidates if c["name"] == value_rotation]
        if not candidates:
            raise ValueError(f"no candidates available for value_rotation={value_rotation!r}")

    if key == "loss":
        return min(candidates, key=lambda c: c["loss"])

    # key == "merit"
    finite: list[Candidate] = [c for c in candidates if np.isfinite(c["merit"])]
    if finite:
        return max(finite, key=lambda c: c["merit"])

    # fallback if denom invalid: minimize loss
    return min(candidates, key=lambda c: c["loss"])

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

    arr0: NDArray[np.floating] = np.asarray(arr, dtype=float)
    if arr0.ndim != 2:
        raise ValueError("arr must be two-dimensional")

    if method not in ['e15', 'ek18']:
        raise ValueError("method must be either 'e15' or 'ek18'")

    if value_rotation not in ['include', 'exclude', 'auto']:
        raise ValueError("value_rotation must be one of 'include', 'exclude', or 'auto'")

    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    shape: tuple[int, int] = arr0.shape
    T: int # rows
    D: int # cols
    T, D = shape

    transposed: bool = False
    if T > D:
        transposed = True
        arr = arr0.T
        T, D = D, T
    else:
        arr = arr0

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


    denom = float((m_error ** 2) * T * D)  # used for merit; if <=0 => merit becomes NaN

    cl_est: NDArray[np.floating] | None = None
    approx_loss_in: NDArray[np.floating] | None = None
    if value_rotation in ("include", "auto"):
        # Construct an estimate of the canonical angle product.
        cos_phi_k: NDArray[np.floating] = 1.0 - T * np.square(u_rmse)
        cos_theta_k: NDArray[np.floating] = 1.0 - D * np.square(v_rmse)
        cl_est = cos_phi_k * cos_theta_k

        cl_est = np.clip(cl_est, 0.0, 1.0)

        if 1 <= r_min < K:
            cl_est[r_min:] = 0.0

    clean_s2: NDArray[np.floating] = np.square(clean_s)

    # Tail sums: tail_sum2[r] = sum_{i=r}^{Kdim-1} clean_s[i]^2
    tail_sum2: NDArray[np.floating] = np.cumsum(clean_s2[::-1])[::-1]

    tail_after_rank: NDArray[np.floating] = np.empty(K + 1, dtype=float)
    tail_after_rank[:K] = tail_sum2
    tail_after_rank[K] = 0.0

    r_grid: NDArray[np.int64] = np.arange(1, K + 1, dtype=int)
    approx_loss_ex: NDArray[np.floating] = (m_error ** 2) * D * r_grid + tail_after_rank[r_grid]

    if cl_est is not None:
        # Rotated tail energy uses (clean_s * cl_est)^2
        clean_rot2: NDArray[np.floating] = np.square(clean_s * cl_est)
        tail_rot2: NDArray[np.floating] = np.cumsum(clean_rot2[::-1])[::-1]
        tail_after_rank_rot: NDArray[np.floating] = np.empty(K + 1, dtype=float)
        tail_after_rank_rot[:K] = tail_rot2
        tail_after_rank_rot[K] = 0.0
        approx_loss_in: NDArray[np.floating] = (m_error ** 2) * D * r_grid + tail_after_rank_rot[r_grid]

    # Candidate build and selection 
    candidates: list[Candidate] = _build_candidates(
        approx_loss_ex=approx_loss_ex,
        approx_loss_in=approx_loss_in,
        r_min=r_min,
        denom=denom,
    )
    best: Candidate = _select_candidate(
        candidates=candidates,
        value_rotation=value_rotation,
        key="merit",  # or "loss"
    )

    # Reconstruction from best candidate 
    r_best: int = int(best["r"])
    use_rotation = bool(best["use_rotation"])

    if use_rotation:
        if cl_est is None:
            raise RuntimeError("internal error: selected include candidate without cl_est")
        s_eff: NDArray[np.floating] = clean_s[:r_best] * cl_est[:r_best]
    else:
        s_eff: NDArray[np.floating] = clean_s[:r_best]

    # U[:, :r] @ diag(s_eff) @ V[:r, :] without forming diag
    reconstruction_work: NDArray[np.floating] = U[:, :r_best] @ (s_eff[:, None] * V[:r_best, :])

    # Restore original orientation
    reconstruction: NDArray[np.floating] = reconstruction_work.T if transposed else reconstruction_work

    # Optional details
    if return_details:
        # Residual stats against the original (non-transposed) input array
        resid: NDArray[np.floating] = arr0 - reconstruction
        resid_std: float = float(np.std(resid, ddof=1))

        # “First index” style diagnostics should be guarded.
        def _first_true_or_default(cond: NDArray[np.bool_], default: int) -> int:
            return int(np.argmax(cond)) if bool(np.any(cond)) else int(default)

        kf: int = _first_true_or_default(S < (m_error * np.sqrt(D * T)), default=K)   # “fails ε̄√(DT)”
        k2: int = _first_true_or_default(S < (m_error * (np.sqrt(D) + np.sqrt(T))), default=K)
        ke: int = _first_true_or_default(S < (m_error * np.sqrt(D)), default=K)

        # Selection payload (TypedDict)
        selection: Selection = {
            "value_rotation": value_rotation,
            "best": best,
            "candidates": list(candidates),
        }

        info: SVDReconstructionInfo = {
            "method": method,
            "value_rotation": value_rotation,
            "shape": shape,
            "transposed": transposed,

            "e_prime": float(e_prime),
            "e_dprime": float(e_dprime),
            "error": float(m_error),

            "r_min": int(r_min),
            "selection": selection,

            # Optional / useful extras:
            "resid_std": resid_std,
            "fit_min": int(fit_min),
            "est_min": int(est_min),
            "c_index": int(c_index),
            "threshold": float(threshold),

            "kf": int(kf),
            "k2": int(k2),
            "ke": int(ke),
        }

        return reconstruction, info

    return reconstruction

if __name__ == "__main__":
    m_error = 1.e-3






