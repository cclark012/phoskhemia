import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel

def project_amplitudes(
        traces: NDArray[np.floating], 
        data: NDArray[np.floating], 
        noise: float, 
        lam: float
    ) -> tuple[
        NDArray[np.floating], 
        NDArray[np.floating], 
        NDArray[np.floating]
    ]:
    """
    Project kinetic basis functions onto data using
    weighted, Tikhonov-regularized least squares.
    
    Tikhonov regularization or Ridge regression is a method 
    to solve ill-posed inverse problems, such as estimating 
    the coefficients of multiple-regression models when 
    variables are highly correlated. Ill-posed problems 
    often have solutions that are not unique (if solvable
    at all). In ordinary least squares, the problem of 
    finding the solution to n linear equations with p 
    unknown coefficients is stated as: Σⱼᵖ xᵢⱼ ⋅ βⱼ = yᵢ, 
    (i = 1, 2, ..., n) or, in matrix notation, 𝐘 = 𝐗𝛃, 
    where 𝐘 is an nx1 vector of the response variables 
    (or data to fit), 𝐗 is an nxp matrix of 
    regressors/parameters (sometimes called the design 
    matrix), and 𝛃 is a px1 vector of unknown parameters.
    While no exact solution is usually possible (especially 
    when dealing with real observations for 𝐘), the "best" 
    solution to the least squares problem is usually chosen
    such that 𝛃 = argminᵦ S(β) is solved for the objective
    function S(β) = Σᵢ₌₁ⁿ |yᵢ - Σⱼ₌₁ᵖ Xᵢⱼ ⋅ βⱼ|² = ∥𝐘 - 𝐗𝛃∥²,
    which in this case is the quadratic form. If the problem
    is well-posed (i.e. the p columns of 𝐗 are linearly
    independent), then there is a unique solution given by
    solving the normal equations: (𝐗ᵀ𝐗)𝛃 = 𝐗ᵀ𝐘, where 𝐗ᵀ𝐗
    is the normal/moment/Gram matrix and 𝐗ᵀ𝐘 is the moment 
    matrix of regressand by regressors. 𝛃 is the coefficient 
    vector of the least squares hyperplane, expressed as 
    𝛃 = (𝐗ᵀ𝐗)⁻¹𝐗ᵀ𝐘 or 𝛃 = β + (𝐗ᵀ𝐗)⁻¹𝐗ᵀε. If the columns of
    𝐗 are not linearly dependent, then 𝐗 is a singular 
    matrix (det(𝐗) = 0) and the solution is not unique. 
    Tikhonov regularization alleviates the issue of the 
    near-singular moment matrix 𝐗ᵀ𝐗 by adding positive 
    elements on the diagonal. This decreases the condition
    number (effectively how much a function changes for
    a small change of input). This is accomplished by an 
    extra term λ𝐈: 𝛃 = (𝐗ᵀ𝐗 + λ𝐈)⁻¹𝐗ᵀ𝐘. This estimator is
    the solution to the least squares problem with the 
    constraint 𝛃ᵀ𝛃 = c, which can be expressed as a
    Lagrangian minimization: argminᵦ ∥𝐘 - 𝐗𝛃∥² + λ(𝛃ᵀ𝛃 - c).
    As λ approaches 0, the constraint becomes non-binding
    and the ordinary least squares estimator is recovered.

    Parameters
    ----------
    traces : ndarray, shape (n_times, n_species)
        Kinetic basis functions
    data : ndarray, shape (n_times,)
        Observed data at one wavelength
    noise : float
        Noise estimate for this wavelength (σ)
    lam : float
        Tikhonov regularization strength

    Returns
    -------
    coeffs : ndarray, shape (n_species,)
        Fitted amplitudes
    cov : ndarray, shape (n_species, n_species)
        Covariance of amplitudes (conditional on kinetics)
    residuals : ndarray, shape (n_times,)
        Unweighted residuals y - C @ coeffs
    """

    # Safety checks
    traces: NDArray[np.floating] = np.asarray(traces, dtype=float)
    data: NDArray[np.floating] = np.asarray(data, dtype=float)

    # Catch 1D traces for cases with one species
    if traces.ndim == 1:
        traces = traces[:, None]

    elif traces.ndim != 2:
        raise ValueError("traces must be 2D (n_times, n_species)")

    if data.ndim != 1:
        raise ValueError("data must be 1D (n_times,)")

    if traces.shape[0] != data.shape[0]:
        raise ValueError("traces and data length mismatch")

    # Treat non-positive noise as unweighted least squares
    if noise <= 0:
        noise = 1.0

    n_times: int
    n_species: int
    n_times, n_species = traces.shape

    # Weighting
    weight: float = 1.0 / noise
    traces_w: NDArray[np.floating] = traces * weight
    data_w: NDArray[np.floating] = data * weight

    # Regularized normal equations (Tikhonov regularization)
    CTC: NDArray[np.floating] = traces_w.T @ traces_w + lam * np.eye(n_species)
    CTy: NDArray[np.floating] = traces_w.T @ data_w
    coeffs: NDArray[np.floating]
    try:
        coeffs = np.linalg.solve(CTC, CTy)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(CTC, CTy, rcond=None)[0]

    # Residuals (unweighted)
    data_fit: NDArray[np.floating] = traces @ coeffs
    residuals: NDArray[np.floating] = data - data_fit

    # Residual variance (weighted)
    RSS: float = np.sum((weight * residuals) ** 2)
    dof: int = max(n_times - n_species, 1)
    sigma2: float = RSS / dof

    # Covariance
    cov: NDArray[np.floating] = sigma2 * np.linalg.inv(CTC)

    return coeffs, cov, residuals

def propagate_kinetic_covariance(
        kinetic_model: KineticModel,
        times: NDArray[np.floating],
        beta: NDArray[np.floating],
        coeffs: NDArray[np.floating],
        cov_beta: NDArray[np.floating],
        data: NDArray[np.floating],
        noise: float,
        lam: float,
        eps=1e-6,
    ) -> NDArray[np.floating]:
    """
    Propagate kinetic parameter covariance into amplitude covariance
    (linearized around the fitted parameters).

    Parameters
    ----------
    kinetic_model : KineticModel
    times : ndarray, shape (n_times,)
    beta : ndarray, shape (n_params,)
        Fitted kinetic parameters (log-space)
    coeffs : ndarray, shape (n_species,)
        Amplitudes at this wavelength
    cov_beta : ndarray, shape (n_params, n_params)
        Kinetic covariance matrix
    data : ndarray, shape (n_times,)
        Observed data at this wavelength
    noise : float
        Noise estimate σ
    lam : float
        Tikhonov regularization strength
    eps : float
        Finite-difference step size

    Returns
    -------
    cov_kin : ndarray, shape (n_species, n_species)
        Covariance contribution from kinetic uncertainty
    """

    times: NDArray[np.floating] = np.asarray(times, dtype=float)
    beta: NDArray[np.floating] = np.asarray(beta, dtype=float)
    coeffs: NDArray[np.floating] = np.asarray(coeffs, dtype=float)
    cov_beta: NDArray[np.floating] = np.asarray(cov_beta, dtype=float)
    data: NDArray[np.floating] = np.asarray(data, dtype=float)

    n_params: int = beta.size
    n_species: int = coeffs.size

    if cov_beta.shape != (n_params, n_params):
        raise ValueError("cov_beta shape mismatch")

    if data.shape[0] != times.shape[0]:
        raise ValueError("data and times length mismatch")

    # Empty array for Jacobian
    jacobian: NDArray[np.floating] = np.zeros((n_species, n_params))

    # Compute finite-difference Jacobian
    for k in range(n_params):
        eps_k = eps * max(abs(beta[k]), 1.0)
        dbeta: NDArray[np.floating] = np.zeros_like(beta)
        dbeta[k] = eps_k

        traces_k: NDArray[np.floating] = kinetic_model.solve(times, beta + dbeta)

        coeffs_k: NDArray[np.floating]
        coeffs_k, _, _ = project_amplitudes(
            traces_k,
            data,
            noise,
            lam,
        )

        jacobian[:, k] = (coeffs_k - coeffs) / eps_k

    cov_kin: NDArray[np.floating] = jacobian @ cov_beta @ jacobian.T
    return cov_kin

