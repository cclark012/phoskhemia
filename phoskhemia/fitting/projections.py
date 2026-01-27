import numpy as np
from numpy.typing import NDArray

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
    traces: np.ndarray = np.asarray(traces, dtype=float)
    data: np.ndarray = np.asarray(data, dtype=float)

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
    traces_w: np.ndarray = traces * weight
    data_w: np.ndarray = data * weight

    # Regularized normal equations
    CTC: np.ndarray = traces_w.T @ traces_w + lam * np.eye(n_species)
    CTy: np.ndarray = traces_w.T @ data_w
    coeffs: np.ndarray
    try:
        coeffs = np.linalg.solve(CTC, CTy)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(CTC, CTy, rcond=None)[0]

    # Residuals (unweighted)
    data_fit: np.ndarray = traces @ coeffs
    residuals: np.ndarray = data - data_fit

    # Residual variance (weighted)
    RSS: float = np.sum((weight * residuals) ** 2)
    dof: int = max(n_times - n_species, 1)
    sigma2: float = RSS / dof

    # Covariance
    cov: np.ndarray = sigma2 * np.linalg.inv(CTC)

    return coeffs, cov, residuals
