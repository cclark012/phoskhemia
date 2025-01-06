import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky

def arpls(
        array: np.typing.ArrayLike, 
        lam: float | int=1e4, 
        ratio: float=0.05, 
        itermax: int=100
    ) -> np.typing.ArrayLike:

    """
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Abstract:
    Baseline correction methods based on penalized least squares are successfully
    applied to various spectral analyses. The methods change the weights iteratively
    by estimating a baseline. If a signal is below a previously fitted baseline,
    large weight is given. On the other hand, no weight or small weight is given
    when a signal is above a fitted baseline as it could be assumed to be a part
    of the peak. As noise is distributed above the baseline as well as below the
    baseline, however, it is desirable to give the same or similar weights in
    either case. For the purpose, we propose a new weighting scheme based on the
    generalized logistic function. The proposed method estimates the noise level
    iteratively and adjusts the weights correspondingly. According to the
    experimental results with simulated spectra and measured Raman spectra, the
    proposed method outperforms the existing methods for baseline correction and
    peak height estimation.

    Other Information:
    Some signal y of length N, sampled at equal intervals, is assumed to be
    composed of a smooth signal z and noise or other perturbations. If both
    y and z are column vectors, z can be found through the minimization of 
    the penalized least squares function (regularized has no weights):
    S(z) = (y - z)ᵀ ⋅ W ⋅ (y - z) + λ ⋅ zᵀ ⋅ Dᵀ ⋅ D ⋅ z, where W is the diagonal
    matrix of weights, λ is a parameter to adjust the balance between the first
    and second terms, and D is the second-order difference matrix. The first 
    term in the above equation represents the fit to the data, while the second
    term expresses the smoothness in z. Setting the vector of partial derivatives
    to zero (∂S / ∂zᵀ = 0) and solving gives the following solution:
    ∂S / ∂zᵀ = -2 ⋅ W ⋅ (y - z) + 2 ⋅ λ ⋅ Dᵀ ⋅ D ⋅ z = 0, 
    z = (W + λ ⋅ Dᵀ ⋅ D)⁻¹ ⋅ W ⋅ y. Iterative adjustements of the weights are
    performed as wᵢ = logistic(yᵢ - zᵢ, m(d⁻), σ(d⁻)) if yᵢ >= zᵢ, else 1, 
    where d = y - z, d⁻ is the negative part of d, and m and σ are the mean
    and standard deviation of d⁻, respectively. The logistic function is
    defined as (1 + exp[2 ⋅ (d - (2 ⋅ σ - m)) / σ])⁻¹, which is a sigmoidal
    function that gives large weighting to signals near the baseline (noise)
    and small or zero weighting to signals much larger than the baseline 
    (actual signal). The iterative procedure runs until the changes in weights
    falls below a given threshold or the maximum number of iterations is reached. 

    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z. Generally best when
            between 1e4 and 1e8 (tested on simulated Raman spectra, spectra
            with broader spectral bands should use larger lambda values).
        ratio:
            weighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    """

    n_elements: int = len(array)

    # Second-order difference matrix.
    diff2nd = sparse.eye(n_elements, format='csc')
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd[1:] - diff2nd[:-1]

    # Symmetric pentadiagonal matrix.
    balanced_pentadiagonal = lam * diff2nd.T * diff2nd
    weights = np.ones(n_elements)

    # Perform itermax number of iterations at most.
    for i in range(itermax):
        weights_diag = sparse.diags(weights, 0, shape=(n_elements, n_elements))

        # Symmetric band-diagonal matrix that allows for more efficient algorithms.
        sym_band_diag = sparse.csc_matrix(weights_diag + balanced_pentadiagonal)

        # Cholesky decomposition.
        chol = sparse.csc_matrix(cholesky(sym_band_diag.todense()))
        background = spsolve(chol, spsolve(chol.T, weights * array))

        # Find d- and the mean and standard deviation for weighting.
        diff = array - background
        diff_negative = diff[diff < 0]
        mean = np.mean(diff_negative)
        sigma = np.std(diff_negative)
        new_weights = 1. / (1 + np.exp(2 * (diff - (2 * sigma - mean)) / sigma))

        # Check exit condition.
        if np.linalg.norm(weights - new_weights) / np.linalg.norm(weights) < ratio:
            break

        # Set weights and loop again.
        weights = new_weights

    # Return fitted background vector.
    return background

def als(
        array: np.typing.ArrayLike, 
        lam: float | int=1e6, 
        ratio: float=0.1, 
        itermax: int=10
    ) -> np.typing.ArrayLike:

    """
    Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005)

    Baseline Correction with Asymmetric Least Squares Smoothing
    based on https://github.com/vicngtor/BaySpecPlots

    Baseline Correction with Asymmetric Least Squares Smoothing
    Paul H. C. Eilers and Hans F.M. Boelens
    October 21, 2005

    Description from the original documentation:

    Most baseline problems in instrumental methods are characterized by a smooth
    baseline and a superimposed signal that carries the analytical information: a series
    of peaks that are either all positive or all negative. We combine a smoother
    with asymmetric weighting of deviations from the (smooth) trend get an effective
    baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
    No prior information about peak shapes or baseline (polynomial) is needed
    by the method. The performance is illustrated by simulation and applications to
    real data.

    See documentation for arpls for more information. ALS is performed in a
    similar manner, but the weights are not iteratively optimized, so the
    parameters must be hand-tuned for each problem. ALS is generally faster
    than arPLS, but is not nearly as effective. arPLS is recommended in most
    cases, especially for spectra with a few thousand data points at most.

    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            weighting deviations. 0.5 = symmetric, <0.5: negative
            deviations are stronger suppressed
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    """

    n_elements: int = len(array)
    diff2nd = sparse.eye(n_elements, format='csc')
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd.T
    weights = np.ones(n_elements)

    for i in range(itermax):
        weights_diag = sparse.diags(weights, 0, shape=(n_elements, n_elements))
        sym_band_diag = weights_diag + lam * diff2nd.dot(diff2nd.T)
        background = spsolve(sym_band_diag, weights * array)
        weights = ratio * (array > background) + (1 - ratio) * (array < background)

    return background

