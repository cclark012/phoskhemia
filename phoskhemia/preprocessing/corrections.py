from __future__ import annotations
from typing import Any, Literal
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from numpy.typing import NDArray
from phoskhemia.utils.typing import ArrayFloatAny
from phoskhemia.data.spectrum_handlers import TransientAbsorption

def arpls(
        array: ArrayFloatAny, 
        lam: float | int=1e4, 
        ratio: float=0.05, 
        itermax: int=100
    ) -> ArrayFloatAny:
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
    diff2nd: ArrayFloatAny = sparse.eye(n_elements, format='csc')
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd[1:] - diff2nd[:-1]

    # Symmetric pentadiagonal matrix.
    balanced_pentadiagonal: ArrayFloatAny = lam * diff2nd.T * diff2nd
    weights: ArrayFloatAny = np.ones(n_elements)

    # Perform itermax number of iterations at most.
    for i in range(itermax):
        weights_diag: ArrayFloatAny = (
            sparse.diags(weights, 0, shape=(n_elements, n_elements))
        )

        # Symmetric band-diagonal matrix that allows for more efficient algorithms.
        sym_band_diag: ArrayFloatAny = (
            sparse.csc_matrix(weights_diag + balanced_pentadiagonal)
        )

        # Cholesky decomposition.
        chol: ArrayFloatAny = (
            sparse.csc_matrix(cholesky(sym_band_diag.todense()))
        )

        background: ArrayFloatAny = (
            spsolve(chol, spsolve(chol.T, weights * array))
        )

        # Find d- and the mean and standard deviation for weighting.
        diff: ArrayFloatAny = array - background
        diff_negative: ArrayFloatAny = diff[diff < 0]
        mean: float = np.mean(diff_negative)
        sigma: float = np.std(diff_negative)
        new_weights: ArrayFloatAny = (
            1. / (1 + np.exp(2 * (diff - (2 * sigma - mean)) / sigma))
        )

        # Check exit condition.
        if np.linalg.norm(weights - new_weights) / np.linalg.norm(weights) < ratio:
            break

        # Set weights and loop again.
        weights = new_weights

    # Return fitted background vector.
    return background

def als(
        array: ArrayFloatAny, 
        lam: float | int=1e6, 
        ratio: float=0.1, 
        itermax: int=10
    ) -> ArrayFloatAny:
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
    diff2nd: ArrayFloatAny = sparse.eye(n_elements, format='csc')
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd[1:] - diff2nd[:-1]
    diff2nd = diff2nd.T
    weights: ArrayFloatAny = np.ones(n_elements)

    for i in range(itermax):
        weights_diag: ArrayFloatAny = (
            sparse.diags(weights, 0, shape=(n_elements, n_elements))
        )
        sym_band_diag: ArrayFloatAny = (
            weights_diag + lam * diff2nd.dot(diff2nd.T)
        )
        background: ArrayFloatAny = spsolve(sym_band_diag, weights * array)
        weights: ArrayFloatAny = (
            ratio * (array > background) + (1 - ratio) * (array < background)
        )

    return background

def find_time_zero(
        original_times: ArrayFloatAny,
        original_data: ArrayFloatAny,
        printing: int=0,
    ) -> tuple[ArrayFloatAny, ArrayFloatAny]:
    """
    Finds the time zero of a transient absorption spectrum 
    by finding the inflection point of the rising signal.
    The inflection point is found for each wavelength, then
    the median value is used to shift the data array.

    Args:
        original_times (np.typing.ArrayLike): The time axis with negative times.
        original_data (np.typing.ArrayLike): The data array with negative time values. 
        printing (int): How much information should be printed. 0 disables printing, 
            1 prints the final result, and 2 prints all intermediate results. 
            Defaults to 0.

    Returns:
        tuple[np.typing.ArrayLike, np.typing.ArrayLike]: The truncated/extended 
            time axis and shifted data array.
    """

    # Constants to choose how wide of a window to use for general smoothing and
    # final time zero estimation. Uses that number of indices on either 
    # side (positive and negative) of the originally notated time zero.
    SMOOTH_RANGE: int = 50
    TRUNC_RANGE: int = 20

    time_zero: int = np.argmin(np.abs(original_times))
    time_range: ArrayFloatAny = original_times[
        time_zero - SMOOTH_RANGE:time_zero + SMOOTH_RANGE
    ]

    truncated_window: ArrayFloatAny = time_range[
        np.logical_and(time_range >= -TRUNC_RANGE, time_range <= TRUNC_RANGE)
    ]
    new_time_zeros: list[float] = []

    # Determine inflection point for each wavelength.
    for i in range(0, len(original_data[0, :])):
        original_series: ArrayFloatAny = original_data[
            time_zero - SMOOTH_RANGE:time_zero + SMOOTH_RANGE, i
        ]

        # Make smoothing spline for first and second derivatives.
        series = sp.interpolate.make_smoothing_spline(time_range, original_series)

        # First derivative to find inflection point (relative maxima).
        first_derivative: ArrayFloatAny = series.derivative(nu=1)(time_range)
        first_inflection: int = np.argmax(np.abs(
            first_derivative[np.logical_and(
                time_range >= -TRUNC_RANGE, time_range <= TRUNC_RANGE
            )]
        ))
        first_new_time_zero: float = truncated_window[first_inflection]

        # Second derivative for inflection point (close to zero).
        second_derivative: ArrayFloatAny = series.derivative(nu=2)(time_range)
        second_inflection: int = np.argmin(np.abs(
            second_derivative[np.logical_and(
                time_range >= -TRUNC_RANGE, time_range <= TRUNC_RANGE
            )]
        ))
        second_new_time_zero: float = truncated_window[second_inflection]

        # Use the smaller index of the obtained inflection points.
        closer_to_zero: int = (first_inflection, second_inflection)[
            np.argmin([np.abs(first_new_time_zero), np.abs(second_new_time_zero)])
        ]

        new_time_zeros += [truncated_window[closer_to_zero]]
        print(truncated_window[closer_to_zero]) if printing == 2 else None

    # Use the median value of the obtained time zeros to shift the array.
    time_zero_ns: float = np.median(new_time_zeros)
    delta_t: float = np.unique(original_times[1:] - original_times[:-1])[0]

    # Shift array values, but leave time axis where it is.
    shift_value: int = int(time_zero_ns // delta_t)
    data: ArrayFloatAny = original_data[time_zero + shift_value:, :]

    # Truncate or extend the time axis.
    if shift_value >= 0:
        times: ArrayFloatAny = original_times[time_zero:len(original_data[:, 0])]

    else:
        times: ArrayFloatAny = np.arange(0, len(data[:, 0]) * delta_t, delta_t)

    # Tell user how the data was adjusted if printing enabled.
    if printing == 1 or printing == 2:
        print(f"Median = {time_zero_ns} ns. ", end='')
        if shift_value > 0:
            print(f"Data shifted backward by {shift_value * delta_t} ns.")

        elif shift_value < 0:
            print(f"Data shifted forward by {np.abs(shift_value) * delta_t} ns.")

        elif shift_value == 0:
            print(f"Data was not shifted.")

        else:
            print(f"Something went wrong. Check the data output.")

    return times, data

def estimate_irf(array: ArrayFloatAny) -> ArrayFloatAny:
    # Get an optimal size for the Fourier transform, 
    # then make a periodic signal from the input array for FFT.
    shape: int = sp.fft.next_fast_len(2 * len(array))
    arr_stack: ArrayFloatAny = np.hstack(array, np.flip(array))
    farray: ArrayFloatAny = sp.fft.fft(arr_stack, shape)

    # Define Fourier coordinate for the derivative.
    four_coord: ArrayFloatAny = (
        2 * np.pi * np.arange(-shape // 2, shape // 2, 1) / shape
    )

    # Make a Gaussian window for smoothing the derivative.
    window: ArrayFloatAny = (
        np.roll(sp.signal.windows.general_gaussian(
            shape, p=1, sig=len(array) // 4
        ), shift=shape // 2)
    )

    # Calculate Fourier transform of derivative and convolve with Gaussian.
    dfarray: ArrayFloatAny = (
        np.roll(np.roll(
            farray, shift=-shape // 2
        ) * (1j * four_coord) ** 1, shift=shape // 2)
    )
    derivative: ArrayFloatAny = (
        np.real(sp.fft.ifft(dfarray * window))[:len(array)]
    )

    # Approximately where IRF stops having significant contribution.
    cutting_point: int = np.argmin(derivative)

    # Deconvolve spectrum and IRF.
    irf: ArrayFloatAny
    remainder: ArrayFloatAny
    irf, remainder = sp.signal.deconvolve(array, array[cutting_point:])

    return irf


def find_t0_index(times: NDArray[np.floating]) -> int:
    times: NDArray[np.floating] = np.asarray(times, dtype=float).reshape(-1)
    return int(np.argmin(np.abs(times)))

def apply_time_zero(
        arr: TransientAbsorption, 
        *, 
        mode: Literal['truncate', 'truncate_shift', 'shift'] = 'truncate_shift',
        t0: float | None = None,
        use_pre_t0_for_noise: bool = True,
        noise_method: Literal['std', 'mad'] = 'std'
    ) -> tuple[TransientAbsorption, dict[str, Any]]:
    """_summary_

    Parameters
    ----------
    arr : TransientAbsorption
        _description_
    mode : Literal['truncate', 'truncate_shift', 'shift'], optional
        _description_, by default 'truncate_shift'
    t0 : float | None, optional
        _description_, by default None
    use_pre_t0_for_noise : bool, optional
        _description_, by default True
    noise_method : Literal['std', 'mad'], optional
        _description_, by default 'std'

    Returns
    -------
    tuple[TransientAbsorption, dict[str, Any]]
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    data: NDArray[np.floating] = np.asarray(arr)
    t: NDArray[np.floating] = np.asarray(arr.y, dtype=float)
    if t0 is None:
        i0: int = find_t0_index(t)
        t0: float = arr.y[i0]
    else:
        t0: float = float(t0)
        i0: int = int(np.argmin(np.abs(t - t0)))

    info: dict[str, Any] = {"t0": t0, "index_t0": i0, "mode_t0": mode}
    
    if use_pre_t0_for_noise and i0 >= 2:
        pre = data[:i0, :]
        if noise_method == "std":
            noise: NDArray[np.floating] = np.std(pre, axis=0, ddof=1)
        elif noise_method == "mad":
            med: NDArray[np.floating] = np.median(pre, axis=0)
            mad: NDArray[np.floating] = np.median(np.abs(pre - med), axis=0)
            # 1.4826 is the conversion to st. dev. assuming gaussian noise.
            noise: NDArray[np.floating] = 1.4826 * mad
        else:
            raise ValueError("noise_method must be 'std' or 'mad'")
        info["noise_t0"] = noise
    
    if mode == "shift":
        y_new: NDArray[np.floating] = t - t0
        meta: dict[str, Any] = getattr(arr, "meta", {}).copy()
        meta.update(info)
        if "noise_t0" in info:
            meta["noise_t0"] = info["noise_t0"]
        return TransientAbsorption(data, x=arr.x, y=y_new, meta=meta), info

    elif mode == "truncate":
        data_new: NDArray[np.floating] = data[i0:, :]
        y_new: NDArray[np.floating] = t[i0:]
        meta: dict[str, Any] = getattr(arr, "meta", {}).copy()
        meta.update(info)
        if "noise_t0" in info:
            meta["noise_t0"] = info["noise_t0"]
        return TransientAbsorption(data_new, x=arr.x, y=y_new, meta=meta), info

    elif mode == "truncate_shift":
        data_new: NDArray[np.floating] = data[i0:, :]
        y_new: NDArray[np.floating] = t[i0:] - t0
        meta: dict[str, Any] = getattr(arr, "meta", {}).copy()
        meta.update(info)
        if "noise_t0" in info:
            meta["noise_t0"] = info["noise_t0"]
        return TransientAbsorption(data_new, x=arr.x, y=y_new, meta=meta), info

    else:
        raise ValueError("mode must be 'truncate', 'shift', or 'truncate_shift'")

