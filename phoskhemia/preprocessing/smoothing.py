import numpy as np
import scipy as sp
from numba import njit, prange

def fourier_gaussian_smooth(
        array: np.typing.ArrayLike | list, 
        sigma: float=40, 
        shape_parameter: float=1, 
        deriv: int=0
    ) -> np.typing.ArrayLike:

    """
    Smooths a one-dimensional array by convolution of the array and a Gaussian.
    As the Fourier transform of a Gaussian is another Gaussian, smoothing with
    a Gaussian filter is readily performed given the properties of convolution.
    Fourier transforms (FT) require a periodic array to avoid artifacts, so 
    concatenation of the flipped array with its mirror provides the periodicity. 
    A Super-Gaussian smoothing filter is chosen for use in Fourier space. 
    For the filter, smaller sigma values result in more smoothing. The shape 
    parameter is the exponent that a Gaussian function is raised to, with larger
    values taking the filter closer to a flat-top function (1 = Gaussian, 
    0.5 = Laplace).

    For derivatives, the Fourier coordinate is defined, which has the same 
    dimension as the data and is shifted to be symmetric about zero.
    An extra factor of 2pi is necessary due to the FT definition in python. 
    Calculation of FT and multiplying by (iq)^n (n = order of derivative)
    followed by taking inverse FT returns the smoothed derivative.

    Args:
        array (np.typing.ArrayLike | list): One-dimensional array of data.
        sigma (float, optional): Standard deviation of Gaussian in Fourier domain.
            Smaller values result in more smoothing. Defaults to 40.
        shape_parameter (float, optional): Shape of general Gaussian. 
            1 = Gaussian, 0.5 = Laplace. Defaults to 1.
        deriv (int, optional): Order of data derivative. Defaults to 0.

    Returns:
        np.typing.ArrayLike: Smoothed data/derivative array.
    """

    # Find an array length that is readily divided in the FFT procedure.
    shape = sp.fft.next_fast_len(2 * len(array))
    mirror_stack = np.hstack((array, np.flip(array)))
    
    # Gaussian window function that is translated to the center of the array.
    win = np.roll(
        sp.signal.windows.general_gaussian(
            shape, shape_parameter, sigma
            ), shape // 2
    )
    
    # Fourier transform of the mirrored data array.
    transform = np.fft.fft(mirror_stack, shape)

    if deriv > 0:
        # Defining the Fourier coordinate for any chosen derivative.
        four_coordinate = 2 * np.pi * np.arange(-shape // 2, shape // 2, 1) / shape

        # Obtaining the derivative in the Fourier domain and taking the inverse transform.
        transform_derivative = np.roll(
            np.roll(transform, -shape // 2) 
            * ((1j * four_coordinate) ** deriv), shape // 2
        )

        derivative = np.real(np.fft.ifft(transform_derivative * win))[:len(array)]

        return derivative

    else:
        # Inverse transform of the product of the data and Gaussian window.
        smooth_array = np.real(np.fft.ifft(transform * win))[:len(array)]

        return smooth_array

@njit(fastmath=True, parallel=True)
def average1D(
        array: np.typing.ArrayLike, 
        half_window: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array[half_window:-half_window])
    for i in prange(len(new_array)):
        new_array[i] = np.mean(array[i - half_window:i + half_window + 1])

    return new_array

@njit(fastmath=True, parallel=True)
def downsample1D(
        array: np.typing.ArrayLike, 
        half_window: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array[::(2 * half_window)])
    new_array[0] = np.mean(array[:half_window + 1])
    new_array[-1] = np.mean(array[-half_window - 1:])

    for i in prange(1, len(array[:-half_window:(2 * half_window)]) - 1):
        index = 2 * half_window * i
        new_array[i] = np.mean(
            array[index - half_window:index + half_window + 1]
        )

    return new_array

@njit(fastmath=True, parallel=True)
def average2D(
        array: np.typing.ArrayLike, 
        m_rows: int=1, 
        n_cols: int=1
    ) -> np.typing.ArrayLike:

    """
    Smooths a two-dimensional array with the use of a sliding window. The 
    sliding window is constructed to be 2 * m_rows + 1 rows by 2 * n_cols + 1 columns.

    Args:
        array (np.typing.ArrayLike): Two-dimensional array of data to be smoothed.
        m_rows (int, optional): Half-window interval along rows. Defaults to 1.
        n_cols (int, optional): Half-window interval along columns. Defaults to 1.

    Returns:
        np.typing.ArrayLike: Smoothed array.
    """

    new_array = np.zeros_like(array)
    for i in prange(len(new_array[:, 0])):
        for j in prange(len(new_array[0, :])):
            new_array[i, j] = np.mean(
                array[i - m_rows:i + m_rows + 1, j - n_cols:j + n_cols + 1]
            )

    return new_array[m_rows:-m_rows, n_cols:-n_cols]

@njit(fastmath=True, parallel=True)
def average2D_rows(
        array: np.typing.ArrayLike, 
        m_rows: int=1
    ) -> np.typing.ArrayLike:
    
    new_array = np.zeros_like(array)
    for i in prange(len(new_array[:, 0])):
        for j in prange(len(new_array[0, :])):
            new_array[i, j] = np.mean(array[i - m_rows:i + m_rows + 1, j])

    return new_array[m_rows:-m_rows, :]

@njit(fastmath=True, parallel=True)
def average2D_cols(
        array: np.typing.ArrayLike, 
        n_cols: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array)
    for i in prange(len(new_array[:, 0])):
        for j in prange(len(new_array[0, :])):
            new_array[i, j] = np.mean(array[i, j - n_cols:j + n_cols + 1])

    return new_array[:, n_cols:-n_cols]

@njit(fastmath=True, parallel=True)
def downsample2D(
        array: np.typing.ArrayLike, 
        m_rows: int=1, 
        n_cols: int=1
    ) -> np.typing.ArrayLike:

    """
    Downsamples a two-dimensional array by averaging the 2 * p + 1 elements along
    the given rows/columns, where p is the half interval of m_rows or n_cols.

    Args:
        array (np.typing.ArrayLike): Two-dimensional array to be downsampled.
        m_rows (int, optional): Half-window interval along rows. Defaults to 1.
        n_cols (int, optional): Half-window interval along columns. Defaults to 1.

    Returns:
        np.typing.ArrayLike: Downsampled array.
    """

    new_array = np.zeros_like(array[::(2 * m_rows), ::(2 * n_cols)])

    # Getting the corners of the new array. 
    # m_rows + 1 by n_cols + 1 are averaged for these.
    new_array[0, 0] = np.mean(array[:m_rows + 1, :n_cols + 1])
    new_array[0, -1] = np.mean(array[:m_rows + 1, -n_cols - 1:])
    new_array[-1, 0] = np.mean(array[-m_rows - 1:, :n_cols + 1])
    new_array[-1, -1] = np.mean(array[-m_rows - 1:, -n_cols - 1:])

    for i in prange(1, len(array[:-m_rows:(2 * m_rows), 0]) - 1):
        # Get values for first and last columns of ith row. 
        new_array[i, 0] = np.mean(
            array[(2 * m_rows * i) - m_rows:(2 * m_rows * i) + m_rows + 1, 
                :n_cols + 1]
        )

        new_array[i, -1] = np.mean(
            array[(2 * m_rows * i) - m_rows:(2 * m_rows * i) + m_rows + 1, 
                -n_cols - 1:]
        )

        for j in prange(1, len(array[0, :-n_cols:(2 * n_cols)]) - 1):
            # Get values for first and last rows of jth column.
            new_array[0, j] = np.mean(
                array[:m_rows + 1, 
                    (2 * n_cols * j) - n_cols:(2 * n_cols * j) + n_cols + 1]
            )

            new_array[-1, j] = np.mean(
                array[-m_rows - 1:, 
                    (2 * n_cols * j) - n_cols:(2 * n_cols * j) + n_cols + 1]
            )

            # Establishing indices for cleaner indexing.
            row_index = 2 * m_rows * i
            col_index = 2 * n_cols * j

            # Binning of 2 * m_rows + 1 rows and 2 * n_cols + 1 columns for 
            # ith row and jth column in new array.
            new_array[i, j] = np.mean(
                array[row_index - m_rows:row_index + m_rows + 1, 
                    col_index - n_cols:col_index + n_cols + 1]
            )

    return new_array

@njit(fastmath=True, parallel=True)
def downsample2D_rows(
        array: np.typing.ArrayLike, 
        m_rows: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array[::(2 * m_rows), :])
    for i in prange(1, len(array[:-m_rows:(2 * m_rows), 0]) - 1):
        for j in prange(0, len(array[0, :])):
            new_array[0, j] = np.mean(array[:m_rows + 1, j])
            new_array[-1, j] = np.mean(array[-m_rows - 1:, j])
            row_index = 2 * m_rows * i

            new_array[i, j] = np.mean(
                array[row_index - m_rows:row_index + m_rows + 1, j]
            )

    return new_array

@njit(fastmath=True, parallel=True)
def downsample2D_cols(
        array: np.typing.ArrayLike, 
        n_cols: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array[:, ::(2 * n_cols)])
    for i in prange(0, len(array[:, 0])):
        new_array[i, 0] = np.mean(array[i, :n_cols + 1])
        new_array[i, -1] = np.mean(array[i, -n_cols - 1:])
        for j in prange(1, len(array[0, :-n_cols:(2 * n_cols)]) - 1):
            col_index = 2 * n_cols * j

            new_array[i, j] = np.mean(
                array[i, col_index - n_cols:col_index + n_cols + 1]
            )

    return new_array

@njit(fastmath=True, parallel=True)
def downsample2D_mixed(
        array: np.typing.ArrayLike, 
        m_rows: int=1, 
        n_cols: int=1
    ) -> np.typing.ArrayLike:

    new_array = np.zeros_like(array[::(2 * m_rows), n_cols:-n_cols])
    for i in prange(1, len(array[:-m_rows:(2 * m_rows), 0]) - 1):
        for j in prange(n_cols, len(array[0, :]) - n_cols):
            new_array[0, j] = np.mean(array[:m_rows + 1, j - n_cols:j + n_cols + 1])
            new_array[-1, j] = np.mean(array[-m_rows - 1:, j - n_cols:j + n_cols + 1])
            row_index = 2 * m_rows * i

            new_array[i, j] = np.mean(
                array[row_index - m_rows:row_index + m_rows + 1, 
                    j - n_cols:j + n_cols + 1]
            )

    return new_array

def average_filter(
        array: np.typing.ArrayLike, 
        quarter_window: tuple[int, int]=(1, 1), 
        downsampling: str='both'
    ) -> np.typing.ArrayLike:

    """
    Composite function consisting of functions defined for smoothing 
    two-dimensional arrays with a sliding window function and 
    whether to downsample the smoothed array. Allows the user to control the
    smoothing (and downsampling) of each axis individually. Downsampling automatically
    includes some smoothing (due to binning of values), but downsampling and smoothing
    can be performed separately for individual axes.

    Args:
        array (np.typing.ArrayLike): Two-dimensional array to be smoothed/downsampled.
        quarter_window (tuple[int, int], optional): Two-tuple of integers for the 
            half-window intervals used in the smoothing/downsampling for 
            rows and columns, respectively. Defaults to (1, 1).
        downsampling (str, optional): Case insensitive string describing if 
            downsampling should be performed on certain axes. 
            Options are 'both', 'rows', 'columns', and 'neither'. Defaults to 'both'.

    Raises:
        TypeError: Array with other than two dimensions passed.
        Certain combinations of arguments raise a ValueError:
            ValueError: quarter_window=(0, 0)
            ValueError: downsampling='rows' and quarter_window=(0, n)
            ValueError: downsampling='columns' and quarter_window=(m, 0)

    Returns:
        np.typing.ArrayLike: Processed array.
    """

    if np.shape(array) != 2:
        raise TypeError('Array must be two-dimensional.')

    if quarter_window[0] == 0 and quarter_window[1] == 0:
        raise ValueError(
            'Setting both half-window sizes to zero does nothing.'
        )

    match downsampling.casefold:
        # Smooth and downsample both axes.
        case 'both':
            new_array = downsample2D(
                array, m_rows=quarter_window[0], n_cols=quarter_window[1]
            )

        # Downsample only rows, with columns having the option to be 
        # smoothed but not downsampled.
        case 'rows' | 'row':
            if quarter_window[1] == 0: 
                new_array = downsample2D_rows(
                    array, m_rows=quarter_window[0]
                )

            elif quarter_window[0] == 0:
                raise ValueError(
                    'Row downsampling with half-window size of 0 is not supported.'
                )

            else:
                new_array = downsample2D_mixed(
                    array, m_rows=quarter_window[0], n_cols=quarter_window[1]
                )

        # Downsample only columns, with rows having the option to be 
        # smoothed but not downsampled.
        case 'columns' | 'cols' | 'column' | 'col':
            if quarter_window[0] == 0:
                new_array = downsample2D_cols(
                    array, n_cols=quarter_window[1]
                )

            elif quarter_window[1] == 0:
                raise ValueError(
                    'Column downsampling with half-window size of 0 is not supported.'
                )

            else:
                new_array = downsample2D_mixed(
                    array.T, m_rows=quarter_window[1], n_cols=quarter_window[0]
                ).T

        # Smooth one or both axes, but do not downsample.
        case 'neither' | 'none':
            if quarter_window[0] > 0 and quarter_window[1]:
                new_array = average2D(
                    array, m_rows=quarter_window[0], n_cols=quarter_window[1]
                )

            elif quarter_window[0] > 0 and quarter_window[1] == 0:
                new_array = average2D_rows(
                    array, m_rows=quarter_window[0]
                )

            elif quarter_window[0] == 0 and quarter_window[1] > 0:
                new_array = average2D_cols(
                    array, ncols=quarter_window[1]
                )

        case _:
            pass

    return new_array

#TODO: Update this function. Currently too broad of scope and messy.
def svd_smooth(
        array: np.typing.ArrayLike,
        components: int=1, 
        relative_tolerance: float=1.e-2, 
        absolute_tolerance: float=1.e-2, 
        optimize_reconstruction: bool=True, 
        relative_magnitude: float | None=None
    ) -> np.typing.ArrayLike:
        # absolute(a - b) <= (atol + rtol * absolute(b))

    U, svd_vals, Vh = sp.linalg.svd(array, lapack_driver='gesdd', full_matrices=False)
    reconstruction = U[:, :components] @ np.diag(svd_vals[:components]) @ Vh[:components, :]

    if optimize_reconstruction:
        reconstruction_converged = np.allclose(array, reconstruction, atol=absolute_tolerance, rtol=relative_tolerance)
        while not reconstruction_converged:
            components = components + 1
            reconstruction = U[:, :components] @ np.diag(svd_vals[:components]) @ Vh[:components, :]
            reconstruction_converged = np.allclose(array, reconstruction, atol=absolute_tolerance, rtol=relative_tolerance)

        if components != len(svd_vals):
            print(f'{f'Reconstruction by SVD converged at {components} components' : ^55}')

        else: 
            print(f'Reconstruction by SVD failed to converge (All {components} components used). Adjustment of tolerances is recommended.')
            continuation = input("Do you want to continue (Y/N)? ")

            match continuation.casefold:
                case "n" | "no" | "stop":
                    raise RuntimeError

                case "y" | "yes" | "continue":
                    pass

                case _:
                    pass

    elif not optimize_reconstruction and relative_magnitude:
        norm_svd = svd_vals / np.max(svd_vals)
        components = len(svd_vals[norm_svd > relative_magnitude])
        reconstruction = U[:, :components] @ np.diag(svd_vals[:components]) @ Vh[:components, :]
        relative_magnitude_reconstruction = relative_magnitude
        print(f'{f'SVD Reconstruction performed with {components} / {len(svd_vals)} components' : ^55}')

    return reconstruction


#from rich import print
#test_array = np.ones((24, 24)) * 8
#for i in range(2, len(test_array[0, :]) // 2, 2):
#    test_array[i:-i, i:-i] -= 1
#print(test_array)
#array = downsample2D(test_array, m_rows=2, n_cols=2)
#print(np.shape(array))
#[print(array[i, :]) for i in range(len(array[:, 0]))]
