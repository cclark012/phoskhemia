from __future__ import annotations
import numpy as np
import copy
from scipy.signal import convolve
from numpy.typing import NDArray
from typing import Callable, Any

class TransientAbsorption(np.ndarray):
    """
    Represents a 2-D transient absorption dataset.

    Shape : (n_times, n_wavelengths)

    TODO:
    1) Add downsampling for large datasets

    Attributes
    ----------
    x : ndarray
        Wavelength axis (length = n_wavelengths).
    y : ndarray
        Time axis (length = n_times).

    Methods
    -------
    smooth(window, normalize=True, separable_tol=1e-10, **kwargs)
        Smooths the data with a provided window or window size.
    combine_with(other, mode='average', fill_value=0.0)
        Combines two TransientAbsorption arrays into one dataset.
    add(other, mode='average', fill_value=0.0)
        Equivalent to combine_with().
    fit_global_kinetics(*args, **kwargs)
        Fit the dataset to a kinetic model.
    """

    __array_priority__: float = 1000.0
    _DEBUG: bool = False
    _COMBINE_MODES: list[str]= ["average", "mean"]
    x: NDArray[np.floating]
    """ 1D array of wavelength values.
    """
    y: NDArray[np.floating]
    """ 1D array of time values.
    """

    _SUPPORTED_ARRAY_FUNCTIONS: set = {
        np.all,
        np.any,
        np.atleast_1d,
        np.atleast_2d,
        np.atleast_3d,
        np.argmax,
        np.argmin,
        np.argpartition,
        np.argsort,
        np.astype,
        np.choose,
        np.clip,
        np.compress,
        np.conj,
        np.conjugate,
        np.copy,
        np.cumprod,
        np.cumsum,
        np.diagonal,
        np.dot,
        np.hstack,
        np.max,
        np.mean,
        np.min,
        np.nanargmax,
        np.nanargmin,
        np.nanmax,
        np.nanmean,
        np.nanmedian,
        np.nanmin,
        np.nanstd,
        np.nanvar,
        np.nonzero,
        np.partition,
        np.prod,
        np.put,
        np.ravel,
        np.repeat,
        np.reshape,
        np.resize,
        np.round,
        np.searchsorted,
        np.sort,
        np.squeeze,
        np.std,
        np.sum,
        np.swapaxes,
        np.take,
        np.trace,
        np.transpose,
        np.var,
        np.vstack,
    }

    def __new__(
            cls, 
            input_array: NDArray[np.floating], 
            x: NDArray[np.floating] | None=None, 
            y: NDArray[np.floating] | None=None, 
            dtype: type=float
        ) -> TransientAbsorption:
        """
        Constructs the TransientAbsorption representation.

        Parameters
        ----------
        input_array : NDArray[np.floating]
            2D array of data values of size (y, x)/(n_times, n_wavelengths)
        x : NDArray[np.floating] | None, optional
            1D array of wavelength values, by default None
        y : NDArray[np.floating] | None, optional
            1D array of time values, by default None
        dtype : type, optional
            numpy datatype to be used for the dataset, by default float.

        Returns
        -------
        TransientAbsorption
            Array of given data values with x (wavelength) and y (time) attributes.

        Raises
        ------
        ValueError
            Raised if array data is not or cannot be interpreted as 2D.
        ValueError
            Raised if either x or y data are not 1D.
        ValueError
            Raised if number of x values does not match the number of data columns.
        ValueError
            Raised if number of y values does not match the number of data rows.
        """

        arr: NDArray[np.floating] = np.asarray(input_array, dtype=dtype)

        # If 1-D, decide whether user meant a row (1 x N) or a column (N x 1).
        if arr.ndim == 1:
            n: int = arr.size
            # If x provided and its length matches n -> treat as row (1 x n)
            if x is not None and len(np.asarray(x)) == n:
                arr: NDArray[np.floating] = arr.reshape(1, n)
            # Else if y provided and its length matches n -> treat as column (n x 1)
            elif y is not None and len(np.asarray(y)) == n:
                arr: NDArray[np.floating] = arr.reshape(n, 1)
            else:
                # default: treat as a single row 1 x n
                arr: NDArray[np.floating] = arr.reshape(1, -1)

        # Now array must be 2-D
        if arr.ndim != 2:
            raise ValueError("TransientAbsorption requires a 2-D array (or 1-D which will become 1xN or Nx1).")

        obj: TransientAbsorption = arr.view(cls)
        nrows: int
        ncols: int
        nrows, ncols = obj.shape

        # Create default coords when missing
        if x is None:
            x: NDArray[np.floating] = np.arange(ncols, dtype=float)
        if y is None:
            y: NDArray[np.floating] = np.arange(nrows, dtype=float)

        x: NDArray[np.floating] = np.asarray(x, dtype=float)
        y: NDArray[np.floating] = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-D arrays")
        if x.shape[0] != ncols:
            raise ValueError(f"length of x ({x.shape[0]}) must equal number of columns ({ncols})")
        if y.shape[0] != nrows:
            raise ValueError(f"length of y ({y.shape[0]}) must equal number of rows ({nrows})")

        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        obj._validate()
        return obj

    def _validate(self) -> None:
        """Check that data is represented correctly."""

        if self.ndim != 2:
            raise ValueError("TransientAbsorption must be 2D")
        if self.shape != (len(self.y), len(self.x)):
            raise ValueError("Data/axis mismatch")

    def __array_finalize__(
            self, 
            obj: TransientAbsorption | None
            ) -> None:

        if obj is None:
            return
        object.__setattr__(self, "x", copy.copy(getattr(obj, "x", None)))
        object.__setattr__(self, "y", copy.copy(getattr(obj, "y", None)))

    def smooth(
        self,
        window: int | NDArray[np.floating] | tuple[int, int],
        *,
        normalize: bool=True,
        separable_tol: float=1e-10,
        **kwargs,
    ) -> TransientAbsorption:
        """
        Smooth data using scipy.signal.convolve.

        The data is smoothed by convolving the dataset with a boxcar window of
        size (ny, nx) or with a user-supplied window function. If the window is
        2D, then a separation of the window is attempted to perform two 1D smoothing
        operations. This is done through singular value decomposition, W = UΣV⁺, 
        where the columns of U are an orthonormal basis in the "time" direction 
        while the rows of V⁺ are an orthonormal basis in the "wavelength" direction.
        The window is deemed separable if the major components of the decomposition
        vastly outweigh any other components. This is chosen through the singular
        values, where the largest singular value denotes the 1st component, the
        2nd largest denotes the 2nd component, and so on. If the ratio of the 
        2nd component, Σ₂, to the 1st component, Σ₁, is below some threshold value 
        (the separable_tol argument, εₜₒₗ), Σ₂ / Σ₁ < εₜₒₗ, then the window is 
        separable and the smoothing is performed in two 1D passes.
        
        Parameters
        ----------
        window : int | array-like | tuple
            Parameters for the window function. If an integer or tuple of integers
            then a simple moving average is performed (boxcar window). If it is 
            and array of values then it is interpreted as a provided window function.
            - int or 1-D array: smooth along x.
            - (1, nx): smooth along x.
            - (ny, 1): smooth along y.
            - (ny, nx): 2-D smoothing.
        normalize : bool
            Normalize the window so it sums to 1, by default True.
        separable_tol : float
            Relative tolerance for separable-kernel detection, by default 1e-10.
        **kwargs
            Passed directly to scipy.signal.convolve
            (e.g. mode='same', method='auto').

        Notes
        -----
        Axis conventions:
            axis 0 → time (y)
            axis 1 → wavelength (x)
        """

        kwargs: dict = {"mode": "same", "method": "auto"} | kwargs

        data: NDArray[np.floating] = np.asarray(self, dtype=float)

        # Build kernel
        if isinstance(window, int):
            if window <= 0:
                raise ValueError("window length must be positive")
            kernel: NDArray[np.floating] = np.ones((1, window), dtype=float)

        elif isinstance(window, tuple | list):
            if len(window) != 2:
                raise ValueError("window tuple must be (ny, nx)")
            kernel: NDArray[np.floating] = np.ones(window, dtype=float)

        else:
            kernel: NDArray[np.floating] = np.asarray(window, dtype=float)
            if kernel.ndim == 1:
                kernel = kernel.reshape(1, -1)
            elif kernel.ndim != 2:
                raise ValueError("window must be int, 1-D, or 2-D")

        ny: int
        nx: int
        ny, nx = kernel.shape

        if normalize:
            s: float = kernel.sum()
            if s != 0:
                kernel = kernel / s

        # 1-D smoothing along x
        if ny == 1 and nx > 1:
            horizontal: NDArray[np.floating] = kernel.ravel()
            out: NDArray[np.floating] = convolve(data, horizontal[None, :], **kwargs)

        # 1-D smoothing along y
        elif ny > 1 and nx == 1:
            vertical: NDArray[np.floating] = kernel.ravel()
            out: NDArray[np.floating] = convolve(data, vertical[:, None], **kwargs)

        # 2-D smoothing
        else:
            # Attempt separable acceleration
            U: NDArray[np.floating]
            S: NDArray[np.floating]
            Vt: NDArray[np.floating]
            U, S, Vt = np.linalg.svd(kernel, full_matrices=False)
            separable: bool = S.size > 1 and S[1] / S[0] < separable_tol

            if separable:
                vertical: NDArray[np.floating] = U[:, 0] * np.sqrt(S[0])
                horizontal: NDArray[np.floating] = Vt[0, :] * np.sqrt(S[0])

                tmp: NDArray[np.floating] = convolve(data, vertical[:, None], **kwargs)
                out: NDArray[np.floating] = convolve(tmp, horizontal[None, :], **kwargs)

            else:
                out: NDArray[np.floating] = convolve(data, kernel, **kwargs)

        result: TransientAbsorption = out.view(TransientAbsorption)
        result.x = self.x
        result.y = self.y

        return result

    # TODO - Potentially add more mode and fill_val options, such as interpolation. 
    def combine(
            self, 
            arr2: TransientAbsorption,
            *,
            mode: str="average",
            fill_val: float | None=None,
        ) -> TransientAbsorption:
        """
        Combines two TransientAbsorption arrays into one instance.
        
        Returns the result of combining two TransientAbsorption arrays. 
        This first tries to combine along the second axis (akin to np.hstack 
        behavior) which switches to combining along the first axis (like 
        np.vstack) if that fails. The function checks if the y (time) values of 
        both instances match and, if they do, will create a new x (wavelength) 
        axis to hold all of the data and similar for combining when the x values 
        match. It should not matter which order the arrays are combined in 
        (i.e. arr1.combine(arr2) should be the same as arr2.combine(arr1)).
        
        The user can also specify whether any breaks between datasets should be 
        filled with fill_val or to have the arrays combined as-is with the 
        default (fill_val=None). If the user specifies a fill_val, then the gap 
        between two datasets is interpolated along the x or y axis by using the 
        average of the stepsize (i.e. if the first dataset has dx=1 and the 
        second has dx=2 then the gap will have a stepsize of dx=1.5). The array 
        values are then filled with fill_val. fill_val only applies if there
        is a gap larger than the stepsize between two datasets. Two datasets
        are also allowed to overlap, i.e. one with x values [400, 405, 410] and
        another with x values [410, 415, 420] are allowed to be combined but 
        they must have the same stepsize. Currently, the mode option only 
        accepts the "average" or "mean" options (which are the same). In the 
        future more modes may be added so as to allow more complex behavior.
        More options may also be added for fill_val but currently any number
        should work.
        
        The current implementation is also likely to break if the user tries to
        combine more than two datasets unless the step size of the first two
        datasets is the same as the third and that there is no gap between the
        first two datasets. In other words, trying to combine datasets with
        x values of [400, 405, 410, 415, 420], [430, 435, 440], and [445, 450]
        will likely fail. It may be possible to get around this by combining 
        the last two datasets first as they have no gap between them and all 
        three datasets have the same stepsize.
        

        Parameters
        ----------
        arr2 : TransientAbsorption
            The array to combine with.
        mode : str, optional
            How overlapping values are handled, by default "average".
            Currently accepts "average" and "mean" (which are the same).
        fill_val : float | None, optional
            How gaps in datasets are dealt with, by default None.

        Returns
        -------
        TransientAbsorption
            The combined dataset.

        Raises
        ------
        TypeError
            Raised if arr2 is not a TransientAbsorption instance.
        ValueError
            Raised if an unknown value is passed to mode.
        ValueError
            Raised if x and y differ in both arrays.
        ValueError
            Raised if axis spacing differs for two arrays when they overlap.
        ValueError
            A generic error occurred while trying to handle overlapping arrays.
        """

        ARRAY_TOLERANCE: int = 2

        if not isinstance(arr2, TransientAbsorption):
            raise TypeError("combine expects another TransientAbsorption instance")

        mode = mode.casefold()
        if mode not in self._COMBINE_MODES:
            raise ValueError(f"mode must be one of {self._COMBINE_MODES}")

        arr1: TransientAbsorption = self
        intersection: NDArray[np.floating]
        idx1: NDArray[np.integer]
        idx2: NDArray[np.integer]
        axis: int
        ax1: NDArray[np.floating]
        ax2: NDArray[np.floating]

        # TODO - May refactor this part to be easier to follow and so there is not so much repetition.
        # Check which axis we are combining along and adjust order of arrays if necessary.
        if np.array_equal(np.round(arr1.y, ARRAY_TOLERANCE), np.round(arr2.y, ARRAY_TOLERANCE)):
            # Two non-overlapping, positive arrays always have positive difference 
            # for min(arr2) - max(arr1) if in the order arr1 arr2. If the arrays 
            # are in the order arr2 arr1, then min(arr2) - max(arr1) is negative 
            # and the labeling should be swapped. Two overlapping, positive arrays 
            # always have negative difference for min(arr2) - max(arr1) regardless 
            # of order if it can be assumed that neither array is completely 
            # "covered" by the other. The other option of max(arr2) - min(arr1)
            # is also always positive for the two arrays. However, since neither of
            # the arrays are completely covered, then for the order arr1 arr2 the
            # inequalities max(arr2) > max(arr1) and min(arr2) > min(arr1). The
            # difference between max(arr2) - max(arr1) should then always be positive
            # for the order arr1 arr2.

            # Case of two non-overlapping arrays. 
            # Operation of min(arr2) - max(arr1) gives different sign depending on labeling.
            #        max  min
            #    arr1  ↓  ↓  arr2          min(arr2) - max(arr1) = +
            # ██████████  ██████████
            # ↑  arr2        arr1  ↑       min(arr2) - max(arr1) = -
            # min                max
            #
            # Case of two overlapping arrays. 
            # The operation min(arr2) - max(arr1) is always negative.
            #     min  max
            #  arr1 |  ↓ arr2              min(arr2) - max(arr1) = -
            # ██████↓███        
            #       ██████████
            # ↑ arr2    arr1 ↑             min(arr2) - max(arr1) = -
            # min          max
            # Find if arrays overlap at all

            intersection, idx1, idx2 = np.intersect1d(
                np.round(arr1.x, ARRAY_TOLERANCE), 
                np.round(arr2.x, ARRAY_TOLERANCE), 
                return_indices=True
            )
            diff: float = (
                (np.max(arr2.x) - np.max(arr1.x)) 
                if intersection.any() else (np.min(arr2.x) - np.max(arr1.x))
            )
            if diff < 0:
                arr1, arr2 = arr2, arr1
                idx1, idx2 = idx2, idx1

            axis, ax1, ax2 = 1, arr1.x, arr2.x

        elif np.array_equal(np.round(arr1.x, ARRAY_TOLERANCE), np.round(arr2.x, ARRAY_TOLERANCE)):
            intersection, idx1, idx2 = np.intersect1d(
                np.round(arr1.y, ARRAY_TOLERANCE), 
                np.round(arr2.y, ARRAY_TOLERANCE), 
                return_indices=True
            )
            diff: float = (
                (np.max(arr2.y) - np.max(arr1.y)) 
                if intersection.any() else (np.min(arr2.y) - np.max(arr1.y))
            )
            if diff < 0:
                arr1, arr2 = arr2, arr1
                idx1, idx2 = idx2, idx1

            axis, ax1, ax2 = 1, arr1.y, arr2.y

        else:
            raise ValueError(
                "Either x or y axes must match in both arrays; "
                "arrays with different numbers of rows and columns are not supported."
                )

        # Overlapping arrays
        if intersection.any():
            dax1: float = float(np.mean(np.diff(ax1)))
            dax2: float = float(np.mean(np.diff(ax2)))
            compatible: bool = np.isclose(dax1, dax2, rtol=1e-6, atol=1e-9)
            if not compatible:
                raise ValueError("Overlapping arrays with different axis spacings are not supported.")
            
            n_steps: int = int((np.max(ax2) - np.min(ax1)) / (0.5 * (dax1 + dax2)))
            new_ax: NDArray[np.floating] = np.linspace(np.min(ax1), np.max(ax2), n_steps + 1)

            new_data: NDArray[np.floating] = (
                np.zeros((len(arr1.y), len(new_ax))) 
                if axis == 1 else np.zeros((len(new_ax), len(arr1.x)))
            )
            # Just place old data into the new array since we will overwrite the overlap later.
            if axis == 1:
                new_data[:, :len(ax1)] = np.asarray(arr1)
                new_data[:, len(ax1)-len(idx1):] = np.asarray(arr2)

            elif axis == 0:
                new_data[:len(ax1), :] = np.asarray(arr1) 
                new_data[len(ax1)-len(idx1):, :] = np.asarray(arr2)

            for i1, i2 in zip(idx1, idx2):
                if ax1[i1] != ax2[i2]:
                    raise ValueError("An error occurred while handling overlapping values.")
                idx_new: int = np.argmin(np.abs(new_ax - ax1[i1]))

                if axis == 1:
                    col1: NDArray[np.floating] = np.atleast_2d(np.array(arr1[:, i1])).T
                    col2: NDArray[np.floating] = np.atleast_2d(np.array(arr2[:, i2])).T
                    stack: NDArray[np.floating] = np.hstack((col1, col2))
                    new_data[:, idx_new] = np.average(stack, axis=1)

                elif axis == 0:
                    row1: NDArray[np.floating] = np.atleast_2d(np.array(arr1[i1, :]))
                    row2: NDArray[np.floating] = np.atleast_2d(np.array(arr2[i2, :]))
                    stack: NDArray[np.floating] = np.vstack((row1, row2))
                    new_data[idx_new, :] = np.average(stack, axis=0)

            return (
                TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y)) 
                if axis == 1 else TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax)
            )

        # Non-overlapping arrays
        else:
            comb: NDArray[np.floating] = np.concatenate((ax1, ax2))
            diff: NDArray[np.floating] = np.diff(comb)
            unique: NDArray[np.floating]
            counts: NDArray[np.integer]
            unique, counts = np.unique(diff, return_counts=True)
            # Concatenate both axes regardless of spacing for fill_value = None and 
            # when gap size is equal to either of the array spacings.
            if fill_val == None or (len(unique) == 2 and not np.any(counts == 1)):
                new_ax: NDArray[np.floating] = comb
                new_data: NDArray[np.floating] = np.concatenate((arr1, arr2), axis=axis)
                return (
                    TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y)) 
                    if axis == 1 else TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax)
                )

            # Fill values between arrays with fill_value using the average stepsize of the two axes.
            else:
                dax: float = 0.5 * (np.mean(np.diff(ax1)) + np.mean(np.diff(ax2)))
                filled: NDArray[np.floating] = np.arange(np.max(ax1) + dax, np.min(ax2), dax)
                new_ax: NDArray[np.floating] = np.concatenate((ax1, filled, ax2))
                if axis == 1:
                    new_data: NDArray[np.floating] = np.zeros((len(arr1.y), len(new_ax))) + fill_val
                    new_data[:, :len(ax1)] = arr1
                    new_data[:, len(ax1)+len(filled):] = arr2
                    return TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y))

                elif axis == 0:
                    new_data: NDArray[np.floating] = np.zeros((len(new_ax), len(arr1.x))) + fill_val
                    new_data[:len(ax1), :] = arr1
                    new_data[len(ax1)+len(filled):, :] = arr2
                    return TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax)

    def add(
            self, 
            other: TransientAbsorption, 
            *,
            mode: str="average", 
            fill_val: float | None=None
        ) -> TransientAbsorption:
        """
        Alias for combine().

        Parameters
        ----------
        other : TransientAbsorption
            The array to combine with.
        mode : str, optional
            How overlapping values are handled, by default "average"
        fill_val : float | None, optional
            How gaps between datasets are handled, by default None

        Returns
        -------
        TransientAbsorption
            The combined dataset.

        Raises
        ------
        ValueError
            Raised if an unknown value is passed to mode.
        """

        if mode in self._COMBINE_MODES:
            return self.combine(other, fill_val=fill_val, mode=mode)

        raise ValueError(f"mode must be one of {self._COMBINE_MODES}")

    def __add__(
            self, 
            other: TransientAbsorption
        ) -> TransientAbsorption:

        # Non-TransientAbsorption (scalar or ndarray)
        if not isinstance(other, TransientAbsorption):
            try:
                res: NDArray[np.floating] = np.asarray(np.asarray(self) + other)
                out: TransientAbsorption = res.view(TransientAbsorption)
                object.__setattr__(out, "x", copy.copy(self.x))
                object.__setattr__(out, "y", copy.copy(self.y))
                return out

            except Exception:
                return NotImplemented

        # Both TransientAbsorption -> default average
        return self.combine(other, fill_val=None, mode="average")

    def __radd__(
            self, 
            other: TransientAbsorption
        ) -> TransientAbsorption:

        return self.__add__(other)

    def __getitem__(
            self, 
            key: tuple | int | slice | NDArray[np.integer] | NDArray[np.bool_]
        ) -> TransientAbsorption:
        """
        Indexing that slices data as well as x (wavelength) and y (time) values.

        Handles cases where self.x or self.y may be None (e.g. after reductions).
        Returns a TransientAbsorption view when possible; scalar results are
        wrapped as 0-D TransientAbsorption arrays.

        Parameters
        ----------
        key : tuple | int | slice | NDArray[np.integer] | NDArray[np.bool_]
            Index, indices, or slice to be taken of the data.

        Returns
        -------
        TransientAbsorption
            The chosen data point(s) with its x and y value(s).

        Raises
        ------
        IndexError
            Too many indices passed for indexing raises this error.
        """

        # Normalize key into (row_idx, col_idx)
        row_idx: int | slice | NDArray[np.integer] | NDArray[np.bool_]
        col_idx: int | slice | NDArray[np.integer] | NDArray[np.bool_]
        if isinstance(key, tuple):
            if len(key) == 2:
                row_idx, col_idx = key
            elif len(key) == 1:
                row_idx, col_idx = key[0], slice(None)
            else:
                raise IndexError("Too many indices for TransientAbsorption")
        else:
            row_idx, col_idx = key, slice(None)

        # Perform numeric indexing
        result: NDArray[np.floating] = super().__getitem__(key)

        # Convert scalars to 0-D array so subclass is preserved
        if not isinstance(result, np.ndarray):
            result: NDArray[np.floating] = np.array(result)

        out: TransientAbsorption = result.view(TransientAbsorption)

        # Helper to safely slice coordinate arrays (returns None if coord is None)
        def slice_coord(
                coord: NDArray[np.floating] | None, 
                idx: int | slice | NDArray[np.integer] | NDArray[np.bool_]
            ) -> NDArray[np.floating] | None:

            if coord is None or coord.ndim == 0:
                return None
            # Integer index -> scalar
            elif isinstance(idx, (int, np.integer)):
                return coord[idx]
            # None or full slice -> whole coord
            elif idx is None or (isinstance(idx, slice) and idx == slice(None)):
                return coord
            # fancy indexing / boolean mask / slice
            return np.asarray(coord[idx])

        # Initialize new_x/new_y before use
        new_x: NDArray[np.floating] | None = None
        new_y: NDArray[np.floating] | None = None

        ndim: int = out.ndim
        if self._DEBUG:
            if isinstance(key, (slice, int)):
                key_len = 1
            else:
                key_len = len(key)
            print("key:                  ", "len(key): ", "ndim: ", "row_idx: ", "col_idx: ", flush=True)
            print(f"{str(key) :<22}", f"{str(key_len) :<10}", f"{str(ndim) :<6}", f"{str(row_idx) :<9}", f"{str(col_idx) :<9}", flush=True)
            print(flush=True)

        if ndim == 2:
            new_x = slice_coord(self.x, col_idx)
            new_y = slice_coord(self.y, row_idx)

        elif ndim == 1:
            # One axis survived; determine which
            if isinstance(key, tuple):
                # Row_idx integer -> row collapsed -> x survives
                if isinstance(row_idx, (int, np.integer)):
                    new_x = slice_coord(self.x, col_idx)
                    new_y = slice_coord(self.y, row_idx)
                # Col_idx integer -> col collapsed -> y survives
                elif isinstance(col_idx, (int, np.integer)):
                    new_y = slice_coord(self.y, row_idx)
                    new_x = slice_coord(self.x, col_idx)
                else:
                    # e.g., arr[1:3, :] -> row slice yields 1D with x surviving
                    if (isinstance(col_idx, slice) and col_idx == slice(None)):
                        new_x = slice_coord(self.x, col_idx)
                        new_y = None
                    else:
                        # Fallback: prefer x if it exists, else y
                        new_x = slice_coord(self.x, col_idx) if self.x is not None else None
                        new_y = None
            else:
                # Single index -> row indexing like arr[k] -> x survives
                new_x = slice_coord(self.x, slice(None))
                new_y = slice_coord(self.y, row_idx)

        elif ndim == 0:
            new_x = slice_coord(self.x, col_idx)
            new_y = slice_coord(self.y, row_idx)

        else:
            # Higher dims unsupported; set coords to None
            new_x = None
            new_y = None

        # Attach coords (may be None)
        object.__setattr__(out, "x", new_x)
        object.__setattr__(out, "y", new_y)
        return out

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> Any:
        """_summary_

        Parameters
        ----------
        func : Callable[..., Any]
            _description_
        types : tuple[type, ...]
            _description_
        args : tuple[Any, ...]
            _description_
        kwargs : dict[str, Any] | None
            _description_

        Returns
        -------
        Any
            _description_

        Raises
        ------
        NotImplemented
        """

        if func not in TransientAbsorption._SUPPORTED_ARRAY_FUNCTIONS:
            return NotImplemented

        # Convert TransientAbsorption in args to ndarray and remember inputs
        numeric_args: list = []
        TransientAbsorption_inputs: list = []
        for a in args:
            if isinstance(a, TransientAbsorption):
                numeric_args.append(np.asarray(a))
                TransientAbsorption_inputs.append(a)
            else:
                numeric_args.append(a)

        def _convert_kwval(value: Any) -> Any:
            # Convert TransientAbsorption to ndarray
            if isinstance(value, TransientAbsorption):
                return np.asarray(value)
            # Check values in list/tuple and convert each item
            elif isinstance(value, (list, tuple)):
                t: type = type(value)
                return t(_convert_kwval(x) for x in value)
            # Check values in dictionary items and convert each
            elif isinstance(value, dict):
                return {kk: _convert_kwval(vv) for kk, vv in value.items()}
            # Return original value if not TransientAbsorption, list, tuple, or dict
            else:
                return value

        numeric_kwargs: dict[str, Any] = {k: _convert_kwval(v) for k, v in (kwargs or {}).items()}

        # Find axis (if not provided in kwargs, try positional)
        axis: int | tuple[int, ...] | None = kwargs.get("axis", None) if kwargs is not None else None
        if axis is None and len(args) >= 2:
            possible_axis: object = args[1]
            if isinstance(possible_axis, (int, tuple, list)):
                axis: int | tuple[int, ...] | list[int] = possible_axis

        # Call numpy implementation
        result: object = func(*numeric_args, **numeric_kwargs)

        # Support functions that return tuples (e.g. returned=True)
        returned_tuple: bool = isinstance(result, tuple)
        results_list: list = list(result) if returned_tuple else [result]
        wrapped_results: list = []

        for res in results_list:
            # Non-array scalar -> return as-is
            if not isinstance(res, np.ndarray):
                wrapped_results.append(res)
                continue

            # Helper: normalize axis description to tuple of axes
            def _normalize_axis(
                    ax: int | tuple[int, ...] | list[int] | None,
                    ndim: int
                ) -> tuple[int, ...]:

                if ax is None:
                    return tuple(range(ndim))

                elif isinstance(ax, (list, tuple)):
                    ax_t: tuple[int, ...] = tuple(((i + ndim) if i < 0 else i) for i in ax)
                    return ax_t

                else:
                    ax_i: int = int(ax)
                    if ax_i < 0:
                        ax_i = ax_i + ndim

                    return (ax_i,)

            orig_ndim: int = self.ndim
            reduced_axes: tuple[int, ...] = _normalize_axis(axis, orig_ndim)
            surviving_axes: list[int] = [i for i in range(orig_ndim) if i not in reduced_axes]

            # Helper: find a common coordinate for axis_idx among all TransientAbsorption inputs, otherwise None
            def _common_coord_for_axis(
                    axis_idx: int
                ) -> NDArray[np.floating] | None:

                # Return None if empty
                if not TransientAbsorption_inputs:
                    return None

                first: NDArray[np.floating] | None = getattr(TransientAbsorption_inputs[0], "x" if axis_idx == 1 else "y", None)
                # Return None if first axis is None
                if first is None:
                    return None
                for m in TransientAbsorption_inputs[1:]:
                    other_coord = getattr(m, "x" if axis_idx == 1 else "y", None)
                    # Return None if any axis is None
                    if other_coord is None:
                        return None

                    # Return None if axes are not equal
                    if not np.array_equal(np.asarray(first), np.asarray(other_coord)):
                        return None

                # Return axis if all axes are equivalent
                return first

            # Decide what coords survive and normalize them into the canonical types:
            # - for 2-D result: x,y -> 1-D np.ndarray or None
            # - for 1-D result: one surviving coord -> 1-D np.ndarray (not scalar) for convenience
            # - for 0-D result: coords set to None (keeping NumPy scalar semantics)
            new_x: NDArray[np.floating] | None = None
            new_y: NDArray[np.floating] | None = None

            if res.ndim == 2:
                cand_x: NDArray[np.floating] | None = _common_coord_for_axis(1)
                cand_y: NDArray[np.floating] | None = _common_coord_for_axis(0)
                new_x = np.asarray(cand_x) if cand_x is not None else None
                new_y = np.asarray(cand_y) if cand_y is not None else None

            elif res.ndim == 1:
                if len(surviving_axes) == 1:
                    surv: int = surviving_axes[0]
                    # Operation preserved columns
                    if surv == 1:
                        cand_x: NDArray[np.floating] | None = _common_coord_for_axis(1)
                        new_x = np.asarray(cand_x) if cand_x is not None else None
                        new_y = None
                    # Operation preserved rows
                    else:
                        cand_y: NDArray[np.floating] | None = _common_coord_for_axis(0)
                        new_y = np.asarray(cand_y) if cand_y is not None else None
                        new_x = None
                else:
                    # ambiguous -> no coords
                    new_x = None
                    new_y = None

            elif res.ndim == 0:
                new_x = None
                new_y = None

            # Now wrap result as TransientAbsorption and attach coords (safe types)
            out: TransientAbsorption = res.view(TransientAbsorption)

            # For 1-D results, ensure coords are 1-D arrays (not scalars) when present
            if new_x is not None:
                object.__setattr__(out, "x", np.asarray(new_x))
            else:
                object.__setattr__(out, "x", None)

            if new_y is not None:
                object.__setattr__(out, "y", np.asarray(new_y))
            else:
                object.__setattr__(out, "y", None)

            wrapped_results.append(out)

        return tuple(wrapped_results) if returned_tuple else wrapped_results[0]

    def __repr__(self) -> str:
        base: str = super().__repr__()
        # present coords cleanly and avoid indexing None or scalar coords
        def coord_repr(
                c:NDArray[np.floating] | float | None
            ) -> str:

            if c is None:
                return "None"
            # convert 0-d numpy scalars into Python scalars for nicer display
            arr: NDArray[np.floating] = np.asarray(c)
            if arr.ndim == 0:
                return repr(arr.item())

            return repr(arr)

        return f"{base}\nmeta: x={coord_repr(getattr(self, 'x', None))}, y={coord_repr(getattr(self, 'y', None))}"

    def fit_global_kinetics(
        self,
        *args: tuple,
        **kwargs: dict[str, Any],
    ) -> dict:
        """
        Fit data to a kinetic model. See phoskhemia.fitting.global_fit.fit_global_kinetics(). 

        Returns
        -------
        dict
            A dictionary of the fit results.
        """

        from phoskhemia.fitting.global_fit import fit_global_kinetics
        return fit_global_kinetics(self, *args, **kwargs)


