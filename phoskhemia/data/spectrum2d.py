from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
from phoskhemia.data.meta import MetaDict, meta_copy_update, subset_meta
from phoskhemia.utils.indexing import _nearest_index

if TYPE_CHECKING:
    from phoskhemia.fitting.results import GlobalFitResult
    from phoskhemia.kinetics.base import KineticModel
    from phoskhemia.fitting.solvers import Solver

class TransientAbsorption(np.ndarray):
    """
    Represents a 2-D transient absorption dataset.

    Shape : (n_times, n_wavelengths)

    Attributes
    ----------
    x : ndarray
        Wavelength axis (length = n_wavelengths).
    y : ndarray
        Time axis (length = n_times).

    Methods
    -------
    add(other, mode='average', fill_value=None)
        Equivalent to combine().
    combine(arr2, mode='average', fill_value=None)
        Combines two TransientAbsorption arrays into one dataset.
    downsample_time(method='log', aggregate='none', **kwargs)
    
    fit_global_kinetics(*args, **kwargs)
        Fit the dataset to a kinetic model.
    smooth(window, normalize=True, separable_tol=1e-10, **kwargs)
        Smooths the data with a provided window or window size.
    spectrum(time, method='nearest', aggregate=0, return_index=False)
    
    svd_denoise(method=ek18, value_rotation='exclude', threshold=0.05, center='time', noise=None, weight='none', return_details=False)
    
    time_zero(mode='truncate_shift', t0=0, use_pre_t0_for_noise=True, noise_method='std')
    
    trace(wavelength_nm, method='nearest', return_index=False)
    
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
    meta: MetaDict
    """ Dictionary of other metadata.
    """


    def __new__(
            cls, 
            input_array: NDArray[np.floating], 
            x: NDArray[np.floating] | None = None, 
            y: NDArray[np.floating] | None = None, 
            meta: dict | None = None,
            freeze_axes: bool = True,
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

        x: NDArray[np.floating] = np.array(x, dtype=float, copy=True)
        y: NDArray[np.floating] = np.array(y, dtype=float, copy=True)
        if freeze_axes:
            x.flags.writeable = False
            y.flags.writeable = False

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-D arrays")
        if x.shape[0] != ncols:
            raise ValueError(f"length of x ({x.shape[0]}) must equal number of columns ({ncols})")
        if y.shape[0] != nrows:
            raise ValueError(f"length of y ({y.shape[0]}) must equal number of rows ({nrows})")

        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        meta: MetaDict = meta_copy_update(meta)
        object.__setattr__(obj, "meta", meta)
        object.__setattr__(obj, "freeze_axes", bool(freeze_axes))

        # One last validation
        if obj.ndim != 2:
            raise ValueError("TransientAbsorption must be 2D")
        if obj.shape != (len(obj.y), len(obj.x)):
            raise ValueError("Data/axis mismatch")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        # share frozen axes (cheap)
        object.__setattr__(self, "x", getattr(obj, "x", None))
        object.__setattr__(self, "y", getattr(obj, "y", None))

        # meta: copy mapping to avoid accidental cross-view mutation
        m = getattr(obj, "meta", None)
        if m is None:
            m2 = MetaDict.coerce({})
        else:
            # shallow copy; does not copy nested arrays (which is fine)
            m2 = MetaDict.coerce(dict(m))
        object.__setattr__(self, "meta", m2)

        object.__setattr__(self, "freeze_axes", getattr(obj, "freeze_axes", True))

    def __add__(
            self, 
            other: TransientAbsorption
        ) -> TransientAbsorption:

        # Non-TransientAbsorption (scalar or ndarray)
        if not isinstance(other, TransientAbsorption):
            try:
                return np.add(self, other)
            except TypeError:
                return NotImplemented

        # Both TransientAbsorption -> default average
        return self.combine(other, fill_val=None, mode="average")

    def __array_function__(self, func, types, args, kwargs):
        """Use NumPy semantics and preserve coordinates only for safe operations.

        Shape-changing operations and reductions return ordinary arrays/scalars.
        Nested arguments are converted too (notably concatenate/stack inputs).
        """
        if not all(issubclass(t, (np.ndarray, TransientAbsorption)) for t in types):
            return NotImplemented

        inputs = []

        def convert(value):
            if isinstance(value, TransientAbsorption):
                inputs.append(value)
                return np.asarray(value)
            if isinstance(value, tuple):
                return tuple(convert(v) for v in value)
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        result = func(*convert(args), **convert(kwargs or {}))
        coordinate_preserving = {
            np.copy, np.clip, np.round, np.conj, np.conjugate,
            np.atleast_1d, np.atleast_2d, np.cumsum, np.cumprod,
        }
        if (
            func in coordinate_preserving
            and isinstance(result, np.ndarray)
            and result.ndim == 2
            and result.shape == self.shape
            and all(
                np.array_equal(a.x, self.x) and np.array_equal(a.y, self.y)
                for a in inputs
            )
        ):
            return TransientAbsorption(
                result, x=self.x, y=self.y, meta=meta_copy_update(self.meta),
                freeze_axes=self.freeze_axes, dtype=result.dtype,
            )
        return result

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        """Keep ufunc/reduction results from acquiring invalid coordinate axes."""
        result = np.asarray(out_arr)
        if result.ndim == 0:
            return result[()]
        compatible = result.ndim == 2 and result.shape == self.shape
        if compatible and context is not None:
            for arg in context[1]:
                if isinstance(arg, TransientAbsorption):
                    compatible = compatible and (
                        np.array_equal(arg.x, self.x)
                        and np.array_equal(arg.y, self.y)
                    )
        if compatible:
            return TransientAbsorption(
                result, x=self.x, y=self.y, meta=meta_copy_update(self.meta),
                freeze_axes=self.freeze_axes, dtype=result.dtype,
            )
        return result

    def transpose(self, *axes):
        """Return an ndarray: transposition swaps the physical axis roles."""
        return np.asarray(self).transpose(*axes)

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        return np.asarray(self).swapaxes(axis1, axis2)

    def reshape(self, *shape, order="C", **kwargs):
        return np.asarray(self).reshape(*shape, order=order, **kwargs)

    def ravel(self, order="C"):
        return np.asarray(self).ravel(order=order)

    def flatten(self, order="C"):
        return np.asarray(self).flatten(order=order)

    def squeeze(self, axis=None):
        return np.asarray(self).squeeze(axis=axis)

    def take(self, indices, axis=None, out=None, mode="raise"):
        return np.asarray(self).take(indices, axis=axis, out=out, mode=mode)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return np.asarray(self).diagonal(offset=offset, axis1=axis1, axis2=axis2)

    def repeat(self, repeats, axis=None):
        return np.asarray(self).repeat(repeats, axis=axis)

    def compress(self, condition, axis=None, out=None):
        return np.asarray(self).compress(condition, axis=axis, out=out)

    def choose(self, choices, out=None, mode="raise"):
        return np.asarray(self).choose(choices, out=out, mode=mode)

    def dot(self, other, out=None):
        return np.asarray(self).dot(np.asarray(other), out=out)

    def resize(self, *args, **kwargs):
        raise ValueError("In-place resizing would invalidate TA axes; create a new dataset instead")

    def __getitem__(self, key):
        """Slice a rectangular TA dataset together with its aligned metadata.

        Two-dimensional selections with separable time/wavelength indices remain
        TransientAbsorption objects. Scalars, individual traces/spectra and
        paired advanced selections follow ordinary NumPy return conventions.
        """
        result = np.ndarray.__getitem__(self, key)
        if not isinstance(result, np.ndarray):
            return result
        if result.ndim != 2:
            return np.asarray(result)

        indices = list(key) if isinstance(key, tuple) else [key]
        if any(index is None for index in indices):
            return np.asarray(result)
        ellipses = [i for i, index in enumerate(indices) if index is Ellipsis]
        if ellipses:
            i = ellipses[0]
            indices[i:i + 1] = [slice(None)] * (3 - len(indices))
        indices += [slice(None)] * (2 - len(indices))
        if len(indices) != 2:
            return np.asarray(result)
        row_index, col_index = indices

        def axis_selection(axis, index):
            if axis is None or np.ndim(axis) != 1:
                return None
            selected = np.asarray(axis)[index]
            selected = np.asarray(selected)
            # np.ix_ supplies (n,1) and (1,m) independent index arrays.
            if selected.ndim == 2 and 1 in selected.shape:
                selected = selected.reshape(-1)
            return selected if selected.ndim == 1 else None

        new_y = axis_selection(self.y, row_index)
        new_x = axis_selection(self.x, col_index)
        if (
            new_x is None or new_y is None
            or result.shape != (new_y.size, new_x.size)
        ):
            return np.asarray(result)

        return TransientAbsorption(
            np.asarray(result), x=new_x, y=new_y,
            meta=subset_meta(self.meta, self.shape, row_index, col_index),
            freeze_axes=self.freeze_axes, dtype=result.dtype,
        )

    def __radd__(
            self, 
            other: TransientAbsorption
        ) -> TransientAbsorption:

        return self.__add__(other)

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

    def add(
            self, 
            other: TransientAbsorption, 
            *,
            mode: Literal['average', 'mean'] = 'average', 
            fill_val: float | None = None
        ) -> TransientAbsorption:
        """
        Alias for combine().

        Parameters
        ----------
        other : TransientAbsorption
            The array to combine with.
        mode : Literal['average', 'mean'], optional
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

    # TODO - Potentially add more mode and fill_val options, such as interpolation.
    # TODO - Currently not the most robust, eventually need to refactor. 
    def combine(
            self, 
            arr2: TransientAbsorption,
            *,
            mode: Literal['average', 'mean'] = 'average',
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
        

        Notes
        -----
        ``a + b`` retains its historical meaning of averaging/combining datasets;
        use ``np.add(a, b)`` for numerical elementwise addition. For identical
        grids, noise standard deviations of distinct inputs are propagated as
        independent measurements: ``hypot(sigma_a, sigma_b) / 2``. Combining an
        object with itself leaves its noise unchanged. For merged grids, noise
        is cleared because overlap changes its time/wavelength dependence;
        reasons are recorded in ``meta["combine_metadata_dropped"]``. Probe
        metadata is retained only when both inputs describe the same values on
        the unchanged wavelength axis. Coordinate comparisons are exact: this
        routine does not perform approximate alignment or resampling. Partial
        overlaps require increasing uniform grids with matching spacing and a
        contiguous suffix/prefix overlap; irregular or contained overlaps raise
        a descriptive error.

        Parameters
        ----------
        arr2 : TransientAbsorption
            The array to combine with.
        mode : Literal['average', 'mean'], optional
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

        if not isinstance(arr2, TransientAbsorption):
            raise TypeError("combine expects another TransientAbsorption instance")

        mode = mode.casefold()
        if mode not in self._COMBINE_MODES:
            raise ValueError(f"mode must be one of {self._COMBINE_MODES}")

        arr1: TransientAbsorption = self
        same_grid = np.array_equal(arr1.x, arr2.x) and np.array_equal(arr1.y, arr2.y)
        combined_meta = meta_copy_update(self.meta, {"combined": True, "combine_mode": mode})
        dropped = {}

        def drop_metadata(key, reason):
            combined_meta.pop(key, None)
            if key in arr1.meta or key in arr2.meta:
                dropped[key] = reason

        if same_grid:
            sigma1, sigma2 = arr1.meta.get("noise_t0"), arr2.meta.get("noise_t0")
            if sigma1 is not None and sigma2 is not None:
                sigma1, sigma2 = np.asarray(sigma1, dtype=float), np.asarray(sigma2, dtype=float)
                valid_shapes = ((), (self.shape[1],), self.shape)
                if (
                    sigma1.shape in valid_shapes and sigma2.shape in valid_shapes
                    and np.isfinite(sigma1).all() and np.isfinite(sigma2).all()
                    and (sigma1 >= 0).all() and (sigma2 >= 0).all()
                ):
                    sigma = np.array(sigma1, copy=True) if arr1 is arr2 else np.hypot(sigma1, sigma2) / 2
                    combined_meta["noise_t0"] = float(sigma) if sigma.ndim == 0 else sigma
                    combined_meta["combine_noise_assumption"] = (
                        "same input object; no noise reduction" if arr1 is arr2
                        else "independent input measurements"
                    )
                else:
                    drop_metadata("noise_t0", "Input standard deviations are invalid or do not match the axes")
            else:
                drop_metadata("noise_t0", "Both input noise estimates are required to propagate an average")
        else:
            drop_metadata("noise_t0", "Merged grids need time/wavelength dependent noise propagation")

        probe1, probe2 = arr1.meta.get("probe_transmittance"), arr2.meta.get("probe_transmittance")
        if (
            np.array_equal(arr1.x, arr2.x) and probe1 is not None and probe2 is not None
            and np.array_equal(probe1, probe2)
            and np.shape(probe1) in ((), (self.shape[1],))
        ):
            combined_meta["probe_transmittance"] = (
                float(probe1) if np.ndim(probe1) == 0 else np.array(probe1, copy=True)
            )
        else:
            drop_metadata("probe_transmittance", "Probe metadata differs, is missing, or the wavelength axis changes")
        if dropped:
            combined_meta["combine_metadata_dropped"] = dropped

        if same_grid:
            # Preserve measured coordinates, including irregular/log time grids.
            return TransientAbsorption(
                (np.asarray(arr1) + np.asarray(arr2)) / 2,
                x=arr1.x, y=arr1.y, meta=combined_meta, freeze_axes=self.freeze_axes,
            )

        intersection: NDArray[np.floating]
        idx1: NDArray[np.integer]
        idx2: NDArray[np.integer]
        axis: int
        ax1: NDArray[np.floating]
        ax2: NDArray[np.floating]

        # TODO - May refactor this part to be easier to follow and so there is not so much repetition.
        # Check which axis we are combining along and adjust order of arrays if necessary.
        # If ys match, stack x.
        if np.array_equal(arr1.y, arr2.y):
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
                arr1.x, 
                arr2.x, 
                return_indices=True
            )
            diff: float = (
                (np.max(arr2.x) - np.max(arr1.x)) 
                if intersection.size > 0 else (np.min(arr2.x) - np.max(arr1.x))
            )
            if diff < 0:
                arr1, arr2 = arr2, arr1
                idx1, idx2 = idx2, idx1

            axis, ax1, ax2 = 1, arr1.x, arr2.x

        # If xs match, stack y.
        elif np.array_equal(arr1.x, arr2.x):
            intersection, idx1, idx2 = np.intersect1d(
                arr1.y,
                arr2.y,
                return_indices=True
            )
            if intersection.size == 0 and (
                max(np.min(arr1.y), np.min(arr2.y))
                <= min(np.max(arr1.y), np.max(arr2.y))
            ):
                raise ValueError(
                    "Time ranges overlap without exact shared coordinates; "
                    "explicitly align or resample the time grids before combining them"
                )
            diff: float = (
                (np.max(arr2.y) - np.max(arr1.y)) 
                if intersection.size > 0 else (np.min(arr2.y) - np.max(arr1.y))
            )
            if diff < 0:
                arr1, arr2 = arr2, arr1
                idx1, idx2 = idx2, idx1

            axis, ax1, ax2 = 0, arr1.y, arr2.y

        else:
            raise ValueError(
                "Either x or y axes must match in both arrays; "
                "arrays with different numbers of rows and columns are not supported."
                )

        # Overlapping arrays
        if intersection.size > 0:
            steps1, steps2 = np.diff(ax1), np.diff(ax2)
            if (
                steps1.size == 0 or steps2.size == 0
                or not np.isfinite(ax1).all() or not np.isfinite(ax2).all()
                or np.any(steps1 <= 0) or np.any(steps2 <= 0)
                or not np.allclose(steps1, steps1[0], rtol=1e-6, atol=0.0)
                or not np.allclose(steps2, steps2[0], rtol=1e-6, atol=0.0)
            ):
                raise ValueError(
                    "Partial overlap requires strictly increasing, uniformly spaced grids; "
                    "align irregular grids explicitly before combining them"
                )
            dax1, dax2 = float(steps1[0]), float(steps2[0])
            if not np.isclose(dax1, dax2, rtol=1e-6, atol=0.0):
                raise ValueError("Overlapping arrays with different axis spacings are not supported.")
            if (
                intersection.size == min(ax1.size, ax2.size)
                or not np.array_equal(idx1, np.arange(ax1.size - intersection.size, ax1.size))
                or not np.array_equal(idx2, np.arange(intersection.size))
            ):
                raise ValueError(
                    "Contained or noncontiguous overlapping grids are not supported; "
                    "only a suffix of the first grid overlapping a prefix of the second is supported"
                )
            # The guarded layout lets us preserve every measured coordinate,
            # rather than inventing a uniform axis from floating-point spacing.
            new_ax = np.concatenate((ax1, ax2[intersection.size:]))

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
                TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y), meta=combined_meta) 
                if axis == 1 else TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax, meta=combined_meta)
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
            if fill_val is None or (len(unique) == 2 and not np.any(counts == 1)):
                new_ax: NDArray[np.floating] = comb
                new_data: NDArray[np.floating] = np.concatenate((np.array(arr1), np.array(arr2)), axis=axis)
                return (
                    TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y), meta=combined_meta) 
                    if axis == 1 else TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax, meta=combined_meta)
                )

            # Fill values between arrays with fill_value using the average stepsize of the two axes.
            else:
                dax: float = 0.5 * (np.mean(np.diff(ax1)) + np.mean(np.diff(ax2)))
                filled: NDArray[np.floating] = np.arange(np.max(ax1) + dax, np.min(ax2), dax)
                new_ax: NDArray[np.floating] = np.concatenate((ax1, filled, ax2))
                if axis == 1:
                    new_data: NDArray[np.floating] = np.zeros((len(arr1.y), len(new_ax))) + fill_val
                    new_data[:, :len(ax1)] = np.array(arr1)
                    new_data[:, len(ax1)+len(filled):] = np.array(arr2)
                    return TransientAbsorption(new_data, x=new_ax, y=np.copy(arr1.y), meta=combined_meta)

                elif axis == 0:
                    new_data: NDArray[np.floating] = np.zeros((len(new_ax), len(arr1.x))) + fill_val
                    new_data[:len(ax1), :] = np.array(arr1)
                    new_data[len(ax1)+len(filled):, :] = np.array(arr2)
                    return TransientAbsorption(new_data, x=np.copy(arr1.x), y=new_ax, meta=combined_meta)

    def downsample_time(
            self,
            *,
            method: Literal['log', 'hybrid', 'linear'] = 'log',
            aggregate: Literal['none', 'mean', 'median', 'min', 'max'] = 'none',
            **kwargs: Any
        ) -> TransientAbsorption:
        """Select or aggregate time samples without preserving stale noise.

        Averaging scales standard deviations by the actual number of samples
        in each bin, assuming independent observations. A scalar/per-wavelength
        ``noise_t0`` remains representable only when every bin has the same
        count. Otherwise it is cleared, with counts and a reason recorded in
        metadata; this fitter currently expects noise constant along time.
        Noise is also cleared for nonlinear aggregation statistics.
        """
        from phoskhemia.preprocessing.downsampling import make_time_indices

        if method == "linear" and kwargs.get("stride") is None:
            raise ValueError("stride must be specified for method='linear'")
        indices = make_time_indices(self.y, method=method, **kwargs)
        if aggregate == 'none':
            array = self[indices, :]
            array.meta["downsample_idx_len"] = int(indices.size)
            return array
        if aggregate not in ('mean', 'median', 'min', 'max'):
            raise ValueError("aggregate must be one of 'mean', 'median', 'min', 'max', or 'none'")

        from phoskhemia.preprocessing.downsampling import downsample_time_binned
        return downsample_time_binned(self, indices, data_stat=aggregate)

    # TODO - May update this to be less verbose with arguments.
    def export_csv(
            self,
            path: str | Path,
            *,
            kind: Literal["auto", "dataset", "trace", "spectrum"] = "auto",
            format: Literal["wide", "long"] = "wide",
            delimiter: str = ",",
            float_format: str = "{:.17g}",
            t: float | None = None,
            w: float | None = None,
            t_idx: slice | NDArray[np.int64] | None = None,
            w_idx: slice | NDArray[np.int64] | None = None,
            warn_mb: float = 50.0,
            max_mb: float = 250.0,
            force: bool = False,
            chars_per_value: int = 20,
        ) -> Path:
        """
        Export slices of TA to CSV.
        
        Notes
        -----
        This API is functional but may change in a future release. Future updates
        would likely change the slice-selection interface (w and w_idx, t and t_idx)
        and formatting controls to be more robust and easy to use.

        1D:
            - kind="trace": time vs signal at wavelength w (or w_idx of length 1)
            - kind="spectrum": wavelength vs signal at time t (or t_idx of length 1)

        2D:
            - kind="dataset": time×wavelength slice (t_idx, w_idx)
            format="wide": rows=time, cols=wavelengths (+time column)
            format="long": rows=time*wavelength with columns (time, wavelength, signal)

        Size guard:
            - warns at warn_mb
            - raises at max_mb unless force=True
        Parameters
        ----------
        path : str | Path
            _description_
        kind : Literal["auto", "dataset", "trace", "spectrum"], optional
            _description_, by default "auto"
        format : Literal["wide", "long"], optional
            _description_, by default "wide"
        delimiter : str, optional
            _description_, by default ","
        float_format : _type_, optional
            _description_, by default "{:.17g}"
        t : float | None, optional
            _description_, by default None
        w : float | None, optional
            _description_, by default None
        t_idx : slice | NDArray[np.int64] | None, optional
            _description_, by default None
        w_idx : slice | NDArray[np.int64] | None, optional
            _description_, by default None
        warn_mb : float, optional
            _description_, by default 50.0
        max_mb : float, optional
            _description_, by default 250.0
        force : bool, optional
            _description_, by default False
        chars_per_value : int, optional
            _description_, by default 20

        Returns
        -------
        Path
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """

        
        from phoskhemia.utils.formatting import estimate_csv_bytes

        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        data: NDArray[np.floating] = np.asarray(self, dtype=float)
        times: NDArray[np.floating] = np.asarray(self.y, dtype=float).reshape(-1)
        waves: NDArray[np.floating] = np.asarray(self.x, dtype=float).reshape(-1)

        # infer kind if needed
        if kind == "auto":
            if w is not None or (w_idx is not None and not isinstance(w_idx, slice)):
                kind = "trace" if t is None and t_idx is None else "dataset"
            elif t is not None or (t_idx is not None and not isinstance(t_idx, slice)):
                kind = "spectrum" if w is None and w_idx is None else "dataset"
            else:
                kind = "dataset"

        # build concrete indices
        if kind == "trace":
            if w is None and w_idx is None:
                raise ValueError("trace export requires w or w_idx")
            j: int = _nearest_index(waves, w) if w is not None else int(np.asarray(w_idx).reshape(-1)[0])
            y1: NDArray[np.floating] = data[:, j]
            # size: 2 columns, n_t rows
            est: int = estimate_csv_bytes(times.size, 2, chars_per_value=chars_per_value, delimiter=delimiter)
            self._csv_size_guard(est_bytes=est, warn_mb=warn_mb, max_mb=max_mb, force=force)

            with path.open("w", newline="") as f:
                writer: csv._writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(["time", "signal"])
                for tt, vv in zip(times, y1):
                    writer.writerow([float_format.format(tt), float_format.format(vv)])
            return path

        if kind == "spectrum":
            if t is None and t_idx is None:
                raise ValueError("spectrum export requires t or t_idx")
            i: int = _nearest_index(times, t) if t is not None else int(np.asarray(t_idx).reshape(-1)[0])
            y1: NDArray[np.floating] = data[i, :]
            est: int = estimate_csv_bytes(waves.size, 2, chars_per_value=chars_per_value, delimiter=delimiter)
            self._csv_size_guard(est_bytes=est, warn_mb=warn_mb, max_mb=max_mb, force=force)

            with path.open("w", newline="") as f:
                writer: csv._writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(["wavelength", "signal"])
                for ww, vv in zip(waves, y1):
                    writer.writerow([float_format.format(ww), float_format.format(vv)])
            return path

        if kind != "dataset":
            raise ValueError("kind must be one of auto|dataset|trace|spectrum")

        ti = t_idx if t_idx is not None else slice(None)
        wi = w_idx if w_idx is not None else slice(None)

        data_s: NDArray[np.floating] = data[ti, :][:, wi]
        t_s: NDArray[np.floating] = times[ti]
        w_s: NDArray[np.floating] = waves[wi]

        if format == "wide":
            # columns: time + wavelengths
            n_rows: int = int(data_s.shape[0])
            n_cols: int = int(data_s.shape[1]) + 1
            est: int = estimate_csv_bytes(n_rows, n_cols, chars_per_value=chars_per_value, delimiter=delimiter)
            self._csv_size_guard(est_bytes=est, warn_mb=warn_mb, max_mb=max_mb, force=force)

            with path.open("w", newline="") as f:
                writer: csv._writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(["time"] + [float_format.format(ww) for ww in w_s])
                for i, tt in enumerate(t_s):
                    writer.writerow([float_format.format(tt)] + [float_format.format(vv) for vv in data_s[i, :]])
            return path

        if format == "long":
            # rows: n_t * n_w, cols: 3
            n_rows: int = int(data_s.size)
            est: int = estimate_csv_bytes(n_rows, 3, chars_per_value=chars_per_value, delimiter=delimiter)
            self._csv_size_guard(est_bytes=est, warn_mb=warn_mb, max_mb=max_mb, force=force)

            with path.open("w", newline="") as f:
                writer: csv._writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(["time", "wavelength", "signal"])
                for i, tt in enumerate(t_s):
                    for j, ww in enumerate(w_s):
                        writer.writerow([float_format.format(tt), float_format.format(ww), float_format.format(data_s[i, j])])
            return path

        raise ValueError("format must be 'wide' or 'long'")

    def fit_global_kinetics(
        self,
        kinetic_model: KineticModel,
        beta0: NDArray[np.floating],
        *,
        noise: NDArray[np.floating] | float | int | None = None,
        lam: float = 0.0,
        propagate_kinetic_uncertainty: bool = False,
        ci_sigma: float | None = None,
        ci_level: float | None = None,
        debug: bool = False,
        solver: str | Solver = "least_squares",
        maxit: int = 50,
        use_noise_as_sy: bool = True,
        sx: NDArray[np.floating] | None = None,
        scope_ppm: float | None = None,
        scope_jitter: float | None = None,
        **kwargs,
    ) -> GlobalFitResult:
        """
        Fit data to a kinetic model. See phoskhemia.fitting.global_fit.fit_global_kinetics(). 

        Perform a global kinetic fit using variable projection.

        Parameters
        ----------
        kinetic_model : KineticModel
            Kinetic model instance
        beta0 : NDArray[np.floating]
            Initial guesses in the coordinates required by kinetic_model.parameterization().
        noise : NDArray[np.floating] | float | int | None, optional
            Per-wavelength noise σ(λ)
        lam : float, optional
            Tikhonov regularization strength
        propagate_kinetic_uncertainty : bool, optional
            Propagate kinetic covariance into amplitude uncertainties
        ci_sigma : float | None, optional
        
        ci_level : float | None, optional

        debug : bool, optional
            Print diagnostic messages on failures

        Returns
        -------
        GlobalFitResult
            A structured object of the fit results.
        """

        from phoskhemia.fitting.global_fit import fit_global_kinetics
        result = fit_global_kinetics(
            self, 
            kinetic_model, 
            beta0, 
            noise=noise, 
            lam=lam, 
            propagate_kinetic_uncertainty=propagate_kinetic_uncertainty, 
            ci_sigma=ci_sigma, 
            ci_level=ci_level, 
            debug=debug,
            solver=solver,
            maxit=maxit,
            use_noise_as_sy=use_noise_as_sy,
            sx=sx,
            scope_ppm=scope_ppm,
            scope_jitter=scope_jitter,
            **kwargs,
        )
        return result

    def smooth(
            self,
            window: int | NDArray[np.floating] | tuple[int, int],
            *,
            normalize: bool = True,
            separable_tol: float = 1e-10,
            **kwargs: Any
        ) -> TransientAbsorption:
        """_summary_

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

        Returns
        -------
        TransientAbsorption
            _description_
        """
        from phoskhemia.preprocessing.smoothing import conv_smooth
        result, noise_scale = conv_smooth(self, window, normalize=normalize, separable_tol=separable_tol, **kwargs)
        meta: MetaDict = meta_copy_update(getattr(self, "meta", None), {"smoothed": True, "smooth_window": str(window)})
        nt0: NDArray[np.floating] | float | None = meta.noise_t0
        if isinstance(nt0, np.ndarray):
            meta.noise_t0 = nt0 * noise_scale
        elif nt0 is not None:
            meta.noise_t0 = float(nt0) * noise_scale
        return TransientAbsorption(result, x=self.x, y=self.y, meta=meta)

    def spectrum(
        self,
        time: float,
        *,
        method: Literal["nearest", "interp"] = "nearest",
        aggregate: int = 0,
        return_index: bool = False,
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], int]:
        """
        Return ΔA(λ) at a time.
        Returns shape (n_wl,).
        """

        from phoskhemia.utils.indexing import _nearest_index, _bracketing_indices
        data = np.asarray(self, dtype=float)
        y = np.asarray(self.y, dtype=float)

        if method == "nearest":
            i = _nearest_index(y, time)
            if aggregate > 0:
                dt = np.mean(np.diff(self.y))
                j = _nearest_index(y, time + aggregate * dt)
                out = np.mean(data[i:j+1, :].copy(), axis=0)
            else:
                out = data[i, :].copy()

            return (out, i) if return_index else out

        if method == "interp":
            i0, i1, w = _bracketing_indices(y, time)
            out = (1.0 - w) * data[i0, :] + w * data[i1, :]
            out = np.asarray(out, dtype=float)
            return (out, i0) if return_index else out

        raise ValueError("method must be 'nearest' or 'interp'")

    def svd_denoise(
            self,
            *,
            method: Literal["e15", "ek18"] = "ek18",
            value_rotation: Literal["include", "exclude", "auto"] = "exclude",
            threshold: float = 0.05,
            center: Literal["none", "time"] = "time",
            noise: NDArray[np.floating] | None = None,
            weight: Literal["none", "column"] = "none",
            return_details: bool = False,
        ) -> "TransientAbsorption" | tuple["TransientAbsorption", dict[str, Any]]:

        from phoskhemia.preprocessing.svd_denoise import svd_denoise
        from phoskhemia.data.meta import meta_copy_update

        if noise is None and self.meta.noise_t0 is not None:
            noise = self.meta.noise_t0

        out = svd_denoise(
            np.asarray(self, dtype=float),
            method=method,
            value_rotation=value_rotation,
            threshold=threshold,
            center=center,
            noise=noise,
            weight=weight,
            return_details=return_details,
        )

        if return_details:
            arr_hat, info = out
            meta = meta_copy_update(getattr(self, "meta", None), {})
            meta["svd_denoise"] = info.get("wrapper", {})
            meta["svd_info"] = info
            return TransientAbsorption(arr_hat, x=self.x, y=self.y, meta=meta), info

        meta = meta_copy_update(
            getattr(self, "meta", None),
            {
                "svd_denoise": {
                    "method": method,
                    "value_rotation": value_rotation,
                    "threshold": threshold
                }
            },
        )

        return TransientAbsorption(out, x=self.x, y=self.y, meta=meta)

    def time_zero(
            self,
            *,
            mode: Literal['truncate', 'truncate_shift', 'shift'] = 'truncate_shift',
            t0: float | None = None,
            use_pre_t0_for_noise: bool = True,
            noise_method: Literal['std', 'mad'] = 'std'
        ) -> TransientAbsorption:
        """_summary_

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
        TransientAbsorption
            _description_
        """
        from phoskhemia.preprocessing.corrections import apply_time_zero
        array, _info = apply_time_zero(
            self, 
            mode=mode, 
            t0=t0, 
            use_pre_t0_for_noise=use_pre_t0_for_noise, 
            noise_method=noise_method
        )
        return array

    def trace(
        self,
        wavelength_nm: float,
        *,
        method: Literal["nearest", "interp"] = "nearest",
        return_index: bool = False,
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], int]:
        """
        Return ΔA(t) at a wavelength.
        Returns shape (n_times,).
        """

        from phoskhemia.utils.indexing import _nearest_index, _bracketing_indices
        data = np.asarray(self, dtype=float)
        x = np.asarray(self.x, dtype=float)

        if method == "nearest":
            j = _nearest_index(x, wavelength_nm)
            out = data[:, j].copy()
            return (out, j) if return_index else out

        if method == "interp":
            j0, j1, w = _bracketing_indices(x, wavelength_nm)
            out = (1.0 - w) * data[:, j0] + w * data[:, j1]
            out = np.asarray(out, dtype=float)
            return (out, j0) if return_index else out  # j0 = left bracket

        raise ValueError("method must be 'nearest' or 'interp'")
    
    def trim(
            self,
            axis: Literal["x", "y", "wavelength", "time", "rows", "cols", 0, 1],
            new_range: float | int | tuple[float | int, float | int]
        ) -> TransientAbsorption:
        """Select a closed coordinate range, including its nearest endpoints.

        A scalar is an upper bound; the lower bound is the axis minimum.
        Input order is preserved, including descending coordinate axes.
        """
        if axis in ("x", "wavelength", "cols", 1):
            dimension = 1
            coordinates = np.asarray(self.x, dtype=float)
        elif axis in ("y", "time", "rows", 0):
            dimension = 0
            coordinates = np.asarray(self.y, dtype=float)
        else:
            raise ValueError("axis must identify wavelength/x/cols/1 or time/y/rows/0")

        if coordinates.size == 0 or not np.isfinite(coordinates).all():
            raise ValueError("trim requires a finite, nonempty coordinate axis")
        limits = np.asarray(new_range, dtype=float)
        minimum, maximum = float(coordinates.min()), float(coordinates.max())
        if limits.ndim == 0:
            lower, upper = minimum, float(limits)
        elif limits.shape == (2,):
            lower, upper = map(float, limits)
            if lower >= upper:
                raise ValueError("new_range must have two distinct values in ascending order")
        else:
            raise ValueError("new_range must be a scalar upper bound or a pair of bounds")
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError("new_range must be finite and in ascending order")
        if lower < minimum or upper > maximum:
            raise ValueError("new_range cannot extend outside the coordinate axis")

        lo = float(coordinates[_nearest_index(coordinates, lower)])
        hi = float(coordinates[_nearest_index(coordinates, upper)])
        selected = (coordinates >= min(lo, hi)) & (coordinates <= max(lo, hi))
        out = self[:, selected] if dimension == 1 else self[selected, :]
        out.meta = meta_copy_update(out.meta, {
            "axis": axis, "trimmed": True, "old_range": (minimum, maximum),
        })
        return out

    @classmethod
    def from_arrays(
            cls,
            *,
            data: NDArray[np.floating],
            x: NDArray[np.floating] | None,
            y: NDArray[np.floating] | None,
            meta: dict[str, Any] | None = None,
            freeze_axes: bool = True,
            dtype: type = float,
        ) -> "TransientAbsorption":

        return cls(data, x=x, y=y, meta=meta, freeze_axes=freeze_axes, dtype=dtype)

    @staticmethod
    def _csv_size_guard(*, est_bytes: int, warn_mb: float, max_mb: float, force: bool) -> None:
        est_mb: float = est_bytes / (1024 * 1024)
        if est_mb >= float(warn_mb):
            warnings.warn(
                f"CSV export estimated size ~{est_mb:.1f} MB. "
                "Consider exporting a smaller slice or using NPZ.",
                RuntimeWarning,
                stacklevel=3,
            )
        if (not force) and est_mb >= float(max_mb):
            raise ValueError(
                f"Refusing to export very large CSV (~{est_mb:.1f} MB). "
                f"Increase max_mb or set force=True to override."
            )


