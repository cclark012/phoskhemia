import numpy as np
import copy
from scipy.signal import convolve
from numpy.typing import NDArray
from typing import Callable, Any
from __future__ import annotations

class _AddWithMode:
    def __init__(
            self: _AddWithMode, 
            obj: TransientAbsorption, 
            mode: str
        ) -> None:
        self.obj = obj
        self.mode = mode

class TransientAbsorption(np.ndarray):
    """
    2-D transient absorption dataset.

    Shape
    -----
    (n_times, n_wavelengths)

    Attributes
    ----------
    x : ndarray
        Wavelength axis (length = n_wavelengths)
    y : ndarray
        Time axis (length = n_times)
    
    TODO:
    1) Add functionality for fill_value in add() and combine_with()
    2) Add downsampling for large datasets
    """

    __array_priority__: float = 1000.0
    x: NDArray[np.floating]
    y: NDArray[np.floating]

    _SUPPORTED_ARRAY_FUNCTIONS: dict = {
        np.mean,
        np.sum,
        np.average,
        np.nanmean,
        np.nanmedian,
        np.nanstd,
        np.nanvar,
    }


    def __new__(
            cls: TransientAbsorption, 
            input_array: NDArray[np.floating], 
            x: NDArray[np.floating] | None=None, 
            y: NDArray[np.floating] | None=None, 
            dtype: type=float
        ) -> TransientAbsorption:

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

        # Now arr must be 2-D
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

    def _validate(self: TransientAbsorption) -> None:
        if self.ndim != 2:
            raise ValueError("TransientAbsorption must be 2D")
        if self.shape != (len(self.y), len(self.x)):
            raise ValueError("Data/axis mismatch")

    def __array_finalize__(
            self: TransientAbsorption, 
            obj: TransientAbsorption | None
            ) -> None:

        if obj is None:
            return
        object.__setattr__(self, "x", copy.copy(getattr(obj, "x", None)))
        object.__setattr__(self, "y", copy.copy(getattr(obj, "y", None)))

    def smooth(
        self: TransientAbsorption,
        window: int | NDArray[np.floating] | tuple[int, int],
        *,
        normalize: bool=True,
        separable_tol: float=1e-10,
        **kwargs,
    ) -> TransientAbsorption:
        """
        Smooth data using scipy.signal.convolve.
        Notes
        -----
        Axis conventions:
            axis 0 → time (y)
            axis 1 → wavelength (x)

        Parameters
        ----------
        window : int | array-like | tuple
            - int or 1-D array: smooth along x
            - (1, nx): smooth along x
            - (ny, 1): smooth along y
            - (ny, nx): true 2-D smoothing
        normalize : bool
            Normalize the window so it sums to 1.
        separable_tol : float
            Relative tolerance for separable-kernel detection.
        **kwargs
            Passed directly to scipy.signal.convolve
            (e.g. mode='same', method='auto')
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
            h: NDArray[np.floating] = kernel.ravel()
            out: NDArray[np.floating] = convolve(data, h[None, :], **kwargs)

        # 1-D smoothing along y
        elif ny > 1 and nx == 1:
            v: NDArray[np.floating] = kernel.ravel()
            out: NDArray[np.floating] = convolve(data, v[:, None], **kwargs)

        # 2-D smoothing
        else:
            # Attempt separable acceleration
            U: NDArray[np.floating]
            S: NDArray[np.floating]
            Vt: NDArray[np.floating]
            U, S, Vt = np.linalg.svd(kernel, full_matrices=False)
            separable: bool = S.size > 1 and S[1] / S[0] < separable_tol

            if separable:
                v: NDArray[np.floating] = U[:, 0] * np.sqrt(S[0])
                h: NDArray[np.floating] = Vt[0, :] * np.sqrt(S[0])

                tmp: NDArray[np.floating] = convolve(data, v[:, None], **kwargs)
                out: NDArray[np.floating] = convolve(tmp, h[None, :], **kwargs)

            else:
                out: NDArray[np.floating] = convolve(data, kernel, **kwargs)

        result: TransientAbsorption = out.view(TransientAbsorption)
        result.x = self.x
        result.y = self.y

        return result

    @staticmethod
    def _infer_spacing(axis: NDArray[np.floating]) -> float | None:
        axis: NDArray[np.floating] = np.asarray(axis, dtype=float)
        if axis.size < 2:
            return None
        diffs: NDArray[np.floating] = np.diff(axis)
        return float(np.mean(diffs))

    @staticmethod
    def _continuous_axis(
            a: NDArray[np.floating], 
            b: NDArray[np.floating], 
            tol: float=1e-8
        ) -> tuple[NDArray[np.floating], float | None]:

        a: NDArray[np.floating] = np.asarray(a, dtype=float)
        b: NDArray[np.floating] = np.asarray(b, dtype=float)
        dx_a: float | None = TransientAbsorption._infer_spacing(a)
        dx_b: float | None = TransientAbsorption._infer_spacing(b)

        if dx_a is not None and dx_b is not None:
            if not np.isclose(dx_a, dx_b, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Axis spacings differ: {dx_a} vs {dx_b}")
            dx: float = 0.5 * (dx_a + dx_b)

        else:
            dx: float | None = dx_a if dx_a is not None else dx_b

        start: float = min(a[0], b[0])
        stop: float = max(a[-1], b[-1])

        if dx is None:
            if np.isclose(start, stop, atol=tol):
                return np.array([start], dtype=float), None

            else:
                axis: NDArray[np.floating] = np.array([start, stop], dtype=float)
                return axis, stop - start

        n: int = int(round((stop - start) / dx)) + 1
        axis: NDArray[np.floating] = start + dx * np.arange(n)
        if not np.isclose(axis[-1], stop, rtol=1e-8, atol=1e-12):
            axis: NDArray[np.floating] = np.append(axis, stop)
        return axis, dx

    def _group_and_average_columns(
            self: TransientAbsorption, 
            cols: list[tuple[float, tuple[float, ...]]]
        ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        cols: list of tuples (x_value(float), column_tuple_of_floats)
        returns (x_new_array, vals_matrix) with duplicated x's averaged.
        """
        if not cols:
            return np.array([], dtype=float), np.zeros((0,0), dtype=float)

        # sort by x then by column values (deterministic)
        cols_sorted: list[tuple[float, tuple[float, ...]]] = sorted(cols, key=lambda t: (t[0], t[1]))

        # group consecutive entries with same x (isclose)
        xs: list[float] = [c[0] for c in cols_sorted]
        cols_vals: list[NDArray[np.floating]] = [np.array(c[1], dtype=float) for c in cols_sorted]

        groups: list[tuple[list[float], list[NDArray[np.floating]]]] = []
        current_group_idx: int = 0
        current_group: list[NDArray[np.floating]] = [cols_vals[0]]
        current_xs: list[float] = [xs[0]]

        for xi, ci in zip(xs[1:], cols_vals[1:]):
            if np.isclose(xi, current_xs[-1], atol=1e-8, rtol=1e-6):
                current_group.append(ci)
                current_xs.append(xi)

            else:
                groups.append((current_xs, current_group))
                current_group_idx += 1
                current_group = [ci]
                current_xs = [xi]

        groups.append((current_xs, current_group))

        # For each group compute mean x and mean column (averaging duplicates)
        averaged_cols: list[NDArray[np.floating]] = []
        averaged_xs: list[float] = []
        for x_group, col_group in groups:
            # stack columns (shape (n_members, nrows)) -> average along axis=0 yields (nrows,)
            stacked: NDArray[np.floating] = np.vstack(col_group)  # shape (n_members, nrows)
            mean_col: NDArray[np.floating] = np.mean(stacked, axis=0)
            mean_x: float = float(np.mean(x_group))
            averaged_xs.append(mean_x)
            averaged_cols.append(mean_col)

        vals: NDArray[np.floating] = np.column_stack(averaged_cols) if averaged_cols else np.zeros((stacked.shape[1], 0))
        return np.array(averaged_xs, dtype=float), vals

    def _group_and_average_rows(
            self: TransientAbsorption, 
            rows: list[tuple[float, tuple[float, ...]]]
        ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        rows: list of tuples (y_value(float), row_tuple_of_floats)
        returns (y_new_array, vals_matrix) with duplicated y's averaged.
        """
        if not rows:
            return np.array([], dtype=float), np.zeros((0,0), dtype=float)

        rows_sorted: list[tuple[float, tuple[float, ...]]] = sorted(rows, key=lambda t: (t[0], t[1]))

        ys: list[float] = [r[0] for r in rows_sorted]
        rows_vals: list[NDArray[np.floating]] = [np.array(r[1], dtype=float) for r in rows_sorted]

        groups: list[tuple[list[float], list[NDArray[np.floating]]]] = []
        current_group: list[NDArray[np.floating]] = [rows_vals[0]]
        current_ys: list[float] = [ys[0]]
        for yi, ri in zip(ys[1:], rows_vals[1:]):
            if np.isclose(yi, current_ys[-1], atol=1e-8, rtol=1e-6):
                current_group.append(ri)
                current_ys.append(yi)

            else:
                groups.append((current_ys, current_group))
                current_group = [ri]
                current_ys = [yi]

        groups.append((current_ys, current_group))

        averaged_rows: list[NDArray[np.floating]] = []
        averaged_ys: list[float] = []
        for y_group, row_group in groups:
            stacked: NDArray[np.floating] = np.vstack(row_group)  # shape (n_members, ncols)
            mean_row: NDArray[np.floating] = np.mean(stacked, axis=0)
            mean_y: float = float(np.mean(y_group))
            averaged_ys.append(mean_y)
            averaged_rows.append(mean_row)

        vals: NDArray[np.floating] = np.vstack(averaged_rows) if averaged_rows else np.zeros((0, stacked.shape[1]))
        return np.array(averaged_ys, dtype=float), vals

    def combine_with(
            self: TransientAbsorption, 
            other: TransientAbsorption, 
            fill_value: float=0.0, 
            mode: str="average"
        ) -> TransientAbsorption:

        if not isinstance(other, TransientAbsorption):
            raise TypeError("combine_with expects another TransientAbsorption")

        if mode not in ("average", "concat"):
            raise ValueError("mode must be 'average' or 'concat'")

        if mode == "concat":
            # concat along x if y match exactly — build columns and average duplicates by x
            if np.array_equal(self.y, other.y):
                cols: list[tuple[float, tuple[float, ...]]] = []
                for x_val, col in zip(self.x, np.asarray(self).T):
                    cols.append((float(x_val), tuple(map(float, col))))
                for x_val, col in zip(other.x, np.asarray(other).T):
                    cols.append((float(x_val), tuple(map(float, col))))
                x_new: NDArray[np.floating]
                vals: NDArray[np.floating]
                x_new, vals = self._group_and_average_columns(cols)
                return TransientAbsorption(vals, x=x_new, y=copy.copy(self.y))

            # concat along y if x match exactly — build rows and average duplicates by y
            if np.array_equal(self.x, other.x):
                rows: list[tuple[float, tuple[float, ...]]] = []
                for y_val, row in zip(self.y, np.asarray(self)):
                    rows.append((float(y_val), tuple(map(float, row))))

                for y_val, row in zip(other.y, np.asarray(other)):
                    rows.append((float(y_val), tuple(map(float, row))))

                y_new: NDArray[np.floating]
                vals: NDArray[np.floating]
                y_new, vals = self._group_and_average_rows(rows)
                return TransientAbsorption(vals, x=copy.copy(self.x), y=y_new)

            # else refuse
            raise ValueError(
                "concat mode requires exact match on the orthogonal axis: "
                "either self.y == other.y (concat columns) or self.x == other.x (concat rows)."
            )

        # average mode
        x_new: NDArray[np.floating]
        y_new: NDArray[np.floating]
        dx: float | None
        dy: float | None
        x_new, dx = self._continuous_axis(self.x, other.x)
        y_new, dy = self._continuous_axis(self.y, other.y)

        new_vals_sum: NDArray[np.floating] = np.full((len(y_new), len(x_new)), 0.0, dtype=float)
        new_counts: NDArray[np.floating] = np.zeros_like(new_vals_sum, dtype=float)

        def find_index(
                axis: NDArray[np.floating], 
                val: float
            ) -> int:

            idx = np.flatnonzero(np.isclose(axis, val, atol=1e-8, rtol=1e-6))
            if idx.size == 0:
                idx = np.array([int(np.argmin(np.abs(axis - val)))])

            return int(idx[0])

        def place_into(
                acc_sum: NDArray[np.floating], 
                acc_count: NDArray[np.floating], 
                src: TransientAbsorption, 
                src_x: NDArray[np.floating], 
                src_y: NDArray[np.floating]
            ) -> None:

            x0: int = find_index(x_new, src_x[0])
            y0: int = find_index(y_new, src_y[0])
            r0: int
            c0: int
            r1: int
            c1: int
            r0, r1 = y0, y0 + src.shape[0]
            c0, c1 = x0, x0 + src.shape[1]
            acc_sum[r0:r1, c0:c1] += np.asarray(src, dtype=float)
            acc_count[r0:r1, c0:c1] += 1.0

        place_into(new_vals_sum, new_counts, self, self.x, self.y)
        place_into(new_vals_sum, new_counts, other, other.x, other.y)

        with np.errstate(invalid='ignore', divide='ignore'):
            averaged = np.where(new_counts > 0, new_vals_sum / new_counts, fill_value)

        return TransientAbsorption(averaged, x=x_new, y=y_new)

    def add(
            self: TransientAbsorption, 
            other: TransientAbsorption, 
            mode: str="average", 
            fill_value: float=0.0
        ) -> TransientAbsorption:

        if mode in ("average", "concat"):
            return self.combine_with(other, fill_value=fill_value, mode=mode)

        raise ValueError("mode must be 'average' or 'concat'")

    def __add__(
            self: TransientAbsorption, 
            other: TransientAbsorption | _AddWithMode
        ) -> TransientAbsorption:

        if isinstance(other, _AddWithMode):
            target: TransientAbsorption = other.obj
            mode = other.mode
            if not isinstance(target, TransientAbsorption):
                return NotImplemented
            return self.combine_with(target, fill_value=0.0, mode=mode)

        # non-TransientAbsorption (scalar or ndarray)
        if not isinstance(other, TransientAbsorption):
            try:
                res: NDArray[np.floating] = np.asarray(np.asarray(self) + other)
                out: TransientAbsorption = res.view(TransientAbsorption)
                object.__setattr__(out, "x", copy.copy(self.x))
                object.__setattr__(out, "y", copy.copy(self.y))
                return out

            except Exception:
                return NotImplemented

        # both TransientAbsorption -> default CONCAT (order-independent due to grouping+averaging)
        return self.combine_with(other, fill_value=0.0, mode="concat")

    def __radd__(
            self: TransientAbsorption, 
            other: TransientAbsorption | _AddWithMode
        ) -> TransientAbsorption:

        if isinstance(other, _AddWithMode):
            target: TransientAbsorption = other.obj
            mode: str = other.mode
            if not isinstance(target, TransientAbsorption):
                return NotImplemented
            return target.combine_with(self, fill_value=0.0, mode=mode)
        return self.__add__(other)

    def __getitem__(
            self: TransientAbsorption, 
            key: tuple | int | slice | NDArray[np.integer] | NDArray[np.bool_]
        ) -> TransientAbsorption:
        """
        Indexing that slices data and coordinates consistently and robustly.
        Handles cases where self.x or self.y may be None (e.g. after reductions).
        Returns a TransientAbsorption view when possible; scalar results are
        wrapped as 0-D TransientAbsorption arrays.
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

            if coord is None:
                return None
            # Integer index -> scalar
            if isinstance(idx, (int, np.integer)):
                return coord[idx]
            # None or full slice -> whole coord
            if idx is None or (isinstance(idx, slice) and idx == slice(None)):
                return coord
            # fancy indexing / boolean mask / slice
            return np.asarray(coord[idx])

        # Initialize new_x/new_y before use
        new_x: NDArray[np.floating] | None = None
        new_y: NDArray[np.floating] | None = None

        ndim: int = out.ndim
        if ndim == 2:
            new_y = slice_coord(self.y, row_idx)
            new_x = slice_coord(self.x, col_idx)

        elif ndim == 1:
            # one axis survived; determine which
            if isinstance(key, tuple):
                # row_idx integer -> row collapsed -> x survives
                if isinstance(row_idx, (int, np.integer)):
                    new_x = slice_coord(self.x, col_idx)
                    new_y = None
                # col_idx integer -> col collapsed -> y survives
                elif isinstance(col_idx, (int, np.integer)):
                    new_y = slice_coord(self.y, row_idx)
                    new_x = None
                else:
                    # e.g., arr[1:3, :] -> row slice yields 1D with x surviving
                    if (isinstance(col_idx, slice) and col_idx == slice(None)):
                        new_x = slice_coord(self.x, col_idx)
                        new_y = None
                    else:
                        # fallback: prefer x if it exists, else y
                        new_x = slice_coord(self.x, col_idx) if self.x is not None else None
                        new_y = None
            else:
                # single index -> row indexing like arr[k] -> x survives
                new_x = slice_coord(self.x, slice(None))
                new_y = None

        elif ndim == 0:
            new_x = slice_coord(self.x, col_idx)
            new_y = slice_coord(self.y, row_idx)

        else:
            # higher dims unsupported; set coords to None
            new_x = None
            new_y = None

        # Attach coords (may be None)
        object.__setattr__(out, "x", new_x)
        object.__setattr__(out, "y", new_y)
        return out

    def __array_function__(
        self: TransientAbsorption,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> Any:

        if func not in TransientAbsorption._SUPPORTED_ARRAY_FUNCTIONS:
            return NotImplemented

        # convert TransientAbsorption in args to ndarray and remember TransientAbsorption inputs
        numeric_args: list = []
        TransientAbsorption_inputs: list = []
        for a in args:
            if isinstance(a, TransientAbsorption):
                numeric_args.append(np.asarray(a))
                TransientAbsorption_inputs.append(a)
            else:
                numeric_args.append(a)

        # convert any TransientAbsorption inside kwargs
        def _convert_kwval(v):
            if isinstance(v, TransientAbsorption):
                return np.asarray(v)
            if isinstance(v, (list, tuple)):
                t: type = type(v)
                return t(_convert_kwval(x) for x in v)
            if isinstance(v, dict):
                return {kk: _convert_kwval(vv) for kk, vv in v.items()}
            return v

        numeric_kwargs = {k: _convert_kwval(v) for k, v in (kwargs or {}).items()}

        # find axis (if not provided in kwargs, try positional)
        axis: int | tuple[int, ...] | None = kwargs.get("axis", None) if kwargs is not None else None
        if axis is None and len(args) >= 2:
            possible_axis: object = args[1]
            if isinstance(possible_axis, (int, tuple, list)):
                axis = possible_axis

        # call numpy implementation
        result: object = func(*numeric_args, **numeric_kwargs)

        # support functions that return tuples (e.g. returned=True)
        returned_tuple: bool = isinstance(result, tuple)
        results_list: list = list(result) if returned_tuple else [result]
        wrapped_results: list = []

        for res in results_list:
            # non-array scalar -> return as-is
            if not isinstance(res, np.ndarray):
                wrapped_results.append(res)
                continue

            # helper: normalize axis description to tuple of axes
            def _normalize_axis(
                    ax: int | tuple[int, ...] | list[int] | None,
                    ndim: int
                ) -> tuple[int, ...]:

                if ax is None:
                    return tuple(range(ndim))

                if isinstance(ax, (list, tuple)):
                    ax_t: tuple[int, ...] = tuple(((i + ndim) if i < 0 else i) for i in ax)
                    return ax_t

                ai: int = int(ax)
                if ai < 0:
                    ai = ai + ndim

                return (ai,)

            orig_ndim: int = self.ndim
            reduced_axes: tuple[int, ...] = _normalize_axis(axis, orig_ndim)
            surviving_axes: list[int] = [i for i in range(orig_ndim) if i not in reduced_axes]

            # helper: find a common coordinate for axis_idx among all TransientAbsorption inputs, otherwise None
            def _common_coord_for_axis(
                    axis_idx: int
                ) -> NDArray[np.floating] | None:

                if not TransientAbsorption_inputs:
                    return None

                first: NDArray[np.floating] | None = getattr(TransientAbsorption_inputs[0], "x" if axis_idx == 1 else "y", None)
                for m in TransientAbsorption_inputs[1:]:
                    other_coord = getattr(m, "x" if axis_idx == 1 else "y", None)
                    if first is None or other_coord is None:
                        return None

                    if not np.array_equal(np.asarray(first), np.asarray(other_coord)):
                        return None

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
                # single axis survived — map which axis it is
                if len(surviving_axes) == 1:
                    surv: int = surviving_axes[0]
                    if surv == 1:
                        cand_x: NDArray[np.floating] | None = _common_coord_for_axis(1)
                        new_x = np.asarray(cand_x) if cand_x is not None else None
                        new_y = None
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

            # For 1-D results, ensure coords are 1-D arrays (not scalars) when present.
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

    def __repr__(
            self: TransientAbsorption
        ) -> str:
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
        self: TransientAbsorption,
        *args,
        **kwargs,
    ) -> dict:
        from phoskhemia.fitting.global_fit import fit_global_kinetics
        return fit_global_kinetics(self, *args, **kwargs)



