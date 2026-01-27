import numpy as np
import copy
from scipy.signal import convolve

# my_array = np.array(
#     [[1, 1, 1, 1, 1],
#     [2, 2, 2, 2, 2],
#     [3, 3, 3, 3, 3],
#     [2, 2, 2, 2, 2],
#     [1, 1, 1, 1, 1]], dtype=float)
# arr = MyArray(my_array, x=np.linspace(0,4,5), y=np.linspace(0,4,5))
# a = arr.smooth(
#     window=(5, 5),
#     mode="same",
# )
# print(a)



class _AddWithMode:
    def __init__(self, obj, mode):
        self.obj = obj
        self.mode = mode

def with_mode(obj, mode):
    return _AddWithMode(obj, mode)


class MyArray(np.ndarray):
    __array_priority__ = 1000.0

    _SUPPORTED_ARRAY_FUNCTIONS = {
        np.pow,
        np.mean,
        np.sum,
        np.average,
        np.nanmean,
        np.nanmedian,
        np.nanstd,
        np.nanvar,
    }


    def __new__(cls, input_array, x=None, y=None, dtype=float):
        arr = np.asarray(input_array, dtype=dtype)

        # If 1-D, decide whether user meant a row (1 x N) or a column (N x 1).
        if arr.ndim == 1:
            n = arr.size
            # If x provided and its length matches n -> treat as row (1 x n)
            if x is not None and len(np.asarray(x)) == n:
                arr = arr.reshape(1, n)
            # Else if y provided and its length matches n -> treat as column (n x 1)
            elif y is not None and len(np.asarray(y)) == n:
                arr = arr.reshape(n, 1)
            else:
                # default: treat as a single row 1 x n
                arr = arr.reshape(1, -1)

        # Now arr must be 2-D
        if arr.ndim != 2:
            raise ValueError("MyArray requires a 2-D array (or 1-D which will become 1xN or Nx1).")

        obj = arr.view(cls)
        nrows, ncols = obj.shape

        # Create default coords when missing
        if x is None:
            x = np.arange(ncols, dtype=float)
        if y is None:
            y = np.arange(nrows, dtype=float)

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-D arrays")

        if x.shape[0] != ncols:
            raise ValueError(f"length of x ({x.shape[0]}) must equal number of columns ({ncols})")
        if y.shape[0] != nrows:
            raise ValueError(f"length of y ({y.shape[0]}) must equal number of rows ({nrows})")

        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        object.__setattr__(self, "x", copy.copy(getattr(obj, "x", None)))
        object.__setattr__(self, "y", copy.copy(getattr(obj, "y", None)))

    def smooth(
        self,
        window,
        *,
        normalize=True,
        separable_tol=1e-10,
        **kwargs,
    ):
        """
        Smooth data using scipy.signal.convolve.

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
        **convolve_kwargs
            Passed directly to scipy.signal.convolve
            (e.g. mode='same', method='auto')
        """
        kwargs = {"mode": "same", "method": "auto"} | kwargs

        data = np.asarray(self, dtype=float)

        # -------------------------------------------------
        # Build kernel
        # -------------------------------------------------
        if isinstance(window, int):
            if window <= 0:
                raise ValueError("window length must be positive")
            kernel = np.ones((1, window), dtype=float)

        elif isinstance(window, tuple | list):
            if len(window) != 2:
                raise ValueError("window tuple must be (ny, nx)")
            kernel = np.ones(window, dtype=float)

        else:
            kernel = np.asarray(window, dtype=float)
            if kernel.ndim == 1:
                kernel = kernel.reshape(1, -1)
            elif kernel.ndim != 2:
                raise ValueError("window must be int, 1-D, or 2-D")

        ny, nx = kernel.shape

        if normalize:
            s = kernel.sum()
            if s != 0:
                kernel = kernel / s

        # -------------------------------------------------
        # 1-D smoothing along x
        # -------------------------------------------------
        if ny == 1 and nx > 1:
            h = kernel.ravel()
            out = convolve(data, h[None, :], **kwargs)

        # -------------------------------------------------
        # 1-D smoothing along y
        # -------------------------------------------------
        elif ny > 1 and nx == 1:
            v = kernel.ravel()
            out = convolve(data, v[:, None], **kwargs)

        # -------------------------------------------------
        # 2-D smoothing
        # -------------------------------------------------
        else:
            # Attempt separable acceleration
            U, S, Vt = np.linalg.svd(kernel, full_matrices=False)
            separable = S.size > 1 and S[1] / S[0] < separable_tol

            if separable:
                v = U[:, 0] * np.sqrt(S[0])
                h = Vt[0, :] * np.sqrt(S[0])

                tmp = convolve(data, v[:, None], **kwargs)
                out = convolve(tmp, h[None, :], **kwargs)

            else:
                out = convolve(data, kernel, **kwargs)

        result = out.view(MyArray)
        result.x = self.x
        result.y = self.y

        return result

    @staticmethod
    def _convolve_loop(out_shape, ny, nx, compute_fn):
        """
        Canonical convolution loop over output indices.

        compute_fn(ip, jp) must return the convolution value
        for output index (ip, jp).
        """
        H, W = out_shape
        out = np.zeros((H, W), dtype=float)

        for ip in range(H):
            for jp in range(W):
                out[ip, jp] = compute_fn(ip, jp)

        return out

    @staticmethod
    def _infer_spacing(axis):
        axis = np.asarray(axis, dtype=float)
        if axis.size < 2:
            return None
        diffs = np.diff(axis)
        return np.mean(diffs)

    @staticmethod
    def _continuous_axis(a, b, tol=1e-8):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dx_a = MyArray._infer_spacing(a)
        dx_b = MyArray._infer_spacing(b)

        if dx_a is not None and dx_b is not None:
            if not np.isclose(dx_a, dx_b, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Axis spacings differ: {dx_a} vs {dx_b}")
            dx = 0.5 * (dx_a + dx_b)
        else:
            dx = dx_a if dx_a is not None else dx_b

        start = min(a[0], b[0])
        stop = max(a[-1], b[-1])

        if dx is None:
            if np.isclose(start, stop, atol=tol):
                return np.array([start], dtype=float), None
            else:
                axis = np.array([start, stop], dtype=float)
                return axis, stop - start

        n = int(round((stop - start) / dx)) + 1
        axis = start + dx * np.arange(n)
        if not np.isclose(axis[-1], stop, rtol=1e-8, atol=1e-12):
            axis = np.append(axis, stop)
        return axis, dx

    def _group_and_average_columns(self, cols):
        """
        cols: list of tuples (x_value(float), column_tuple_of_floats)
        returns (x_new_array, vals_matrix) with duplicated x's averaged.
        """
        if not cols:
            return np.array([], dtype=float), np.zeros((0,0), dtype=float)

        # sort by x then by column values (deterministic)
        cols_sorted = sorted(cols, key=lambda t: (t[0], t[1]))

        # group consecutive entries with same x (isclose)
        xs = [c[0] for c in cols_sorted]
        cols_vals = [np.array(c[1], dtype=float) for c in cols_sorted]

        groups = []
        current_group_idx = 0
        current_group = [cols_vals[0]]
        current_xs = [xs[0]]

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
        averaged_cols = []
        averaged_xs = []
        for x_group, col_group in groups:
            # stack columns (shape (n_members, nrows)) -> average along axis=0 yields (nrows,)
            stacked = np.vstack(col_group)  # shape (n_members, nrows)
            mean_col = np.mean(stacked, axis=0)
            mean_x = float(np.mean(x_group))
            averaged_xs.append(mean_x)
            averaged_cols.append(mean_col)

        vals = np.column_stack(averaged_cols) if averaged_cols else np.zeros((stacked.shape[1], 0))
        return np.array(averaged_xs, dtype=float), vals

    def _group_and_average_rows(self, rows):
        """
        rows: list of tuples (y_value(float), row_tuple_of_floats)
        returns (y_new_array, vals_matrix) with duplicated y's averaged.
        """
        if not rows:
            return np.array([], dtype=float), np.zeros((0,0), dtype=float)

        rows_sorted = sorted(rows, key=lambda t: (t[0], t[1]))

        ys = [r[0] for r in rows_sorted]
        rows_vals = [np.array(r[1], dtype=float) for r in rows_sorted]

        groups = []
        current_group = [rows_vals[0]]
        current_ys = [ys[0]]

        for yi, ri in zip(ys[1:], rows_vals[1:]):
            if np.isclose(yi, current_ys[-1], atol=1e-8, rtol=1e-6):
                current_group.append(ri)
                current_ys.append(yi)
            else:
                groups.append((current_ys, current_group))
                current_group = [ri]
                current_ys = [yi]
        groups.append((current_ys, current_group))

        averaged_rows = []
        averaged_ys = []
        for y_group, row_group in groups:
            stacked = np.vstack(row_group)  # shape (n_members, ncols)
            mean_row = np.mean(stacked, axis=0)
            mean_y = float(np.mean(y_group))
            averaged_ys.append(mean_y)
            averaged_rows.append(mean_row)

        vals = np.vstack(averaged_rows) if averaged_rows else np.zeros((0, stacked.shape[1]))
        return np.array(averaged_ys, dtype=float), vals

    def combine_with(self, other, fill_value=0.0, mode="average"):
        if not isinstance(other, MyArray):
            raise TypeError("combine_with expects another MyArray")

        if mode not in ("average", "concat"):
            raise ValueError("mode must be 'average' or 'concat'")

        if mode == "concat":
            # concat along x if y match exactly — build columns and average duplicates by x
            if np.array_equal(self.y, other.y):
                cols = []
                for x_val, col in zip(self.x, np.asarray(self).T):
                    cols.append((float(x_val), tuple(map(float, col))))
                for x_val, col in zip(other.x, np.asarray(other).T):
                    cols.append((float(x_val), tuple(map(float, col))))
                x_new, vals = self._group_and_average_columns(cols)
                return MyArray(vals, x=x_new, y=copy.copy(self.y))

            # concat along y if x match exactly — build rows and average duplicates by y
            if np.array_equal(self.x, other.x):
                rows = []
                for y_val, row in zip(self.y, np.asarray(self)):
                    rows.append((float(y_val), tuple(map(float, row))))
                for y_val, row in zip(other.y, np.asarray(other)):
                    rows.append((float(y_val), tuple(map(float, row))))
                y_new, vals = self._group_and_average_rows(rows)
                return MyArray(vals, x=copy.copy(self.x), y=y_new)

            # else refuse
            raise ValueError(
                "concat mode requires exact match on the orthogonal axis: "
                "either self.y == other.y (concat columns) or self.x == other.x (concat rows)."
            )

        # ---------- average mode (unchanged) ----------
        x_new, dx = self._continuous_axis(self.x, other.x)
        y_new, dy = self._continuous_axis(self.y, other.y)

        new_vals_sum = np.full((len(y_new), len(x_new)), 0.0, dtype=float)
        new_counts = np.zeros_like(new_vals_sum, dtype=float)

        def find_index(axis, val):
            idx = np.flatnonzero(np.isclose(axis, val, atol=1e-8, rtol=1e-6))
            if idx.size == 0:
                idx = np.array([int(np.argmin(np.abs(axis - val)))])
            return int(idx[0])

        def place_into(acc_sum, acc_count, src, src_x, src_y):
            x0 = find_index(x_new, src_x[0])
            y0 = find_index(y_new, src_y[0])
            r0, r1 = y0, y0 + src.shape[0]
            c0, c1 = x0, x0 + src.shape[1]
            acc_sum[r0:r1, c0:c1] += np.asarray(src, dtype=float)
            acc_count[r0:r1, c0:c1] += 1.0

        place_into(new_vals_sum, new_counts, self, self.x, self.y)
        place_into(new_vals_sum, new_counts, other, other.x, other.y)

        with np.errstate(invalid='ignore', divide='ignore'):
            averaged = np.where(new_counts > 0, new_vals_sum / new_counts, fill_value)

        return MyArray(averaged, x=x_new, y=y_new)

    def add(self, other, mode="average", fill_value=0.0):
        if mode in ("average", "concat"):
            return self.combine_with(other, fill_value=fill_value, mode=mode)
        raise ValueError("mode must be 'average' or 'concat'")

    def __add__(self, other):
        # wrapper carrying mode?
        if isinstance(other, _AddWithMode):
            target = other.obj
            mode = other.mode
            if not isinstance(target, MyArray):
                return NotImplemented
            return self.combine_with(target, fill_value=0.0, mode=mode)

        # non-MyArray (scalar or ndarray)
        if not isinstance(other, MyArray):
            try:
                res = np.asarray(np.asarray(self) + other)
                out = res.view(MyArray)
                object.__setattr__(out, "x", copy.copy(self.x))
                object.__setattr__(out, "y", copy.copy(self.y))
                return out
            except Exception:
                return NotImplemented

        # both MyArray -> default CONCAT (order-independent due to grouping+averaging)
        return self.combine_with(other, fill_value=0.0, mode="concat")

    def __radd__(self, other):
        if isinstance(other, _AddWithMode):
            target = other.obj
            mode = other.mode
            if not isinstance(target, MyArray):
                return NotImplemented
            return target.combine_with(self, fill_value=0.0, mode=mode)
        return self.__add__(other)

    def __getitem__(self, key):
        """
        Indexing that slices data and coordinates consistently and robustly.
        Handles cases where self.x or self.y may be None (e.g. after reductions).
        """
        # Normalize key into (row_idx, col_idx)
        if isinstance(key, tuple):
            if len(key) == 2:
                row_idx, col_idx = key
            elif len(key) == 1:
                row_idx, col_idx = key[0], slice(None)
            else:
                raise IndexError("Too many indices for MyArray")
        else:
            row_idx, col_idx = key, slice(None)

        # Perform numeric indexing
        result = super().__getitem__(key)

        # Convert scalars to 0-D array so subclass is preserved
        if not isinstance(result, np.ndarray):
            result = np.array(result)

        out = result.view(MyArray)

        # Helper to safely slice coordinate arrays (returns None if coord is None)
        def slice_coord(coord, idx):
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
        new_x = None
        new_y = None

        ndim = out.ndim
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

    def __array_function__(self, func, types, args, kwargs):
        if func not in MyArray._SUPPORTED_ARRAY_FUNCTIONS:
            return NotImplemented

        # convert MyArray in args to ndarray and remember MyArray inputs
        numeric_args = []
        myarray_inputs = []
        for a in args:
            if isinstance(a, MyArray):
                numeric_args.append(np.asarray(a))
                myarray_inputs.append(a)
            else:
                numeric_args.append(a)

        # convert any MyArray inside kwargs
        def _convert_kwval(v):
            if isinstance(v, MyArray):
                return np.asarray(v)
            if isinstance(v, (list, tuple)):
                t = type(v)
                return t(_convert_kwval(x) for x in v)
            if isinstance(v, dict):
                return {kk: _convert_kwval(vv) for kk, vv in v.items()}
            return v

        numeric_kwargs = {k: _convert_kwval(v) for k, v in (kwargs or {}).items()}

        # find axis (if not provided in kwargs, try positional)
        axis = kwargs.get("axis", None) if kwargs is not None else None
        if axis is None and len(args) >= 2:
            possible_axis = args[1]
            if isinstance(possible_axis, (int, tuple, list)):
                axis = possible_axis

        # call numpy implementation
        result = func(*numeric_args, **numeric_kwargs)

        # support functions that return tuples (e.g. returned=True)
        returned_tuple = isinstance(result, tuple)
        results_list = list(result) if returned_tuple else [result]
        wrapped_results = []

        for res in results_list:
            # non-array scalar -> return as-is
            if not isinstance(res, np.ndarray):
                wrapped_results.append(res)
                continue

            # helper: normalize axis description to tuple of axes
            def _normalize_axis(ax, ndim):
                if ax is None:
                    return tuple(range(ndim))
                if isinstance(ax, (list, tuple)):
                    ax_t = tuple(((i + ndim) if i < 0 else i) for i in ax)
                    return ax_t
                ai = int(ax)
                if ai < 0:
                    ai = ai + ndim
                return (ai,)

            orig_ndim = self.ndim
            reduced_axes = _normalize_axis(axis, orig_ndim)
            surviving_axes = [i for i in range(orig_ndim) if i not in reduced_axes]

            # helper: find a common coordinate for axis_idx among all MyArray inputs, otherwise None
            def _common_coord_for_axis(axis_idx):
                if not myarray_inputs:
                    return None
                first = getattr(myarray_inputs[0], "x" if axis_idx == 1 else "y", None)
                for m in myarray_inputs[1:]:
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
            new_x = None
            new_y = None

            if res.ndim == 2:
                cand_x = _common_coord_for_axis(1)
                cand_y = _common_coord_for_axis(0)
                new_x = np.asarray(cand_x) if cand_x is not None else None
                new_y = np.asarray(cand_y) if cand_y is not None else None

            elif res.ndim == 1:
                # single axis survived — map which axis it is
                if len(surviving_axes) == 1:
                    surv = surviving_axes[0]
                    if surv == 1:
                        cand_x = _common_coord_for_axis(1)
                        new_x = np.asarray(cand_x) if cand_x is not None else None
                        new_y = None
                    else:
                        cand_y = _common_coord_for_axis(0)
                        new_y = np.asarray(cand_y) if cand_y is not None else None
                        new_x = None
                else:
                    # ambiguous -> no coords
                    new_x = None
                    new_y = None

            elif res.ndim == 0:
                new_x = None
                new_y = None

            # Now wrap result as MyArray and attach coords (safe types)
            out = res.view(MyArray)

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

    def __repr__(self):
        base = super().__repr__()
        # present coords cleanly and avoid indexing None or scalar coords
        def coord_repr(c):
            if c is None:
                return "None"
            # convert 0-d numpy scalars into Python scalars for nicer display
            arr = np.asarray(c)
            if arr.ndim == 0:
                return repr(arr.item())
            return repr(arr)
        return f"{base}\nmeta: x={coord_repr(getattr(self, 'x', None))}, y={coord_repr(getattr(self, 'y', None))}"

    def fit_global_kinetics(
        self: MyArray,
        *args,
        **kwargs,
    ) -> dict:
        from phoskhemia.fitting.global_fit import fit_global_kinetics
        return fit_global_kinetics(self, *args, **kwargs)



