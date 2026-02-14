import numpy as np
from numpy.typing import NDArray
from typing import Literal

def _nearest_index(axis: NDArray[np.floating], value: float) -> int:
    a = np.asarray(axis, dtype=float).reshape(-1)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("axis must be finite and non-empty")
    return int(np.argmin(np.abs(a - float(value))))

def _bracketing_indices(
        axis: NDArray[np.floating], 
        value: float
    ) -> tuple[int, int, float]:
    """
    Return (i0, i1, w) such that value is between axis[i0], axis[i1] and
    y(value) ≈ (1-w)*y[i0] + w*y[i1]. Requires strictly monotone axis.
    """

    a = np.asarray(axis, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError("axis must have at least 2 points for interpolation")
    if not (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0)):
        raise ValueError("axis must be strictly monotone for interp")
    v = float(value)

    # handle decreasing axes by working on increasing view
    if a[0] > a[-1]:
        a_inc = a[::-1]
        flipped = True
    else:
        a_inc = a
        flipped = False

    # clamp to range
    if v <= a_inc[0]:
        j0, j1, w = 0, 1, 0.0
    elif v >= a_inc[-1]:
        j0, j1, w = a_inc.size - 2, a_inc.size - 1, 1.0
    else:
        j1 = int(np.searchsorted(a_inc, v, side="right"))
        j0 = j1 - 1
        denom = (a_inc[j1] - a_inc[j0])
        w = 0.0 if denom == 0 else (v - a_inc[j0]) / denom

    if flipped:
        # map back to original indices
        n = a.size
        i0 = (n - 1) - j0
        i1 = (n - 1) - j1
    else:
        i0, i1 = j0, j1

    # ensure i0 < i1 for mixing convenience (swap if needed)
    if i0 > i1:
        i0, i1 = i1, i0
        w = 1.0 - w
    return i0, i1, float(w)
