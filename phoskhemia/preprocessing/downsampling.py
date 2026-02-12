from __future__ import annotations
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray

from phoskhemia.data.spectrum_handlers import TransientAbsorption

def time_indices_uniform(n: int, *, stride: int) -> NDArray[np.int64]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return np.arange(0, n, stride, dtype=np.int64)

def time_indices_log(
        times: NDArray[np.floating], 
        *, 
        n_keep: int, 
        eps: float = 1e-12
    ) -> NDArray[np.int64]:

    t: NDArray[np.floating] = np.asarray(times, dtype=float).reshape(-1)
    if n_keep < 2:
        raise ValueError("n_keep must be >= 2")

    t0: float = t.min()
    # Shift so domain is positive for log, then convert t -> u
    u: NDArray[np.floating] = np.log((t - t0) + eps)
    umin: float
    umax: float
    umin, umax = float(u[0]), float(u[-1])
    grid: NDArray[np.floating] = np.linspace(umin, umax, n_keep)
    idx: NDArray[np.integer] = np.unique(np.searchsorted(u, grid, side="left"))
    idx[idx >= t.size] = t.size - 1
    return np.unique(np.r_[0, idx, t.size - 1]).astype(np.int64)

def time_indices_hybrid(
        times: NDArray[np.floating], 
        *, 
        t_dense_max: float, 
        n_log: int
    ) -> NDArray[np.int64]:

    t: NDArray[np.floating] = np.asarray(times, dtype=float).reshape(-1)
    dense: NDArray[np.int64] = np.where(t <= t_dense_max)[0].astype(np.int64)
    tail: NDArray[np.int64] = np.where(t > t_dense_max)[0].astype(np.int64)
    if tail.size == 0:
        return dense
    tail_idx: NDArray[np.int64] = time_indices_log(t[tail], n_keep=n_log)
    return np.unique(np.r_[dense, tail[tail_idx]])

def downsample_time(
        arr: TransientAbsorption, 
        idx: NDArray[np.int64]
    ) -> TransientAbsorption:

    idx: NDArray[np.int64] = np.asarray(idx, dtype=np.int64)
    data: NDArray[np.floating] = np.asarray(arr)[idx, :]
    y: NDArray[np.floating] = np.asarray(arr.y, dtype=float)[idx]
    meta: dict[str, Any] = getattr(arr, "meta", {}).copy()
    meta.update({"downsample_idx_len": int(idx.size)})
    return TransientAbsorption(data, x=arr.x, y=y, meta=meta)

def make_time_indices(
        times: NDArray[np.floating],
        *,
        method: Literal['log', 'hybrid', 'linear'] = 'log',
        **kwargs,
    ) -> NDArray[np.int64]:

    t = np.asarray(times)
    if method == 'log':
        indices = time_indices_log(t, **kwargs)

    elif method == 'hybrid':
        indices = time_indices_hybrid(t, **kwargs)

    elif method == 'linear':
        raise NotImplementedError

    else:
        raise ValueError("method must be one of 'log', 'hybrid', or 'linear'")

    return indices
