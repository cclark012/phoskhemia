from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from phoskhemia.data.spectrum2d import TransientAbsorption
from phoskhemia.data.meta import MetaDict, meta_copy_update

def time_indices_uniform(n: int, *, stride: int) -> NDArray[np.int64]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return np.arange(0, n, stride, dtype=np.int64)

def time_indices_log(
        times: NDArray[np.floating],
        *,
        n_log: int,
        eps: float | None = None,
    ) -> NDArray[np.int64]:
    """Choose log-spaced indices with a default independent of time units.

    By default the logarithm uses elapsed time divided by the full window span
    and an offset of 1e-12 in those dimensionless coordinates. An explicit
    ``eps`` retains its historical meaning as an offset in the supplied time
    units, so it must be converted along with the time axis.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    if n_log < 2:
        raise ValueError("n_log must be >= 2")
    if t.size == 0 or not np.isfinite(t).all():
        raise ValueError("times must be finite and nonempty")
    if eps is not None and (not np.isfinite(eps) or eps <= 0):
        raise ValueError("eps must be finite and positive")
    if t.size == 1:
        return np.array([0], dtype=np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")

    elapsed = t - t[0]
    span = elapsed[-1]
    offset = 1e-12 if eps is None else float(eps) / span
    u = np.log(elapsed / span + offset)
    grid = np.linspace(float(u[0]), float(u[-1]), n_log)
    idx = np.searchsorted(u, grid, side="left")
    idx = np.minimum(idx, t.size - 1)
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
    tail_idx: NDArray[np.int64] = time_indices_log(t[tail], n_log=n_log)
    return np.unique(np.r_[dense, tail[tail_idx]])

def downsample_time(
        arr: TransientAbsorption, 
        idx: NDArray[np.int64]
    ) -> TransientAbsorption:

    idx: NDArray[np.int64] = np.asarray(idx, dtype=np.int64)
    result = arr[idx, :]
    result.meta["downsample_idx_len"] = int(idx.size)
    return result

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
        indices = time_indices_uniform(len(t), **kwargs)

    else:
        raise ValueError("method must be one of 'log', 'hybrid', or 'linear'")

    return indices

def downsample_time_binned(
        arr: TransientAbsorption,
        indices: NDArray[np.int64],
        *,
        time_stat: Literal['mean', 'median', 'anchor'] = "mean",      # "mean" or "anchor"
        data_stat: Literal['mean', 'median', 'min', 'max'] = "mean",      # currently only mean
    ) -> TransientAbsorption:

    """Aggregate midpoint bins and update measurement-noise metadata.

    ``noise_t0`` denotes a standard deviation constant along time. For means
    with equal bin counts it is scaled assuming independent observations. An
    aggregate requiring time-dependent uncertainty, or a nonlinear statistic
    without a noise model, clears that key and records the reason. The actual
    sample counts are available in ``meta["downsample_bin_counts"]``.
    """
    data: NDArray[np.floating] = np.asarray(arr, dtype=float)
    t: NDArray[np.floating] = np.asarray(arr.y, dtype=float).reshape(-1)

    idx: NDArray[np.int64] = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1 or idx.size < 2:
        raise ValueError("indices must be 1D with at least 2 entries")
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != t.size - 1:
        idx = np.r_[idx, t.size - 1]

    a: NDArray[np.floating] = t[idx]
    if np.any(np.diff(a) <= 0):
        raise ValueError("anchor times must be strictly increasing (check indices and time ordering)")

    # Make bin edges; values will be aggregated between bin edges.
    n_bins: int = a.size
    edges: NDArray[np.floating] = np.empty(n_bins + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = 0.5 * (a[:-1] + a[1:])

    # Number of values (counts) between each downsampled time value (bins).
    bin_id: NDArray[np.int64] = np.searchsorted(edges, t, side="right") - 1  # shape (n_times,)
    counts: NDArray[np.floating] = np.bincount(bin_id, minlength=n_bins).astype(float)
    if np.any(counts == 0):
        # Should not happen with the +/- inf edges, but guard anyway
        raise RuntimeError("Empty bin encountered")

    starts: NDArray[np.bool] = np.flatnonzero(np.r_[True, bin_id[1:] != bin_id[:-1]])
    stops: NDArray[np.bool] = np.r_[starts[1:], bin_id.size]
    n_bins: int = starts.size
    assert len(starts) == len(stops), "Number of start and stop indices does not match"
    assert np.all(stops > starts), "Not all stop indices are greater than their start counterparts"

    # Time aggregation, new time values are either 
    # average of each bin or the first time in each bin.
    if time_stat == "mean":
        t_sum: NDArray[np.floating] = np.bincount(bin_id, weights=t, minlength=n_bins)
        t_new: NDArray[np.floating] = (t_sum / counts).astype(float)
    elif time_stat == "median":
        t_new = np.array([np.median(t[s:e]) for s, e in zip(starts, stops)], dtype=float)
    elif time_stat == "anchor":
        t_new: NDArray[np.floating] = a.astype(float)
    else:
        raise ValueError("time_stat must be 'mean', 'median', or 'anchor'")

    # Data aggregation (loop wavelengths; n_wl small)
    n_wl: int = data.shape[1]
    data_new: NDArray[np.floating] = np.empty((n_bins, n_wl), dtype=float)
    if data_stat == 'mean':
        for j in range(n_wl):
            s: NDArray[np.floating] = np.bincount(bin_id, weights=data[:, j], minlength=n_bins)
            data_new[:, j] = s / counts

    elif data_stat == "median":
        for j in range(n_wl):
            col: NDArray[np.floating] = data[:, j]
            data_new[:, j] = [np.median(col[s:e]) for s, e in zip(starts, stops)]

    elif data_stat == "max":
        for j in range(n_wl):
            col: NDArray[np.floating] = data[:, j]
            data_new[:, j] = [np.max(col[s:e]) for s, e in zip(starts, stops)]

    elif data_stat == "min":
        for j in range(n_wl):
            col: NDArray[np.floating] = data[:, j]
            data_new[:, j] = [np.min(col[s:e]) for s, e in zip(starts, stops)]
    else:
        raise ValueError("data_stat must be one of 'mean', 'median', 'min', or 'max'")

    meta: MetaDict = meta_copy_update(
        getattr(arr, "meta", None),
        {
        "downsample_method": "binned",
        "downsample_bins": int(n_bins),
        "downsample_time_stat": time_stat,
        "downsample_data_stat": data_stat,
        "downsample_bin_counts": counts.astype(np.int64),
    })
    noise = arr.meta.get("noise_t0")
    if noise is not None:
        sigma = np.asarray(noise, dtype=float)
        constant_time_shape = sigma.ndim == 0 or sigma.shape == (n_wl,)
        if data_stat == 'mean' and constant_time_shape and np.all(counts == counts[0]):
            scaled = sigma / np.sqrt(counts[0])
            meta["noise_t0"] = float(scaled) if scaled.ndim == 0 else scaled
            meta["downsample_noise_note"] = "Independent observations; actual equal bin counts used"
        else:
            meta.pop("noise_t0", None)
            if data_stat != 'mean':
                reason = "Noise propagation for nonlinear aggregation statistics is not implemented"
            else:
                reason = "Aggregation requires time-dependent noise; constant-time noise_t0 was cleared"
            meta["downsample_noise_note"] = reason
    return TransientAbsorption(data_new, x=arr.x, y=t_new, meta=meta)
