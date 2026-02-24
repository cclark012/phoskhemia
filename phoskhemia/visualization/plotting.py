from __future__ import annotations

from typing import Literal, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from matplotlib import rcParams, ticker
import numpy as np
from numpy.typing import NDArray
from phoskhemia.data.spectrum_handlers import TransientAbsorption
from phoskhemia.fitting.results import GlobalFitResult
from phoskhemia.fitting.reconstructions import reconstruct_fit

def _decimate_time(
        arr: NDArray[np.floating], 
        times: NDArray[np.floating], 
        max_points: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    # TODO - Add downsampling based on time_scale argument.
    n_t: int
    n_w: int
    n_t, n_w = arr.shape
    if arr.size <= max_points:
        return arr, times

    n_t_keep: int = int(max(2, int(max_points // n_w)))
    idx: NDArray[np.int64] = np.linspace(0, n_t - 1, n_t_keep, dtype=np.int64)
    idx = np.unique(idx)

    return arr[idx, :], times[idx]

def _edges(x: NDArray[np.floating]) -> NDArray[np.floating]:

    x: NDArray[np.floating] = np.asarray(x, dtype=float)
    dx: NDArray[np.floating] = np.diff(x)
    if x.size < 2:
        return np.array([x[0] - 0.5, x[0] + 0.5])

    edges: NDArray[np.floating] = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges

def plot_ta(
    ta: TransientAbsorption,
    *,
    ax: Axes | None = None,
    time_scale: Literal['log', 'linear', 'symlog'] = 'log',
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap | None = None,
    max_points: int = 2_000_000,
    fast: bool = True,
    return_mappable: bool = False,
    **kwargs: Any
    ) -> Axes | tuple[Axes, QuadContourSet]:

    if ax is None:
        _, ax = plt.subplots()

    if cmap is None:
        cmap = rcParams["image.cmap"]
    
    arr: NDArray[np.floating] = np.asarray(ta)
    times: NDArray[np.floating] = np.asarray(ta.y).reshape(-1)
    waves: NDArray[np.floating] = np.asarray(ta.x).reshape(-1)
    arr, times = _decimate_time(arr=arr, times=times, max_points=max_points)
    
    if np.any(times <= 0) and time_scale == 'log':
        raise ValueError("time_scale='log' requires all times > 0. Use 'symlog' or truncate/shift time.")
    
    if fast:
        X = _edges(waves)
        Y = _edges(times)
        plot: Axes = ax.pcolormesh(X, Y, arr, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    else:
        plot: QuadContourSet = ax.contourf(waves, times, arr, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if time_scale == "log":
        ax.set_yscale('log')
        ax.set_ylim((np.min(np.abs(times[times > 0])), np.max(times)))
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(10 ** y)))

    elif time_scale == "symlog":
        ax.set_yscale('symlog')
    
    elif time_scale != 'linear':
        raise ValueError("time_scale must be 'log', 'symlog', or 'linear'")

    if return_mappable:
        return ax, plot

    else:
        return ax

def plot_residuals(
        ta: TransientAbsorption,
        result: GlobalFitResult | TransientAbsorption,
        *,
        ax: Axes | None = None,
        time_scale: Literal['log', 'linear', 'symlog'] = 'log',
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | Colormap | None = None,
        max_points: int = 2_000_000,
        fast: bool = True,
        return_mappable: bool = False,
        **kwargs
    ) -> Axes | tuple[Axes, QuadContourSet]:

    if isinstance(result, GlobalFitResult):
        fit: NDArray[np.floating] = np.asarray(reconstruct_fit(result), dtype=float)

    elif isinstance(result, TransientAbsorption):
        fit: NDArray[np.floating] = np.asarray(result, dtype=float)

    else:
        raise ValueError("result must a GlobalFitResult or TransientAbsorption instance")

    arr: NDArray[np.floating] = np.asarray(ta)
    if arr.shape != fit.shape:
        raise ValueError(f"fit shape {fit.shape} does not match data shape {arr.shape}")

    times: NDArray[np.floating] = np.asarray(ta.y).reshape(-1)
    waves: NDArray[np.floating] = np.asarray(ta.x).reshape(-1)
    resids: NDArray[np.floating] = arr - fit
    residuals = TransientAbsorption(resids, waves, times)

    return plot_ta(
            residuals, 
            ax=ax, 
            time_scale=time_scale, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            max_points=max_points, 
            fast=fast, 
            return_mappable=return_mappable, 
            **kwargs
        )

def plot_fit(
        result: GlobalFitResult | TransientAbsorption,
        *,
        ax: Axes | None = None,
        time_scale: Literal['log', 'linear', 'symlog'] = 'log',
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | Colormap | None = None,
        max_points: int = 2_000_000,
        fast: bool = True,
        return_mappable: bool = False,
        **kwargs
    ) -> Axes | tuple[Axes, QuadContourSet]:

    if isinstance(result, GlobalFitResult):
        fit: NDArray[np.floating] = np.asarray(reconstruct_fit(result), dtype=float)

    elif isinstance(result, TransientAbsorption):
        fit: NDArray[np.floating] = np.asarray(result, dtype=float)

    else:
        raise ValueError("result must a GlobalFitResult or TransientAbsorption instance")

    return plot_ta(
            fit, 
            ax=ax, 
            time_scale=time_scale, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            max_points=max_points, 
            fast=fast, 
            return_mappable=return_mappable, 
            **kwargs
        )

def plot_trace(
        ta: TransientAbsorption,
        wavelength: int | float,
        *,
        time_scale: Literal['log', 'linear', 'symlog'] = 'log',
        ax: Axes | None = None,
        method: Literal['nearest', 'interp'] = 'nearest',
        **kwargs,
    ) -> Axes:

    if ax is None:
        _, ax = plt.subplots()
    
    trace: NDArray[np.floating] = ta.trace(wavelength, method=method)

    ax.plot(ta.y, trace, **kwargs)

    if time_scale == 'log':
        ax.set_xscale('log')

    return ax

def plot_spectrum(
        ta: TransientAbsorption,
        time: float | int,
        *,
        ax: Axes | None = None,
        method: Literal['nearest', 'interp'] = 'nearest',
        aggregate: int = 0,
        **kwargs
    ) -> Axes:
    
    if ax is None:
        fig, ax = plt.subplots()

    spectrum: NDArray[np.floating] = ta.spectrum(time, method=method, aggregate=aggregate)
    ax.plot(ta.x, spectrum, **kwargs)
    
    return ax

if __name__ == "__main__":
    rng = np.random.default_rng()
    array = rng.normal(0, 1, (123456, 100))
    times = np.arange(1, array.shape[0]+1)
    waves = np.arange(1, array.shape[1]+1)
    array2 = rng.normal(0, 1, (123456, 100))
    arr = TransientAbsorption(array, waves, times)
    fit = TransientAbsorption(array2, waves, times)
    plot = plot_ta(arr)
    plt.show()
