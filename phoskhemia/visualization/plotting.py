from __future__ import annotations

from typing import Literal, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from matplotlib import rcParams, ticker
import numpy as np
from numpy.typing import NDArray
from phoskhemia.data.spectrum_handlers import TransientAbsorption
from phoskhemia.fitting.results import GlobalFitResult
from phoskhemia.fitting.reconstructions import reconstruct_fit

def plot_ta(
    ta: TransientAbsorption,
    *,
    ax: Axes | None = None,
    time_scale: Literal['log', 'linear', 'symlog'] = 'log',
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap | None = None,
    max_points: int = 2_000_000,
    **kwargs: Any
    ) -> Axes:

    if ax is None:
        fig, ax = plt.subplots()

    if cmap is None:
        cmap = rcParams["image.cmap"]
    
    arr: NDArray[np.floating] = np.asarray(ta)
    times: NDArray[np.floating] = np.asarray(ta.y).reshape(-1)
    waves: NDArray[np.floating] = np.asarray(ta.x).reshape(-1)
    if arr.size > max_points:
        # TODO - Add downsampling based on time_scale argument.
        n_times: int = int(max_points // ta.x.size)
        idx: NDArray[np.int64] = np.linspace(0, ta.y.size, n_times, dtype=np.int64, endpoint=False)
        times = times[idx]
        arr = arr[idx, :]
    
    if time_scale == "log":
        times = np.log10(times, dtype=float)
    
    plot: Axes = ax.contourf(waves, times, arr, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if time_scale == "log":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(10 ** y)))

    return plot

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
        **kwargs
    ) -> Axes:

    if ax is None:
        fig, ax = plt.subplots()

    if cmap is None:
        cmap = rcParams["image.cmap"]
    
    if isinstance(result, GlobalFitResult):
        fit_ta: TransientAbsorption = reconstruct_fit(result)
        fit: NDArray[np.floating] = np.asarray(fit_ta)

    elif isinstance(result, TransientAbsorption):
        fit: NDArray[np.floating] = np.asarray(result)

    else:
        raise ValueError("result must be either a GlobalFitResult or TransientAbsorption object")

    arr: NDArray[np.floating] = np.asarray(ta)
    times: NDArray[np.floating] = np.asarray(ta.y).reshape(-1)
    waves: NDArray[np.floating] = np.asarray(ta.x).reshape(-1)
    resids: NDArray[np.floating] = arr - fit
    if resids.size > max_points:
        # TODO - Add downsampling based on time_scale argument.
        n_times: int = int(max_points // ta.x.size)
        idx: NDArray[np.int64] = np.linspace(0, ta.y.size, n_times, dtype=np.int64, endpoint=False)
        times = times[idx]
        resids = resids[idx, :]
    
    if time_scale == "log":
        times = np.log10(times, dtype=float)
    
    plot: Axes = ax.contourf(waves, times, resids, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if time_scale == "log":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(10 ** y)))

    return plot

def plot_trace(
        ta: TransientAbsorption,
        wavelength: int | float,
        *,
        time_scale: Literal['log', 'linear', 'symlog'] = 'log',
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:

    if ax is None:
        fig, ax = plt.subplots()
    
    trace: NDArray[np.floating] = ta.trace(wavelength)

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
    plot = plot_spectrum(arr, 20)
    plt.show()
