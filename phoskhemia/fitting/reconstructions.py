from __future__ import annotations

from typing import TYPE_CHECKING
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from phoskhemia.fitting.results import GlobalFitResult
if TYPE_CHECKING:
    from phoskhemia.data.spectrum2d import TransientAbsorption


def _components(result: GlobalFitResult):
    traces = np.asarray(result.traces, dtype=float)
    amplitudes = np.asarray(result.amplitudes, dtype=float)
    if traces.shape != (result.times.size, len(result.species)):
        raise ValueError("traces must match the fitted time and species axes")
    if amplitudes.shape != (result.wavelengths.size, len(result.species)):
        raise ValueError("amplitudes must match the fitted wavelength and species axes")
    return traces, amplitudes


def _species_index(result: GlobalFitResult, species: int | str) -> int:
    if isinstance(species, str):
        try:
            return result.species.index(species)
        except ValueError as exc:
            raise ValueError(f"Unknown species '{species}'") from exc
    if isinstance(species, bool) or not isinstance(species, Integral):
        raise TypeError("species must be an integer index or a species name")
    index = int(species)
    if index < 0 or index >= len(result.species):
        raise IndexError("Species index out of range")
    return index


def reconstruct_fit(
        result: GlobalFitResult,
    ) -> TransientAbsorption:
    """
    Reconstruct the fitted transient absorption signal
    from a global kinetic fit.

    Parameters
    ----------
    result : GlobalFitResult
        Result returned by fit_global_kinetics

    Returns
    -------
    fit : TransientAbsorption
        Reconstructed fitted signal (n_times, n_wavelengths)
    """

    traces, amplitudes = _components(result)
    times: NDArray[np.floating] = result.times
    wavelengths: NDArray[np.floating] = result.wavelengths

    data: NDArray[np.floating] = traces @ amplitudes.T        # (n_times, n_wl)

    from phoskhemia.data.spectrum2d import TransientAbsorption
    return TransientAbsorption(
        data,
        x=wavelengths,
        y=times,
    )

def reconstruct_species(
        result: GlobalFitResult,
        species: int | str,
    ) -> TransientAbsorption:
    """
    Reconstruct the contribution of a single kinetic species
    to the fitted transient absorption signal.

    Parameters
    ----------
    result : GlobalFitResult
        Result returned by fit_global_kinetics
    species : int or str
        Species index or name

    Returns
    -------
    component : TransientAbsorption
        Contribution of the selected species
    """

    traces, amplitudes = _components(result)
    times: NDArray[np.floating] = result.times
    wavelengths: NDArray[np.floating] = result.wavelengths
    idx = _species_index(result, species)

    data: NDArray[np.floating] = np.outer(traces[:, idx], amplitudes[:, idx])

    from phoskhemia.data.spectrum2d import TransientAbsorption
    return TransientAbsorption(
        data,
        x=wavelengths,
        y=times,
    )


def reconstruct_residuals(
        result: GlobalFitResult,
        data: TransientAbsorption,
    ) -> TransientAbsorption:
    """Subtract the stored fit from data on the same time/wavelength grid.

    Amplitudes remain fixed at their fitted values. To re-estimate amplitudes
    at new wavelengths, use ``validation.compute_residual_maps`` instead.
    Matching array sizes alone are insufficient: the coordinates must match.
    """
    from phoskhemia.data.spectrum2d import TransientAbsorption

    if not isinstance(data, TransientAbsorption):
        raise TypeError("data must be a TransientAbsorption with time and wavelength axes")
    if not np.array_equal(data.x, result.wavelengths) or not np.array_equal(data.y, result.times):
        raise ValueError("data coordinates must match the fitted time and wavelength axes")
    prediction = np.asarray(reconstruct_fit(result))
    if data.shape != prediction.shape:
        raise ValueError("data shape must match the fitted signal")
    return TransientAbsorption(
        np.asarray(data, dtype=float) - prediction,
        x=result.wavelengths,
        y=result.times,
        meta=data.meta,
    )


def reconstruct_weighted_residuals(
        result: GlobalFitResult,
        data: TransientAbsorption,
        *,
        noise: NDArray[np.floating] | float | None = None,
    ) -> TransientAbsorption:
    """Return fixed-fit residuals divided by observation standard deviations.

    By default, use the noise cached by the fit. Explicit noise may be scalar,
    per wavelength, or broadcastable to the complete data matrix. This scales
    individual residuals; it does not remove correlations introduced by fitting
    or by preprocessing.
    """
    from phoskhemia.data.spectrum2d import TransientAbsorption
    from phoskhemia.fitting.validation import _validated_noise

    raw = reconstruct_residuals(result, data)
    if noise is None:
        noise = result._cache.get("noise")
    if noise is None:
        raise ValueError("noise must be provided when no fitted noise is cached")
    sigma = _validated_noise(noise, raw.shape)
    metadata = dict(data.meta)
    # The original signal's noise units no longer apply to normalized residuals.
    metadata.pop("noise_t0", None)
    return TransientAbsorption(
        np.asarray(raw) / sigma,
        x=result.wavelengths,
        y=result.times,
        meta=metadata,
    )


def reconstruct_species_sum(
        result: GlobalFitResult,
        species: list[int | str],
    ) -> TransientAbsorption:
    """Reconstruct selected species once each; an empty selection gives zero."""
    from phoskhemia.data.spectrum2d import TransientAbsorption

    traces, amplitudes = _components(result)
    indices = [_species_index(result, value) for value in species]
    if len(set(indices)) != len(indices):
        raise ValueError("species selection must not contain duplicates")
    signal = traces[:, indices] @ amplitudes[:, indices].T
    return TransientAbsorption(signal, x=result.wavelengths, y=result.times)
