from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from phoskhemia.fitting.results import GlobalFitResult
if TYPE_CHECKING:
    from phoskhemia.data.spectrum_handlers import TransientAbsorption

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

    traces: NDArray[np.floating] = result.traces            # (n_times, n_species)
    times: NDArray[np.floating] = result.times
    amplitudes: NDArray[np.floating] = result.amplitudes      # (n_wl, n_species)
    wavelengths: NDArray[np.floating] = result.wavelengths

    data: NDArray[np.floating] = traces @ amplitudes.T        # (n_times, n_wl)

    from phoskhemia.data.spectrum_handlers import TransientAbsorption
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

    cache: dict[str, Any] = result._cache

    traces: NDArray[np.floating] = result.traces           # (n_times, n_species)
    times: NDArray[np.floating] = result.times
    amplitudes: NDArray[np.floating] = result.amplitudes      # (n_wl, n_species)
    wavelengths: NDArray[np.floating] = result.wavelengths
    species_names: str | list[str] | tuple[str, ...] = result.species

    if isinstance(species, str):
        try:
            idx: int = species_names.index(species)
        except ValueError:
            raise ValueError(f"Unknown species '{species}'")
    else:
        idx: int = int(species)
        if idx < 0 or idx >= traces.shape[1]:
            raise IndexError("Species index out of range")

    data: NDArray[np.floating] = np.outer(traces[:, idx], amplitudes[:, idx])

    from phoskhemia.data.spectrum_handlers import TransientAbsorption
    return TransientAbsorption(
        data,
        x=wavelengths,
        y=times,
    )


def reconstruct_residuals(result: GlobalFitResult, data: TransientAbsorption):
    raise NotImplementedError

def reconstruct_weighted_residuals():
    raise NotImplementedError

def reconstruct_species_sum(result: GlobalFitResult, species: list[int | str]):
    raise NotImplementedError
