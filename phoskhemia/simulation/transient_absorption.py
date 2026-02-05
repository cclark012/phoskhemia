import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics import KineticModel
from phoskhemia.data import TransientAbsorption

def simulate_ta(
        kinetic_model: KineticModel,
        times: NDArray[np.floating],
        wavelengths: NDArray[np.floating],
        beta: NDArray[np.floating],
        amplitudes: NDArray[np.floating],
        *,
        noise_std: float | NDArray[np.floating] = 0.0,
        random_state: int | None = None,
    ) -> TransientAbsorption:
    """
    Simulate transient absorption data.
    """
    rng = np.random.default_rng(random_state)

    traces = kinetic_model.solve(times, beta)     # (n_times, n_species)

    if amplitudes.shape != (len(wavelengths), traces.shape[1]):
        raise ValueError("Amplitude shape mismatch")

    data = traces @ amplitudes.T

    if np.any(noise_std):
        noise = noise_std if np.ndim(noise_std) else noise_std * np.ones(len(wavelengths))
        data = data + rng.normal(scale=noise, size=data.shape)

    return TransientAbsorption(data, x=wavelengths, y=times)
