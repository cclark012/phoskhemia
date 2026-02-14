import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

def gaussian(
        x: NDArray[np.floating] | float, 
        x0: NDArray[np.floating] | float, 
        sigma: float = 1.0
    ) -> NDArray[np.floating]:
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def lorentzian(
        x: NDArray[np.floating] | float, 
        x0: NDArray[np.floating] | float, 
        gamma: float = 1.0
    ) -> NDArray[np.floating]:
    return (1 / np.pi) * (gamma / ((x - x0) ** 2 + gamma ** 2))

def voigt(
        x: NDArray[np.floating] | float, 
        x0: NDArray[np.floating] | float, 
        sigma: float = 1.0, 
        gamma: float = 1.0
    ) -> NDArray[np.floating]:
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
