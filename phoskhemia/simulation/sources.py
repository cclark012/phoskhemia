
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from phoskhemia.simulation.lineshapes import gaussian, lorentzian, voigt


from importlib import resources
def load_xenon_lines_csv() -> NDArray[np.floating]:
    with resources.files("phoskhemia.simulation").joinpath("data/xenon_lines_286-905.csv").open("r") as f:
        return np.loadtxt(f, delimiter=",", skiprows=1)

def xenon_spectrum(
        wavelengths: NDArray[np.floating], 
        *,
        scale: float = 1.0, 
        sigma: float = 1.0, 
        gamma: float = 1.0, 
        distribution: Literal['gaussian', 'lorentzian', 'lorentz', 'cauchy', 'voigt']="gaussian"
    ) -> NDArray[np.floating]:

    wl: NDArray[np.floating]  = np.asarray(wavelengths, dtype=float).reshape(-1)
    xenon_lines: NDArray[np.floating] = load_xenon_lines_csv()
    lw: NDArray[np.floating] = xenon_lines[:, 0]
    li: NDArray[np.floating] = xenon_lines[:, 1]

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    if distribution == "gaussian":
        spectrum = gaussian(wl[:, None], lw[None, :], sigma)

    elif distribution in ["lorentzian", "lorentz", "cauchy"]:
        spectrum = lorentzian(wl[:, None], lw[None, :], gamma)

    elif distribution == "voigt":
        spectrum = voigt(wl[:, None], lw[None, :], sigma, gamma)

    else:
        raise ValueError("distribution must be one of gaussian, lorentzian, lorentz, cauchy, or voigt")

    return (scale * (spectrum @ li)).astype(float)

def xenon_irradiance_emulated(
        wavelengths_nm: NDArray[np.floating],
        *,
        temperature_K: float = 6000.0,
        normalize: bool = True,
        window_nm: tuple[float, float] | None = None,
    ) -> NDArray[np.floating]:
    """
    Smooth xenon-like continuum surrogate (relative irradiance vs wavelength).

    Simulate an approximate xenon lamp spectrum with Planck's law for blackbody
    emitters: B(λ, T) = (2hc² / λ⁵) ⋅ [1 / (exp(hc / λkᵦT) - 1)]
    This is NOT a calibrated xenon lamp spectrum. Use measured / tabulated spectra
    when realism matters; this is a convenience fallback for synthetic noise.
    """
    wl_nm: NDArray[np.floating] = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    wl_m: NDArray[np.floating] = wl_nm * 1e-9

    # Planck spectral radiance vs wavelength (up to a constant factor)
    h: float = 6.62607015e-34
    c: float = 299792458.0
    kB: float = 1.380649e-23
    T: float = float(temperature_K)

    a: float = 2.0 * h * c**2
    b: float = h * c / (kB * T)
    x: NDArray[np.floating] = b / wl_m
    # avoid overflow in exp
    x = np.clip(x, 1e-9, 700.0)
    I: NDArray[np.floating] = a / (wl_m**5 * (np.exp(x) - 1.0))

    # Optional broad windowing to focus on a band (e.g., 320–850 nm)
    if window_nm is not None:
        lo: float
        hi: float
        lo, hi = map(float, window_nm)
        # smooth logistic edges
        s: float = 5.0  # nm edge softness
        w: NDArray[np.floating] = 1.0 / (1.0 + np.exp(-(wl_nm - lo) / s)) * (1.0 / (1.0 + np.exp((wl_nm - hi) / s)))
        I = I * w

    I = np.where(np.isfinite(I) & (I > 0), I, 0.0)

    if normalize:
        m: float = float(np.max(I)) if np.max(I) > 0 else 1.0
        I = I / m

    return I.astype(float)

