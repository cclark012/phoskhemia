from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz, gammaln

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

# TODO - Add other lineshapes. Add Franck-Condon factors for other transitions.
def displaced_harmonic_oscillator(
        wavelengths_nm: NDArray[np.floating], 
        huang_rhys_factor: float=1.0,
        lam00_nm: float=400.0, 
        frequency: float=1000.0, 
        linewidth: float=200.0,
        *,
        summations: int=10,
        amplitude: float = 1,
        normalize: bool = False,
        lineshape: Literal["gaussian", "lorentzian"] = "gaussian",
        process: Literal["absorption", "emission"] = "absorption"
    ) -> NDArray[np.floating]:
    """
    Generates a molecular absorption spectrum based on the displaced harmonic oscillator (DHO) model.
    
    The model is derived from the simplistic treatment of absorption as the 
    transition from the ground state to some vibronic (vibrational and electronic) 
    excited state. This model only accounts for harmonic potentials, so this model 
    becomes even less accurate for molecules with moderate to severe anharmonicity.

    Parameters
    ----------
    wavelengths_nm : NDArray[np.floating]
        Array of wavelengths to calculate the spectrum for.
    huang_rhys_factor : float, optional
        The Huang-Rhys factor, generally between ≈0.5 and 2.0 for molecules, 
        but can be much larger, by default 1.0.
    lam00_nm : float, optional
        Wavelength of 0-0 transition, by default 400.0.
    frequency : float, optional
        Spacing between energy levels in wavenumbers, by default 1000.0.
    linewidth : float, optional
        Line profile broadening in wavenumbers. By default 200.0.
    summations : int, optional
        Number of peaks to calculate for the spectrum. Should be 
        increased for large Huang-Rhys factors, by default 10.
    amplitude : float, optional
        Number to multiply the final spectrum values by. By default 1.
    normalize : bool, optional
        Whether to scale the maximum of the spectrum to 1. 
        If combined with the amplitude argument, will scale 
        the maximum to the value of amplitude. By default False.
    lineshape : Literal["gaussian", "lorentzian"], optional
        Lineshape to use for the progression, by default "gaussian".
    process : Literal["absorption", "emission"], optional
        Photophysical process to be modelled, by default "absorption".

    Returns
    -------
    NDArray[np.floating]
        The calculated spectrum in the wavelength representation.

    Raises
    ------
    ValueError
        If an unrecognized value is passed to lineshape.
    """

    if process not in ["absorption", "emission"]:
        raise ValueError("process must be either absorption or emission")

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    # Constants for conversions
    plancks: float = 6.62607015e-34
    speed_of_light: float = 299792458.0
    charge: float = 1.602176634e-19

    # Convert everything to eV
    lam: NDArray[np.floating] = (plancks * speed_of_light) / (1e-9 * wavelengths_nm * charge)
    lam00: float = (plancks * speed_of_light) / (1e-9 * lam00_nm * charge)
    displacement: float = (frequency * speed_of_light * plancks * 100) / charge
    sigma: float = (linewidth * plancks * speed_of_light * 100) / charge

    m: NDArray[np.int64] = np.arange(0, summations, 1, dtype=np.int64)[:, None]
    # Calculate Franck-Condon factors using Huang-Rhys factor S: For v" = 0, ⟨ν'|v"⟩ = (exp(-S) ∙ Sⱽ') / ν'!.
    franck_condon_factor: NDArray[np.floating] = np.exp(
        (m * np.log(huang_rhys_factor) - huang_rhys_factor) - gammaln(m+1)
    )

    # Calculate each peak in the progression assuming some distribution.
    sign = 1 if process == "absorption" else -1 # Reverse progression if process is emission
    if lineshape == "gaussian":
        progression: NDArray[np.floating] = (
            np.exp(-((lam00 + sign * m * displacement - lam) ** 2) / (2 * (sigma ** 2)))
        )
    elif lineshape == "lorentzian":
        progression: NDArray[np.floating] = (
            (1 / np.pi) * (sigma / ((lam00 + sign * m * displacement - lam) ** 2 + sigma ** 2))
        )
    else:
        raise ValueError("lineshape must be either gaussian or lorentzian")

    # Weight each peak by their wavenumber ṽ (according to Einstein coefficients) and Franck-Condon factors.
    # Emission process is also additionally modified by the nonlinear energy of each counting bin.
    # For absorption, a(ṽ) ⋅ ṽ = A(ṽ) = A(λ),and for emission, f(ṽ) ⋅ ṽ³ = F(ṽ) = F(λ) ⋅ λ².
    # x(ṽ) is the lineshape/transition dipole representation (directly proportional to populations).
    # X(ṽ) and X(λ) are the wavenumber and wavelength representations, respectively.
    nu: NDArray[np.floating] = 1e7 / wavelengths_nm
    prefactor: NDArray[np.floating] = (
        nu if process == "absorption" 
        else (nu ** 3) / (wavelengths_nm ** 2)
    )
    indiv: NDArray[np.floating] = (
        prefactor * franck_condon_factor * progression
    )

    # Final spectral shape is the sum of all contributions.
    spectrum: NDArray[np.floating] = np.sum(indiv, axis=0)

    # Normalize maximum to 1 if requested.
    if normalize:
        spectrum /= np.max(spectrum)

    # Scale spectrum by provided amplitude. Makes maximum = amplitude if also normalized.
    spectrum *= amplitude
    
    return spectrum
