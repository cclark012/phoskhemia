from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz, gammaln


def gaussian(
        x: NDArray[np.float64] | float, 
        x0: NDArray[np.float64] | float = 0.0, 
        sigma: float = 1.0
    ) -> NDArray[np.float64]:
    """
    Calculates the Gaussian/Normal probability distribution for given x.

    Parameters
    ----------
    x : NDArray[np.float64] | float
        Value(s) for which to calculate the PDF.
    x0 : NDArray[np.float64] | float
        Mean/expectation value of the distribution, by default 0.0.
    sigma : float, optional
        Standard deviation of the distribution, by default 1.0.

    Returns
    -------
    NDArray[np.float64]
        Value(s) of the probability distribution.
    """

    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def lorentzian(
        x: NDArray[np.float64] | float, 
        x0: NDArray[np.float64] | float = 0.0, 
        gamma: float = 1.0
    ) -> NDArray[np.float64]:
    """
    Calculates the Lorentzian/Cauchy probability distribution for given x.

    Parameters
    ----------
    x : NDArray[np.float64] | float
        Value(s) for which to calculate the PDF.
    x0 : NDArray[np.float64] | float
        Median/mode value of the distribution, by default 0.0.
    gamma : float, optional
        Half-width half-maximum of the distribution, by default 1.0.

    Returns
    -------
    NDArray[np.float64]
        Value(s) of the probability distribution.
    """

    return (1 / np.pi) * (gamma / ((x - x0) ** 2 + gamma ** 2))


def voigt(
        x: NDArray[np.float64] | float, 
        x0: NDArray[np.float64] | float = 0.0, 
        sigma: float = 1.0, 
        gamma: float = 1.0
    ) -> NDArray[np.float64]:
    """
    Calculates the Voigt probability distribution for given x.

    The Voigt distribution is a convolution of the Gaussian and Lorentzian 
    probability distributions. It finds use in many spectroscopy applications.
    The full-width half-maximum (FWHM) can be approximated with the FWHM of 
    the Gaussian and Lorentzian functions.

    Gaussian FWHM: $f_G = 2\sigma\sqrt{2\ln(2)}$
    
    Lorentzian FWHM: $f_L = 2\gamma$
    
    The FWHM of the Voigt profile can be approximated as:
    
    $f_V \approx f_L/2 + \sqrt{f^2_L/4 + f^2_G}$ to within 1.2%.[1]_
    
    $f_V \approx 0.5343f_L + \sqrt{0.2169f^2_L + f^2_G}$ to within 0.02%.[2]_


    Parameters
    ----------
    x : NDArray[np.float64] | float
        Value(s) for which to calculate the PDF.
    x0 : NDArray[np.float64] | float
        Median/mode value of the distribution, by default 0.0.
    sigma : float, optional
        Standard deviation of the Gaussian profile, by default 1.0.
    gamma : float, optional
        Half-width half-maximum of the Lorentzian profile, by default 1.0.

    Returns
    -------
    NDArray[np.float64]
        Value(s) of the probability distribution.
    
    References
    ----------
    .. [1] E. E. Whiting, Journal of Quantitative Spectroscopy and Radiative Transfer, 1968, 8, 1379-1384.
    .. [2] J. F. Kielkopf, J. Opt. Soc. Am., 1973, 63, 987.
    """

    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


# TODO - Add other lineshapes. Add Franck-Condon factors for other transitions.
def displaced_harmonic_oscillator(
        wavelengths_nm: NDArray[np.float64], 
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
    ) -> NDArray[np.float64]:
    """
    Generates a molecular absorption spectrum based on the displaced harmonic oscillator (DHO) model.[1]_ [2]_
    
    The model is derived from the simplistic treatment of absorption as the 
    transition from the ground state to some vibronic (vibrational and electronic) 
    excited state. This model only accounts for harmonic potentials, so this model 
    becomes even less accurate for molecules with moderate to severe anharmonicity.

    Parameters
    ----------
    wavelengths_nm : NDArray[np.float64]
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
    NDArray[np.float64]
        The calculated spectrum in the wavelength representation.

    Raises
    ------
    ValueError
        If an unrecognized value is passed to lineshape or process.
    
    References
    ----------
    .. [1] Y. Fujimura, K. Sato and T. Nakajima, Bulletin of the Chemical Society of Japan, 1982.
    .. [2] R. A. Marcus, J. Phys. Chem., 1989, 93, 3078-3086.
    """

    if process not in ["absorption", "emission"]:
        raise ValueError("process must be either absorption or emission")

    wavelengths_nm: NDArray[np.float64] = np.asarray(wavelengths_nm, dtype=float)

    # Constants for conversions
    plancks: float = 6.62607015e-34
    speed_of_light: float = 299792458.0
    charge: float = 1.602176634e-19

    # Convert everything to eV
    lam: NDArray[np.float64] = (plancks * speed_of_light) / (1e-9 * wavelengths_nm * charge)
    lam00: float = (plancks * speed_of_light) / (1e-9 * lam00_nm * charge)
    displacement: float = (frequency * speed_of_light * plancks * 100) / charge
    sigma: float = (linewidth * plancks * speed_of_light * 100) / charge

    m: NDArray[np.int64] = np.arange(0, summations, 1, dtype=np.int64)[:, None]
    # Calculate Franck-Condon factors using Huang-Rhys factor S: For v" = 0, ⟨ν'|v"⟩ = (exp(-S) ∙ Sⱽ') / ν'!.
    franck_condon_factor: NDArray[np.float64] = np.exp(
        (m * np.log(huang_rhys_factor) - huang_rhys_factor) - gammaln(m+1)
    )

    # Calculate each peak in the progression assuming some distribution.
    sign: int = 1 if process == "absorption" else -1 # Reverse progression if process is emission
    if lineshape == "gaussian":
        progression: NDArray[np.float64] = (
            np.exp(-((lam00 + sign * m * displacement - lam) ** 2) / (2 * (sigma ** 2)))
        )
    elif lineshape == "lorentzian":
        progression: NDArray[np.float64] = (
            (1 / np.pi) * (sigma / ((lam00 + sign * m * displacement - lam) ** 2 + sigma ** 2))
        )
    else:
        raise ValueError("lineshape must be either gaussian or lorentzian")

    # Weight each peak by their wavenumber ṽ (according to Einstein coefficients) and Franck-Condon factors.
    # Emission process is also additionally modified by the nonlinear energy of each counting bin.
    # For absorption, a(ṽ) ⋅ ṽ = A(ṽ) = A(λ),and for emission, f(ṽ) ⋅ ṽ³ = F(ṽ) = F(λ) ⋅ λ².
    # x(ṽ) is the lineshape/transition dipole representation (directly proportional to populations).
    # X(ṽ) and X(λ) are the wavenumber and wavelength representations, respectively.
    nu: NDArray[np.float64] = 1e7 / wavelengths_nm
    prefactor: NDArray[np.float64] = (
        nu if process == "absorption" 
        else (nu ** 3) / (wavelengths_nm ** 2)
    )
    indiv: NDArray[np.float64] = (
        prefactor * franck_condon_factor * progression
    )

    # Final spectral shape is the sum of all contributions.
    spectrum: NDArray[np.float64] = np.sum(indiv, axis=0)

    # Normalize maximum to 1 if requested.
    if normalize:
        spectrum /= np.max(spectrum)

    # Scale spectrum by provided amplitude. Makes maximum = amplitude if also normalized.
    spectrum *= amplitude
    
    return spectrum
