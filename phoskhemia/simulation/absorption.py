import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


def dho_absorption(
        wavelengths_nm: NDArray[np.floating],  
        huang_rhys_factor: float=1.0,
        lam00_nm: float=400.0, 
        effective_freq_wn: float=1000.0, 
        sigma_wn: float=207.12,
        summations: int=10
    ) -> NDArray[np.floating]:


    plancks: float = 6.62607015e-34
    speed_of_light: float = 299792458.0
    charge: float = 1.602176634e-19

    # Convert everything to eV
    zero_phonon: float = (plancks * speed_of_light) / (1e-9 * lam00_nm * charge)
    displacement: float = (effective_freq_wn * speed_of_light * plancks * 100) / charge
    mu: NDArray[np.floating] = (plancks * speed_of_light) / (1e-9 * wavelengths_nm * charge)
    sigma: float = (sigma_wn * plancks * speed_of_light * 100) / charge

    m: NDArray[np.int64] = np.arange(0, summations, 1, dtype=np.int64)[:, None]
    franck_condon_factor: NDArray[np.floating] = (
        ((huang_rhys_factor ** m) * np.exp(-huang_rhys_factor)) / np.exp(gammaln(m))
    )
    progression: NDArray[np.floating] = (
        np.exp(-((zero_phonon + m * displacement - mu) ** 2) / (2 * (sigma ** 2)))
    )
    abs_indiv: NDArray[np.floating] = (
        (1e7 / wavelengths_nm) * franck_condon_factor * progression
    )

    return np.sum(abs_indiv, axis=0)

if __name__ == "__main__":
    wavelength1 = np.arange(325, 405, 10)
    absorption1 = dho_absorption(wavelength1)
    wavelength = np.arange(250, 450, 0.1)
    absorption = dho_absorption(wavelength)
    print(absorption1)
    import matplotlib.pyplot as plt
    plt.plot(wavelength, absorption)
    plt.scatter(wavelength1, absorption1, c='k', s=10)
    plt.show()
