import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import scipy as sp

def displaced_harmonic_oscillator_absorption(
        wavelength_range_nm: tuple[float, float]=(350.0, 450.0), 
        amplitude: float=0.001, 
        huang_rhys_factor: float=2.0,
        lam00_nm: float=425.0, 
        effective_freq_wn: float=2500.0, 
        sigma_wn: float=100.0,
        resolution_nm: float=1.0,
        temperature: float=298.0,
        units: str="wavenumber",
        summations: int=10
    ) -> tuple[NDArray[np.floating], ...]:
    """
    Generates a molecular absorption spectrum based on the displaced harmonic oscillator (DHO) model.
    The model is derived from the simplistic treatment of absorption as the transition from the ground state
    to some vibronic (vibrational and electronic) excited state. This model only accounts for harmonic potentials,
    so this model becomes even less accurate for molecules with moderate to severe anharmonicity.

    Args:
        wavelength_range_nm (tuple[float, float], optional): _description_. Defaults to (350, 450).
        amplitude (float, optional): _description_. Defaults to 0.001.
        huang_rhys_factor (float, optional): _description_. Defaults to 2.0.
        lam00_nm (float, optional): _description_. Defaults to 425.
        effective_freq_wn (float, optional): _description_. Defaults to 2500.
        sigma_wn (float, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_

    """

    boltzmann: float = 1.380649e-23
    plancks: float = 6.62607015e-34
    speed_of_light: float = 299792458.0
    charge: float = 1.602176634e-19
    thermal_wn: float = (temperature * boltzmann) / (speed_of_light * plancks * 100)
    #sigma_wn: float = thermal_wn
    # print(thermal_wn)
    
    lam: NDArray[np.floating] = np.arange(wavelength_range_nm[0], wavelength_range_nm[1], resolution_nm)
    # nu = 1e7 / lam
    # nu0: float = (1e7 / lam00_nm)
    # Convert everything to eV
    if units == "wavenumber":
        zero_phonon: float = (plancks * speed_of_light) / (1e-9 * lam00_nm * charge)
        displacement: float = (effective_freq_wn * speed_of_light * plancks * 100) / charge
        mu: NDArray[np.floating] = (plancks * speed_of_light) / (1e-9 * lam * charge)
        sigma: float = (sigma_wn * plancks * speed_of_light * 100) / charge

    # print(zero_phonon, displacement, sigma)
    m: NDArray[np.floating] = np.atleast_2d(np.arange(0, summations, 1)).T
    franck_condon_factor: NDArray[np.floating] = (((huang_rhys_factor ** m) * np.exp(-huang_rhys_factor)) / sp.special.factorial(m))
    progression: NDArray[np.floating] = np.exp(-((zero_phonon + m * displacement - mu) ** 2) / (2 * (sigma ** 2)))
    a_abs_indiv: NDArray[np.floating] = amplitude * (1e7 / lam) * franck_condon_factor * progression

    return lam, a_abs_indiv, np.sum(a_abs_indiv, axis=0)

def displaced_harmonic_oscillator_emission(
        wavelength_range_nm: tuple[float, float]=(350.0, 450.0), 
        amplitude: float=0.001, 
        huang_rhys_factor: float=2.0,
        lam00_nm: float=425.0, 
        effective_freq_wn: float=2500.0, 
        sigma_wn: float=100.0,
        resolution_nm: float=1.0,
        temperature: float=298.0,
        units: str="wavenumber"
    ) -> tuple[NDArray[np.floating], ...]:
    """
    Generates a molecular absorption spectrum based on the displaced harmonic oscillator (DHO) model.
    The model is derived from the simplistic treatment of absorption as the transition from the ground state
    to some vibronic (vibrational and electronic) excited state. This model only accounts for harmonic potentials,
    so this model becomes even less accurate for molecules with moderate to severe anharmonicity.

    Args:
        wavelength_range_nm (tuple[float, float], optional): _description_. Defaults to (350, 450).
        amplitude (float, optional): _description_. Defaults to 0.001.
        huang_rhys_factor (float, optional): _description_. Defaults to 2.0.
        lam00_nm (float, optional): _description_. Defaults to 425.
        effective_freq_wn (float, optional): _description_. Defaults to 2500.
        sigma_wn (float, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_

    """

    boltzmann: float = 1.380649e-23
    plancks: float = 6.62607015e-34
    speed_of_light: float = 299792458.0
    charge: float = 1.602176634e-19
    thermal_wn: float = (temperature * boltzmann) / (speed_of_light * plancks * 100)
    #sigma_wn: float = thermal_wn
    # print(thermal_wn)
    
    lam: NDArray[np.floating] = np.arange(wavelength_range_nm[0], wavelength_range_nm[1], resolution_nm)
    # nu = 1e7 / lam
    # nu0: float = (1e7 / lam00_nm)
    # Convert everything to eV
    if units == "wavenumber":
        zero_phonon: float = (plancks * speed_of_light) / (1e-9 * lam00_nm * charge)
        displacement: float = (effective_freq_wn * speed_of_light * plancks * 100) / charge
        mu = (plancks * speed_of_light) / (1e-9 * lam * charge)
        sigma = (sigma_wn * plancks * speed_of_light * 100) / charge

    # print(zero_phonon, displacement, sigma)
    m: NDArray[np.floating] = np.atleast_2d(np.arange(0, 10, 1)).T
    franck_condon_factor: NDArray[np.floating] = (((huang_rhys_factor ** m) * np.exp(-huang_rhys_factor)) / sp.special.factorial(m))
    progression: NDArray[np.floating] = np.exp(-((zero_phonon - m * displacement - mu) ** 2) / (2 * (sigma ** 2)))
    a_em_indiv: NDArray[np.floating] = amplitude * ((1e7 / lam) ** 3) * franck_condon_factor * progression

    return lam, a_em_indiv, np.sum(a_em_indiv, axis=0)

def dho_gui() -> None:
    fig = plt.figure(figsize=(12, 8))
    subfig = fig.subfigures(1, 1)
    gs = subfig.add_gridspec(1, 1, left=0.125, bottom=0.25, right=0.95, top=0.95)
    axes = gs.subplots()

    dw = 0.25
    init_center1 = 450
    init_scale1 = -3
    init_width1 = 300
    init_progress1 = 1.0
    init_frequency1 = 2000
    wavelength, progression, abs_spectrum1 = displaced_harmonic_oscillator_absorption(
        wavelength_range_nm=(50, 1250), 
        amplitude=1e-4,
        huang_rhys_factor=init_progress1,
        lam00_nm=init_center1,
        effective_freq_wn=init_frequency1,
        sigma_wn=init_width1,
        resolution_nm=dw,
        temperature=298,
        units="wavenumber",
        summations=20
        )
    
    init_center2 = 350
    init_scale2 = -3
    init_width2 = 200
    init_progress2 = 1.5
    init_frequency2 = 1500
    wavelength, progression, abs_spectrum2 = displaced_harmonic_oscillator_absorption(
        wavelength_range_nm=(50, 1250), 
        amplitude=1e-4,
        huang_rhys_factor=init_progress2,
        lam00_nm=init_center2,
        effective_freq_wn=init_frequency2,
        sigma_wn=init_width2,
        resolution_nm=dw,
        temperature=298,
        units="wavenumber",
        summations=20
        )
    
    abs_spectrum = init_scale1 * abs_spectrum1 + init_scale2 * abs_spectrum2

    dydx = np.gradient(abs_spectrum, wavelength)
    mask = np.isclose(dydx, 0.0, rtol=1e-6, atol=np.max(np.abs(dydx)) * 0.000001)
    line1, = axes.plot(wavelength[~mask], abs_spectrum1[~mask], 'tab:blue', lw=2, alpha=0.75)
    line2, = axes.plot(wavelength[~mask], abs_spectrum2[~mask], 'tab:red', lw=2, alpha=0.75)
    line3, = axes.plot(wavelength[~mask], abs_spectrum[~mask], 'k', lw=1)
    # line2, = axes.plot(wavelength[~mask], dydx[~mask], 'k', lw=1)
    axes.set(xlim=(np.min(wavelength[~mask]), np.max(wavelength[~mask])), 
                ylim=(
                    np.min((np.min(abs_spectrum), np.min(abs_spectrum1), np.min(abs_spectrum2))), 
                    np.max((np.max(abs_spectrum), np.max(abs_spectrum1), np.max(abs_spectrum2))) * 1.1)
                )

    # Create axes of [left, bottom, width, height]
    ax_center1 = fig.add_axes([0.15, 0.075, 0.1, 0.05])
    ax_scale1 = fig.add_axes([0.3, 0.075, 0.1, 0.05])
    ax_width1 = fig.add_axes([0.45, 0.075, 0.1, 0.05])
    ax_progress1 = fig.add_axes([0.6, 0.075, 0.1, 0.05])
    ax_frequency1 = fig.add_axes([0.75, 0.075, 0.1, 0.05])

    ax_center2 = fig.add_axes([0.15, 0.025, 0.1, 0.05])
    ax_scale2 = fig.add_axes([0.3, 0.025, 0.1, 0.05])
    ax_width2 = fig.add_axes([0.45, 0.025, 0.1, 0.05])
    ax_progress2 = fig.add_axes([0.6, 0.025, 0.1, 0.05])
    ax_frequency2 = fig.add_axes([0.75, 0.025, 0.1, 0.05])

    ax_center1.set(title='0-0 λ (nm)')
    ax_scale1.set(title='Log(A)')
    ax_width1.set(title='Width (ν)')
    ax_progress1.set(title='Huang-Rhys Factor')
    ax_frequency1.set(title='Frequency')

    reset_ax = fig.add_axes([0.9, 0.025, 0.075, 0.05])
    print_ax = fig.add_axes([0.9, 0.125, 0.075, 0.05])

    center_slider1 = Slider(ax=ax_center1, label="", valmin=100, valmax=1200, valstep=10, valinit=init_center1, orientation='horizontal')
    scale_slider1 = Slider(ax=ax_scale1, label="", valmin=-10, valmax=10, valstep=0.1, valinit=init_scale1, orientation='horizontal')
    width_slider1 = Slider(ax=ax_width1, label="", valmin=10, valmax=1200, valstep=10, valinit=init_width1, orientation='horizontal')
    progress_slider1 = Slider(ax=ax_progress1, label="", valmin=0, valmax=5, valstep=0.1, valinit=init_progress1, orientation='horizontal')
    frequency_slider1 = Slider(ax=ax_frequency1, label="", valmin=100, valmax=5000, valstep=50, valinit=init_frequency1, orientation='horizontal')

    center_slider2 = Slider(ax=ax_center2, label="", valmin=100, valmax=1200, valstep=10, valinit=init_center2, orientation='horizontal')
    scale_slider2 = Slider(ax=ax_scale2, label="", valmin=-10, valmax=10, valstep=0.1, valinit=init_scale2, orientation='horizontal')
    width_slider2 = Slider(ax=ax_width2, label="", valmin=10, valmax=1200, valstep=10, valinit=init_width2, orientation='horizontal')
    progress_slider2 = Slider(ax=ax_progress2, label="", valmin=0, valmax=5, valstep=0.1, valinit=init_progress2, orientation='horizontal')
    frequency_slider2 = Slider(ax=ax_frequency2, label="", valmin=100, valmax=5000, valstep=50, valinit=init_frequency2, orientation='horizontal')

    def update(val):
        _, _, abs_spectrum1 = displaced_harmonic_oscillator_absorption(
            wavelength_range_nm=(50, 1250), 
            amplitude=1e-4,
            huang_rhys_factor=progress_slider1.val,
            lam00_nm=center_slider1.val,
            effective_freq_wn=frequency_slider1.val,
            sigma_wn=width_slider1.val,
            resolution_nm=dw,
            temperature=298,
            units="wavenumber",
            summations=20
            )
        _, _, abs_spectrum2 = displaced_harmonic_oscillator_absorption(
            wavelength_range_nm=(50, 1250), 
            amplitude=1e-4,
            huang_rhys_factor=progress_slider2.val,
            lam00_nm=center_slider2.val,
            effective_freq_wn=frequency_slider2.val,
            sigma_wn=width_slider2.val,
            resolution_nm=dw,
            temperature=298,
            units="wavenumber",
            summations=20
            )
        abs_spectrum = scale_slider1.val * abs_spectrum1 + scale_slider2.val * abs_spectrum2
        dydx = np.gradient(abs_spectrum, wavelength)
        mask = np.isclose(dydx, 0.0, rtol=1e-6, atol=np.max(np.abs(dydx)) * 0.000001)
        line1.set_xdata(wavelength[~mask])
        line2.set_xdata(wavelength[~mask])
        line3.set_xdata(wavelength[~mask])
        line1.set_ydata(abs_spectrum1[~mask])
        line2.set_ydata(abs_spectrum2[~mask])
        line3.set_ydata(abs_spectrum[~mask])
        # line2.set_xdata(wavelength[~mask])
            # line2.set_ydata(dydx[~mask])
        axes.set(xlim=(np.min(wavelength[~mask]), np.max(wavelength[~mask])), 
                    ylim=(
                        np.min((np.min(abs_spectrum), np.min(abs_spectrum1), np.min(abs_spectrum2))), 
                        np.max((np.max(abs_spectrum), np.max(abs_spectrum1), np.max(abs_spectrum2))) * 1.1)
                    )
        fig.canvas.draw_idle()

    center_slider1.on_changed(update)
    scale_slider1.on_changed(update)
    width_slider1.on_changed(update)
    progress_slider1.on_changed(update)
    frequency_slider1.on_changed(update)
    
    center_slider2.on_changed(update)
    scale_slider2.on_changed(update)
    width_slider2.on_changed(update)
    progress_slider2.on_changed(update)
    frequency_slider2.on_changed(update)

    reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    print_button = Button(print_ax, 'Print', hovercolor='0.975')
    
    def reset(event):
        center_slider1.reset()
        scale_slider1.reset()
        width_slider1.reset()
        progress_slider1.reset()
        frequency_slider1.reset()
        
        center_slider2.reset()
        scale_slider2.reset()
        width_slider2.reset()
        progress_slider2.reset()
        frequency_slider2.reset()

    def print_vals(event):
        center1 = center_slider1.val
        width1 = width_slider1.val
        scale1 = scale_slider1.val
        progress1 = progress_slider1.val
        frequency1 = frequency_slider1.val

        center2 = center_slider2.val
        width2 = width_slider2.val
        scale2 = scale_slider2.val
        progress2 = progress_slider2.val
        frequency2 = frequency_slider2.val
        
        print(f"Spectrum 1")
        print(f"λ₀₋₀: {center1 :.1f}")
        print(f"σᵥ: {width1 :.1f}")
        print(f"A: {scale1 :.1f}")# = log({float(np.power(10., scale1)) :.1e})")
        print(f"HR: {progress1 :.1f}")
        print(f"ν: {frequency1 :.1f}")

        print(f"\nSpectrum 2")
        print(f"λ₀₋₀: {center2 :.1f}")
        print(f"σᵥ: {width2 :.1f}")
        print(f"A: {scale2 :.1f}")# = log({float(np.power(10., scale1)) :.1e})")
        print(f"HR: {progress2 :.1f}")
        print(f"ν: {frequency2 :.1f}")

        
    reset_button.on_clicked(reset)
    print_button.on_clicked(print_vals)
    plt.show()
