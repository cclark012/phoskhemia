from scipy.integrate import solve_ivp
import numpy as np
from numpy.typing import NDArray # noqa
from matplotlib import ticker

from phoskhemia.simulation.transient_absorption import excitation_fraction, PumpSpec, simulate_ta
from phoskhemia.kinetics import KineticModel
from phoskhemia.visualization.colormaps import sample_colormap


class TTAKineticModel(KineticModel):
    """
    Triplet–triplet annihilation kinetic model.
    """

    def n_params(self):
        return 4

    def param_names(self):
        return ["τ₃ₛ", "τᵣ", "τ₃ₐ", "k₂"]

    def species_names(self):
        return ["[³A*]", "[³S*]"]

    def solve(self, times, beta):
        t3s, tr, t3a, kappa2 = np.exp(beta)

        times = np.asarray(times, dtype=float)
        # Sensitizer population, kₚₕ = τ₃ₛ⁻¹ + τᵣ⁻¹
        C_S = np.exp(-times * ((1 / t3s) + (1 / tr)))

        def rhs(t, y):
            A = y[0]
            S = np.exp(-t * ((1 / t3s) + (1 / tr)))
            dA = (1.0 / tr) * S - (1.0 / t3a) * A - kappa2 * A * A
            return [dA]

        sol = solve_ivp(
            rhs,
            (times[0], times[-1]),
            y0=[0.0],
            t_eval=times,
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError("ODE solver failed")

        C_A = sol.y[0]

        return np.column_stack([C_A, C_S])


def simulation() -> None:
    import matplotlib.pyplot as plt
    from phoskhemia.simulation.lineshapes import displaced_harmonic_oscillator
    from phoskhemia.fitting.reconstructions import reconstruct_fit
    from phoskhemia.kinetics import SensitizedTTAModel
    from time import perf_counter_ns

    #1) Time and wavelength ranges
    times: NDArray[np.floating] = np.linspace(1, 1e6, 50000)
    wavelengths: NDArray[np.floating] = np.arange(350, 650, 2)
    
    # Spectrum 1 (GSB)
    # A: -1
    # λ₀₋₀: 420.0
    # HR: 1.0
    # ν: 1500.0
    # σᵥ: 520.0
    
    # Spectrum 2 (ESA)
    # A: 0.4
    # λ₀₋₀: 470.0
    # HR: 0.9
    # ν: 2050.0
    # σᵥ: 900.0

    # Spectrum 3 (GSB)
    # A: -1.0
    # λ₀₋₀: 550.0
    # HR: 0.3
    # ν: 750.0
    # σᵥ: 300.0

    # Spectrum 4 (ESA)
    # A: 0.2
    # λ₀₋₀: 670.0
    # HR: 2.2
    # ν: 950.0
    # σᵥ: 520.0

    #2) Ground and excited absorption spectra for molecules
    gsb1: NDArray[np.floating] = displaced_harmonic_oscillator(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=1.0,
            lam00_nm=470,
            frequency=1500,
            linewidth=520,
            )
    esa1: NDArray[np.floating] = displaced_harmonic_oscillator(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=0.9,
            lam00_nm=520,
            frequency=2050,
            linewidth=900,
        )
    gsb2: NDArray[np.floating] = displaced_harmonic_oscillator(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=0.3,
            lam00_nm=550,
            frequency=750,
            linewidth=300,
        )
    esa2: NDArray[np.floating] = displaced_harmonic_oscillator(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=2.2,
            lam00_nm=670,
            frequency=950,
            linewidth=520,
        )

    # Molecular components
    spectrum1: NDArray[np.floating] = -1.0 * gsb1 + 0.4 * esa1
    spectrum2: NDArray[np.floating] = -1.0 * gsb2 + 0.2 * esa2

    # Scale spectra to chosen reference points
    scale_factor1: float = 44000 / np.max(gsb1)
    gsb1 *= scale_factor1
    esa1 = esa1 * 0.4 * scale_factor1
    spectrum1 *= scale_factor1
    
    scale_factor2: float = (98500 / np.max(gsb2))
    gsb2 *= scale_factor2
    esa2 = 0.2 * esa2 * scale_factor2
    spectrum2 *= scale_factor2
    
    # Final (2, n) array of spectra
    spectra: NDArray[np.floating] = np.vstack((spectrum2, spectrum1))

    #3) Parameters for simulation
    conc1: float = 5.e-4 # Concentration of first molecule, in M
    conc2: float = 1.e-5 # Concentration of second molecule, in M
    path: float = 0.1 # cm
    pump_wl: float = 532.0 # nm
    pump_energy: float = 2.5e-4 # J
    beam_area: float = 0.5 # cm²
    
    # Ground-state spectrum
    ground_state: NDArray[np.floating] = path * (conc1 * gsb1 + conc2 * gsb2) # noqa

    eps: float = gsb2[np.argmin(np.abs(wavelengths - pump_wl))]
    pumped: float = excitation_fraction(
        eps_pump=eps, 
        conc_M=conc2, 
        pump=PumpSpec(
            lambda_pump_nm=pump_wl, 
            pathlength_cm=path, 
            pulse_energy_J=pump_energy, 
            beam_area_cm2=beam_area
        )
    )

    #4) Kinetic rate constants of simulation
    t3s: float = 100000 # ns, S lifetime
    k3s: float = 1e9 / t3s # s⁻¹, S decay rate
    ktet: float = 1.5e9 # M⁻¹ s⁻¹, bimolecular TET rate constant
    k2s: float = 0.0 # noqa M⁻¹ ∙ s⁻¹, S annihilation rate
    t3a: float = 250000 # ns, A lifetime
    k3a: float = 1e9 / t3a # noqa s⁻¹, A decay rate
    kr: float = ktet * conc1 # s⁻¹, A rise rate
    tr: float = 1e9 / kr # ns, A rise lifetime
    # k2: float = 1e-6 # M⁻¹ ∙ s⁻¹, A annihilation rate
    k2: float = 3.5 # M⁻¹ ∙ ns⁻¹, A annihilation rate
    k2: float = k2 * conc2 * pumped # ns⁻¹, effective quadratic rate for S0-normalized populations
    phi_tet: float = ktet * conc1 / (k3s + ktet * conc1) # TET efficiency
    
    #5) Construct TA spectrum
    C_S1: NDArray[np.floating] = TTAKineticModel().solve(times, beta=np.log([t3s, 1e9/(ktet * conc1), t3a, k2]))

    def ann(t: NDArray[np.floating], y: NDArray[np.floating]) -> list[NDArray[np.floating]]:
        S: float
        A: float
        S, A = y
        # d[³S*] / dt = -(k₃ₛ + kₜₑₜ[A]) * [³S*]
        dS: NDArray[np.floating] = -((k3s + ktet * conc1) / 1e9) * S - k2s * S ** 2
        # d[³A*] / dt = kₜₑₜ[A][³S*] - k₃ₐ[³A*] - k₂[³A*]²
        dA: NDArray[np.floating] = ((ktet * conc1) / 1e9) * S - (1.0 / t3a) * A - k2 * A * A
        return [dS, dA]

    sol = solve_ivp(
        ann,
        (times[0], times[-1]),
        y0=[1.0, 0.0],
        t_eval=times,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    C_S: NDArray[np.floating] = np.array([sol.y[0], sol.y[1]]).T

    print(f"Φₜₑₜ: {phi_tet}, τᵣ: {tr}, k₂: {k2}")
    print(f"fₑᵪ: {pumped :.6f}, Cₑᵪ(A): {phi_tet * pumped * conc2 :.6e}, Cₑᵪ(A)ᵒᵇˢ: {np.max(C_S1[:, 0]) * pumped * conc2}")

    ta, truth = simulate_ta(times=times, wavelengths_nm=wavelengths, traces=C_S, spectra_dA=spectra * pumped * conc2,)
    
    #6) Add noise to spectrum
    snr: float = 1000
    sigma: float = np.max(np.abs(ta)) / snr
    rng: np.random.Generator = np.random.default_rng()
    noise: NDArray[np.floating] = rng.normal(0, sigma, size=ta.shape)
    ta += noise

    #7) Guess values for fitting start
    tau_3s: float = 100000
    tau_r: float = 5000
    tau_3a: float = 200000
    k2: float = 2e-6
    # guess: NDArray[np.floating] = np.log([tau_3s, tau_r, tau_3a, k2])

    start_time: int = perf_counter_ns()
    # result = ta.fit_global_kinetics(
    #     kinetic_model=TTAKineticModel(),
    #     beta0=guess,
    #     noise=np.std(noise, axis=0),
    #     lam=1.0e-12,
    #     debug=False,
    #     fix_beta=np.array([True, False, False, False], dtype=np.bool_)
    # )
    # Use the fast approximation to the analytical solution. This retains
    # the original source-driven model, including the initial condition at t=0.
    model = SensitizedTTAModel(
        time_unit="ns",
        evaluation="accelerated",
        interpolation_tolerance=1e-11,
    )

    result = ta.fit_global_kinetics(
        kinetic_model=model,
        beta0=[1 / t3s + 1 / tau_r, 1 / tau_3a, 1e-3],
        noise=np.std(noise, axis=0),
        lam=0.0,
        debug=False,
        fix_beta=np.array([False, False, False], dtype=np.bool_)
    )
    end_time: int = perf_counter_ns()

    print(result.summary(style='journal', digits=3))
    print(f"Fit Performed in {(end_time - start_time) * 1e-9 :.2f} s")

    fit: NDArray[np.floating] = reconstruct_fit(result)

    #8) Plot several traces and the resulting fit
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(hspace=0.2, left=0.125, bottom=0.125, right=0.95, top=0.90))

    trace_wl: range = range(30, 150, 30)
    spectrum_ts: NDArray[np.floating] = np.geomspace(1, 10000, 10, dtype=np.int64, endpoint=False)
    colors1: NDArray[np.floating] = sample_colormap(len(trace_wl), "rainbow_r")
    try:
        # Decay profiles
        for i, w in enumerate(trace_wl):
            axes[0].plot(ta.y, ta[:, w], lw=1.5, color=colors1[i], label=f"{ta.x[w]} nm", alpha=0.8)
            axes[0].plot(fit.y, fit[:, w], lw=1, color='k', ls='--')

        [axes[0].axvline(x=ta.y[t], color='k', lw=0.5, ls='--') for t in spectrum_ts]
        axes[0].set(xscale='log')
        axes[0].legend(frameon=False)

    except IndexError as e:
        print(e)

    colors1: NDArray[np.floating] = sample_colormap(len(spectrum_ts), "rainbow_r")
    # Plot several spectra at various time delays
    try:
        # Spectral profiles
        for i, t in enumerate(spectrum_ts):
            axes[1].plot(ta.x, ta[t, :], lw=1.5, color=colors1[i], label=f"{ta.y[t] :.1f} ns", alpha=0.75)
            axes[1].plot(fit.x, fit[t, :], lw=1, color='k', ls='--')

        [axes[1].axvline(x=ta.x[w], color='k', lw=0.5, ls='--') for w in trace_wl]
        axes[1].legend(ncols=2, frameon=False)

    except IndexError as e:
        print(e)

    for ax in axes:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y * 1e3)))
        ax.set(ylabel="$\\mathrm{\\Delta Abs \\ (mOD)}$")
        # ax.label_outer()
    
    axes[0].set(xlabel="$\\mathrm{Time\\ Delay\\ (ns)}$", xlim=(times[times > 0].min(), times.max()))
    axes[1].set(xlabel="$\\mathrm{Wavelength\\ (nm)}$", xlim=(wavelengths.min(), wavelengths.max()))

    # axes[0].set(title=f"SNR = {np.max(surface) / sigma :.0f}")

    plt.show()


if __name__ == "__main__":
    simulation()

