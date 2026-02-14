import scipy as sp
import numpy as np
from numpy.typing import NDArray

from phoskhemia.simulation.transient_absorption import excitation_fraction, PumpSpec, simulate_ta
from phoskhemia.kinetics import KineticModel

class TTAKineticModel(KineticModel):
    """
    Triplet–triplet annihilation kinetic model.
    """

    def n_params(self):
        return 4

    def param_names(self):
        return ["τᵣ", "τ₃ₛ", "τ₃ₐ", "k₂"]

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

        sol = sp.integrate.solve_ivp(
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

def run():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from phoskhemia.simulation.absorption import dho_absorption
    from phoskhemia.fitting.reconstructions import reconstruct_fit
    from time import perf_counter_ns
    # times = np.arange(0, 1e6, 0.1)
    # times = np.logspace(-9, -3, 10000)
    times = np.linspace(1, 1e6, 50000)
    wavelengths = np.arange(350, 650, 2)
    
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

    gsb1 = dho_absorption(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=1.0,
            lam00_nm=470,
            effective_freq_wn=1500,
            sigma_wn=520,
            )
    esa1 = dho_absorption(
        wavelengths_nm=wavelengths, 
        huang_rhys_factor=0.9,
        lam00_nm=520,
        effective_freq_wn=2050,
        sigma_wn=900,
    )
    gsb2 = dho_absorption(
            wavelengths_nm=wavelengths, 
            huang_rhys_factor=0.3,
            lam00_nm=550,
            effective_freq_wn=750,
            sigma_wn=300,
            )
    esa2 = dho_absorption(
        wavelengths_nm=wavelengths, 
        huang_rhys_factor=2.2,
        lam00_nm=670,
        effective_freq_wn=950,
        sigma_wn=520,
    )

    spectrum1 = -1.0 * gsb1 + 0.4 * esa1
    spectrum2 = -1 * gsb2 + 0.2 * esa2
    scale_factor1 = 44000 / np.max(gsb1)
    gsb1 = gsb1 * scale_factor1
    esa1 = esa1 * 0.4 * scale_factor1
    spectrum1 = spectrum1 * scale_factor1
    scale_factor2 = (98500 / np.max(gsb2))
    gsb2 = gsb2 * scale_factor2
    esa2 = 0.2 * esa2 * scale_factor2
    spectrum2 = spectrum2 * scale_factor2
    spectra = np.vstack((spectrum2, spectrum1))

    conc1 = 1.e-4
    conc2 = 1.e-5
    path = 0.1
    ground_state = path * (conc1 * gsb1 + conc2 * gsb2)

    eps = gsb2[np.argmin(np.abs(wavelengths - 532))]
    pumped = excitation_fraction(eps_pump=eps, conc_M=conc2, pump=PumpSpec(lambda_pump_nm=532, pathlength_cm=0.1, pulse_energy_J=1.e-3, beam_area_cm2=1.0))

    t3s = 100000 # ns, S lifetime
    k3s = 1e9 / t3s # s⁻¹, S decay rate
    ktet = 1.5e9 # s⁻¹, TET rate
    k2s = 2e-9 # M⁻¹ ∙ s⁻¹, S annihilation rate
    t3a = 250000 # ns, A lifetime
    k3a = 1e9 / t3a # s⁻¹, A decay rate
    kr = ktet * conc1 # s⁻¹, A rise rate
    tr = 1e9 / kr # ns, A rise lifetime
    k2 = 3.5 # M⁻¹ ∙ s⁻¹, A annihilation rate
    k2 = k2 * conc2 * pumped # M⁻¹ ∙ s⁻¹, A apparent annihilation rate
    phi_tet = ktet * conc1 / (k3s + ktet * conc1) # TET efficiency
    
    C_S1 = TTAKineticModel().solve(times, beta=np.log([t3s, 1e9/(ktet * conc1), t3a, k2]))

    def ann(t, y):
        S, A = y
        # d[³S*] / dt = -(k₃ₛ + kₜₑₜ[A]) * [³S*]
        dS = -((k3s + ktet * conc1) / 1e9) * S 
        # d[³A*] / dt = kₜₑₜ[A][³S*] - k₃ₐ[³A*] - k₂[³A*]²
        dA = ((ktet * conc1) / 1e9) * S - (1.0 / t3a) * A - k2 * A * A
        return [dS, dA]

    sol = sp.integrate.solve_ivp(
        ann,
        (times[0], times[-1]),
        y0=[1.0, 0.0],
        t_eval=times,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
            )
    C_S = np.array([sol.y[0], sol.y[1]]).T
    print(f"Φₜₑₜ: {phi_tet}, τᵣ: {tr}, k₂: {k2}")
    print(f"fₑᵪ: {pumped :.6f}, Cₑᵪ(A): {phi_tet * pumped * conc2 :.6e}, Cₑᵪ(A)ᵒᵇˢ: {np.max(C_S1[:, 0]) * pumped * conc2}")

    # plt.plot(times, C_S[:, 0], color='tab:red')
    # plt.plot(times, C_S[:, 1], color='tab:blue')
    # plt.plot(times, C_S1[:, 0], color='k', lw=1, ls='--')
    # plt.plot(times, C_S1[:, 1], color='k', lw=1, ls='--')
    
    # plt.xscale('log')
    # plt.show()
    # return
    ta, truth = simulate_ta(times=times, wavelengths_nm=wavelengths, traces=C_S, spectra_dA=spectra * pumped * conc2,)
    snr = 250
    sigma = np.max(np.abs(ta)) / snr
    rng = np.random.default_rng()
    noise = rng.normal(0, sigma, size=ta.shape)
    ta = ta + noise

    tau_3s = 120000
    tau_r = 500
    tau_3a = 200000
    k2 = 1e-6
    guess = np.log([tau_3s, tau_r, tau_3a, k2])

    start_time = perf_counter_ns()
    result = ta.fit_global_kinetics(
        kinetic_model=TTAKineticModel(),
        beta0=guess,
        noise=np.std(noise, axis=0),
        lam=1.0e-12,
        debug=False,
    )
    end_time = perf_counter_ns()

    print(result.summary(style='journal', digits=3))
    print(f"Fit Performed in {(end_time - start_time) * 1e-9 :.2f} s")

    fit = reconstruct_fit(result)

    fig = plt.figure(figsize=(12, 8))
    subfig = fig.subfigures(1, 1)
    gs = subfig.add_gridspec(2, 1, hspace=0.2, left=0.125, bottom=0.125, right=0.95, top=0.90)
    axes = gs.subplots()
    
    cmap1 = mpl.colormaps['rainbow_r']
    colors1 = cmap1(np.linspace(0, 1.0, 10))
    
    [axes[0].plot(ta.y, ta[:, w], lw=1.5, color=colors1[w // 15], label=f"{ta.x[w]} nm") for w in range(30, 150, 30)]
    [axes[0].plot(fit.y, fit[:, w], lw=1, color='k', ls='--') for w in range(30, 150, 30)]
    axes[0].set(xscale='log')
    [axes[0].axvline(x=ta.y[t], color='k', lw=0.5, ls='--') for t in range(0, 10000, 1000)]
    axes[0].legend()
    [axes[1].plot(ta.x, ta[t, :], lw=1.5, color=colors1[t // 1000], label=f"{ta.y[t] :.1f} ns") for t in range(0, 10000, 1000)]
    [axes[1].plot(fit.x, fit[t, :], lw=1, color='k', ls='--') for t in range(0, 10000, 1000)]
    [axes[1].axvline(x=ta.x[w], color='k', lw=0.5, ls='--') for w in range(30, 150, 30)]
    axes[1].legend()

    # cmap1 = mpl.colormaps['rainbow_r']
    # colors1 = cmap1(np.linspace(0, 1.0, len(idw)))
    # cmap2 = mpl.colormaps['jet_r']
    # colors2 = cmap2(np.linspace(0, 1.0, len(idw)))
    # cmap3 = mpl.colormaps['rainbow_r']
    # colors3 = cmap3(np.linspace(0, 1.0, len(idt)))

    # [axes[0].plot(times, arr[:, i], color=c, lw=2.5, alpha=0.25, label=f"{wavelength[i]} nm") for i, c in zip(idw, colors1)]
    # [axes[0].plot(times, decay * (surface[0, i] / np.max(decay)), color='k', lw=0.5, ls='-', label=f"True {wavelength[i]} nm") for i in idw]
    # [axes[1].plot(wavelength, arr[t, :], lw=2.5, color=c, alpha=0.5, label=f"{times[t] :.1f} ns") for t, c in zip(idt, colors3)]
    # [axes[1].plot(wavelength, surface[t, :], lw=0.5, color='k', ls='-') for t in idt]
    # if do_fit:
    #     [axes[0].plot(fit.y, fit[:, i], color=c, lw=2, ls='--', alpha=0.75, label=f"Fit {wavelength[i]} nm") for i, c in zip(idw, colors2)]
    #     [axes[1].plot(wavelength, result.amplitudes[:, amp] * np.sum(result.traces[idt[0], :]), color='k', lw=1, ls='--') for amp in range(result.amplitudes.shape[1])]

    # axes[0].set(xlim=(dt, t_max), xscale="log")
    # axes[1].set(xlim=(np.min(wavelength), np.max(wavelength)))
    # axes[0].legend()
    # axes[1].legend(loc="lower right")
    # axes[0].set(title=f"SNR = {np.max(surface) / sigma :.0f}")

    plt.show()

if __name__ == "__main__":
    run()

