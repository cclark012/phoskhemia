from time import perf_counter_ns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import scipy as sp
from phoskhemia.kinetics import KineticModel
from phoskhemia.fitting.reconstructions import reconstruct_fit
from phoskhemia.data.io import load_mat

def plot_params():
    # rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = 12
    rcParams['lines.linewidth'] = 1
    rcParams['contour.negative_linestyle'] = 'solid'
    rcParams['axes.linewidth'] = 1.5
    rcParams['axes.labelpad'] = 10
    rcParams['axes.titlepad'] = 15
    rcParams['axes.titlesize'] = 14
    # rcParams['axes.titleweight'] = 'bold'
    # rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.labelsize'] = 14
    rcParams['legend.frameon'] = False
    rcParams['legend.title_fontsize'] = 12
    rcParams['legend.fontsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['xtick.major.size'] = 5
    rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.size'] = 3
    rcParams['xtick.minor.width'] = 1
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.labelsize'] = 10
    rcParams['ytick.major.size'] = 5
    rcParams['ytick.major.width'] = 2
    rcParams['ytick.minor.size'] = 3
    rcParams['ytick.minor.width'] = 1
    rcParams['ytick.minor.visible'] = True
    rcParams['savefig.dpi'] = 1200
    rcParams['savefig.format'] = "png"
    rcParams['figure.figsize'] = (6, 4)
    rcParams['figure.titlesize'] = 14
    # rcParams['figure.titleweight'] = 'bold'
    # rcParams['figure.labelweight'] = 'bold'
    rcParams['figure.subplot.wspace'] = 0.10
    rcParams['figure.subplot.hspace'] = 0.10
    rcParams['figure.subplot.left'] = 0.10
    rcParams['figure.subplot.right'] = 0.90
    rcParams['figure.subplot.bottom'] = 0.10
    rcParams['figure.subplot.top'] = 0.90

plot_params()


def analysis():
    @njit(fastmath=True, parallel=True)
    def downsample2D(array: np.typing.ArrayLike, start: int, smoothx: int=1, smoothy: int=1) -> np.typing.ArrayLike:
        newy = np.zeros_like(array[start::(2 * smoothx), smoothy:-smoothy])
        for i in prange(1, len(array[start:-smoothx:(2 * smoothx), 0]) - 1):
            for j in prange(smoothy, len(array[0, :]) - smoothy):
                newy[0, j] = np.sum(array[start:smoothx, j - smoothy:j + smoothy + 1]) / (smoothx * (2 * smoothy + 1))
                newy[-1, j] = np.sum(array[-smoothx:, j - smoothy:j + smoothy + 1]) / (smoothx * (2 * smoothy + 1))
                x_ind = 2 * smoothx * i + start
                newy[i, j] = np.sum(array[x_ind - smoothx:x_ind + smoothx + 1, j - smoothy:j + smoothy + 1]) / ((2 * smoothx + 1) * (2 * smoothy + 1))

        return newy

    @njit(fastmath=True, parallel=True)
    def downsample2D_noy(array: np.typing.ArrayLike, start: int, smoothx: int=1) -> np.typing.ArrayLike:
        newy = np.zeros_like(array[start::(2 * smoothx), :])
        for j in prange(0, len(array[0, :])):
            newy[0, j] = np.mean(array[start:start + smoothx, j])
            newy[-1, j] = np.mean(array[-smoothx:, j])
            for i in prange(1, len(array[start:-smoothx:(2 * smoothx), 0]) - 1):
                x_ind = 2 * smoothx * i + start
                newy[i, j] = np.mean(array[x_ind - smoothx:x_ind + smoothx + 1, j])
        return newy

    class ExponentialModel(KineticModel):
        """
        Single exponential model.
        """
        def n_params(self):
            return 2

        def param_names(self):
            return ["τ", "β"]

        def species_names(self):
            return "*A"

        def solve(self, times, beta):
            tau, b = np.exp(beta)
            # tau, b = beta
            # tau = np.exp(log_tau)
            a_star = np.exp(-times / tau) + b
            return np.atleast_2d(a_star).T
        
        def parameterization(self):
            return "log"

    class HTTAModel(KineticModel):
        """
        Homomolecular Triplet-Triplet Annihilation model (no sensitizer).
        """
        def n_params(self):
            return 2

        def param_names(self):
            return ["τ", "β"]

        def species_names(self):
            return "[³S*]"

        def solve(self, times, beta):
            tau, b = np.exp(beta)
            s_star = (1 - b) / (np.exp(times / tau) - b)
            return s_star[:, None]

        def parameterization(self):
            return "log"
    
    class TTAKineticModel(KineticModel):
        """
        Triplet–triplet annihilation kinetic model.
        """

        def n_params(self):
            return 3

        def param_names(self):
            return ["τᵣ", "τ₃ₐ", "k₂"]

        def species_names(self):
            return ["[³A*]", "[³S*]"]

        def solve(self, times, beta):
            log_tr, log_t3a, log_kappa2 = beta
            tr = np.exp(log_tr)
            t3a = np.exp(log_t3a)
            kappa2 = np.exp(log_kappa2)

            times = np.asarray(times, dtype=float)

            # Sensitizer population
            C_S = np.exp(-times / tr)

            def rhs(t, y):
                A = y[0]
                S = np.exp(-t / tr)
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
        
    class TETKineticModel(KineticModel):
        """
        Triplet Energy Transfer kinetic model.
        """

        def n_params(self):
            return 5

        def param_names(self):
            return ["τ₃ₛ", "τₜₑₜ", "k₂ₛ", "τ₃ₐ", "k₂"]

        def species_names(self):
            return ["[³A*]", "[³S*]"]

        def solve(self, times, beta):
            t3s, ttet, k2s, t3a, k2 = np.exp(beta)

            times = np.asarray(times, dtype=float)

            # Sensitizer population

            def ann(t, y):
                S, A = y
                dS = -(1 / (t3s + ttet)) * S - k2s * S * S 
                dA = (1.0 / ttet) * S - (1.0 / t3a) * A - k2 * A * A
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

            if not sol.success:
                raise RuntimeError("ODE solver failed")

            C_S = sol.y[0]
            C_A = sol.y[1]

            return np.column_stack([C_A, C_S])
    
    ta_file2 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_545_800nm_0-3.mat"
    ta_file = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/012926/TA PtOEP_370-520nm.mat"
    ta_file = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/102924/TIPSAn_300uM_PtOEP_6-6uM_532exc_200hz_10wp_0-5nd_1048576ns_9-6nmres_2-5nmstep_400-800nm.mat"
    ta_file1 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"
    ta_file1 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/PtOEPalone/09082025_PtOEP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"

    start_load = perf_counter_ns()
    
    # array1 = sio.loadmat(ta_file1)["data"]
    # array2 = sio.loadmat(ta_file2)["data"]
    array = load_mat(ta_file1, gui=True)

    end_load = perf_counter_ns()
    print(f"Data load time: {(end_load - start_load) / 1e9} s")

    start_smooth = perf_counter_ns()
    noise = np.std(array[:np.argmin(np.abs(array.y)), :], axis=0)
    arr = array.time_zero()
    # arr = array.downsample_time(method='log', aggregate='mean', n_log=10000)
    signal = np.max(np.abs(array.smooth((65, 1))), axis=0)

    end_smooth = perf_counter_ns()
    [print(f"SNR({array.x[i]}) = {float(signal[i] / noise[i]) :.3f}") for i in range(len(array.x))]
    print(f"Data smooth time: {(end_smooth - start_smooth) / 1e9} s")
    print(f"Shape: {arr.shape}, Size: {arr.x.shape[0] * arr.y.shape[0]}")

    tail_frac = int(arr.shape[0] * (1 - 0.2))
    noise = np.std(arr[tail_frac:], axis=0)
    running_noise = np.median([np.std(arr[tail_frac+i:tail_frac+i+10], axis=0, ddof=1) for i in range(0, len(arr[tail_frac::]), 5)], axis=0)
    # print(np.asarray(running_noise).shape)
    # [print(f"Wavelength: {wavelengths[i] :.0f}, Noise: {noise[i] :.4e}, Noise ± St. Dev.: {np.mean(running_noise[:, i]) :.4e} ± {np.std(running_noise[:, i]) :.4e}, RSD: {100 * (np.std(running_noise[:, i]) / np.mean(running_noise[:, i])) :.3f}%") for i in range(len(wavelengths))]
    do_fit      = True

    if do_fit:
        tau_3s = 2000
        tau_tet = 10000
        k2s = 1e-6
        tau_3a = 800000
        k2 = 1e-6
        tau = 90000
        b = 0.5
        # guess = np.log([tau_3s, tau_3a, k2])
        # guess = np.log([tau_3s, tau_tet, k2s, tau_3a, k2])
        guess = np.log([tau, b])

        start_time = perf_counter_ns()
        result = arr.fit_global_kinetics(
            kinetic_model=HTTAModel(),
            beta0=guess,
            noise=running_noise,
            lam=1.0e-12,
            debug=False,
        )
        end_time = perf_counter_ns()

        print(result.summary(style='journal', digits=6))
        print(f"Fit Performed in {(end_time - start_time) * 1e-9 :.2f} s")

        fit = reconstruct_fit(result)

    time_slices = [10, 100, 1000, 10000, 100000]
    wavelengths = [420, 440, 492.5]

    idw = [np.argmin(np.abs(arr.x - w)) for w in wavelengths]
    idt = [np.argmin(np.abs(arr.y - t)) for t in time_slices]

    fig = plt.figure(figsize=(8, 6))
    subfig = fig.subfigures(1, 1)
    gs = subfig.add_gridspec(2, 1, hspace=0.2, left=0.125, bottom=0.125, right=0.95, top=0.90)
    axes = gs.subplots()

    cmap1 = mpl.colormaps['rainbow_r']
    colors1 = cmap1(np.linspace(0, 1.0, len(idw)))
    cmap2 = mpl.colormaps['jet_r']
    colors2 = cmap2(np.linspace(0, 1.0, len(idw)))
    cmap3 = mpl.colormaps['rainbow_r']
    colors3 = cmap3(np.linspace(0, 1.0, len(idt)))

    [axes[0].plot(arr.y, arr[:, i], color=c, lw=2.5, alpha=0.25, label=f"{arr.x[i]} nm") for i, c in zip(idw, colors1)]
    [axes[1].plot(arr.x, arr[t, :], lw=2.5, color=c, alpha=0.5, label=f"{arr.y[t] :.1f} ns") for t, c in zip(idt, colors3)]
    if do_fit:
        [axes[0].plot(fit.y, fit[:, i], color=c, lw=2, ls='--', alpha=0.75, label=f"Fit {arr.x[i]} nm") for i, c in zip(idw, colors2)]
        [axes[1].plot(arr.x, result.amplitudes[:, amp] * np.sum(result.traces[idt[0], :]), color='k', lw=1, ls='--') for amp in range(result.amplitudes.shape[1])]

    axes[0].set(xlim=(np.min(arr.y[1:]), np.max(arr.y)), xscale="log")
    axes[1].set(xlim=(np.min(arr.x), np.max(arr.x)))
    axes[0].legend()
    axes[1].legend(loc="lower right")
    
    plt.show()
