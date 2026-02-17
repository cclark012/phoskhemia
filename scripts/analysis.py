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

class SiNDIKineticModel(KineticModel):
    """
    Triplet Energy Transfer kinetic model.
    """

    def n_params(self):
        return 5

    def parameterization(self):
        return 'linear'

    def param_names(self):
        return ["τ",  "τₜₑₜ", "τ₁", "τ₂", "β"]

    def species_names(self):
        # return ["[³NDI*]", "[³Si*]"]
        return ["[³NDI*]"]

    def solve(self, times, beta):
        tau, tau_tet, tau1, tau2, b = beta
        k, k_tet, k1, k2 = 1 / tau, 1 / tau_tet, 1 / tau1, 1 / tau2
        C_Si = b * np.exp(-times * (k_tet + k1)) + (1 - b) * np.exp(-times * (k_tet + k2))
        C_NDI = (
            (k_tet / (k_tet + k1 - k)) * (np.exp(-times * k) - np.exp(-times * (k_tet + k1)))
            + (k_tet / (k_tet + k2 - k)) * (np.exp(-times * k) - np.exp(-times * (k_tet + k2)))
        )
        return np.column_stack([C_NDI, C_Si])
        # return C_NDI[:, None]

        # Sensitizer population

        # def ann(t, y):
        #     S, A = y
        #     dS = -(1 / (t3s + ttet)) * S - k2s * S * S 
        #     dA = (1.0 / ttet) * S - (1.0 / t3a) * A - k2 * A * A
        #     return [dS, dA]

        # sol = sp.integrate.solve_ivp(
        #     ann,
        #     (times[0], times[-1]),
        #     y0=[1.0, 0.0],
        #     t_eval=times,
        #     method="LSODA",
        #     rtol=1e-6,
        #     atol=1e-9,
        # )

        # if not sol.success:
        #     raise RuntimeError("ODE solver failed")

        # C_S = sol.y[0]
        # C_A = sol.y[1]

        # return np.column_stack([C_A, C_S])

class SiNDIKineticModel(KineticModel):
    """
    Triplet Energy Transfer kinetic model.
    """

    def n_params(self):
        return 2

    def param_names(self):
        return ["τ",  "τₜₑₜ"]

    def species_names(self):
        return ["[³NDI*]", "[³Si*]"]

    def solve(self, times, beta):
        tau, tau_tet = np.exp(beta)
        # k, k_tet = 1 / tau, 1 / tau_tet, 
        C_Si = np.exp(-times / (tau_tet))
        C_NDI = (np.exp(-times / tau) - np.exp(-times / (tau_tet)))

        return np.column_stack([C_NDI, C_Si])
        # return C_NDI[:, None]

    def parameterization(self):
        return 'log'



def analysis():
    ta_file2 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_545_800nm_0-3.mat"
    ta_file = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/102924/TIPSAn_300uM_PtOEP_6-6uM_532exc_200hz_10wp_0-5nd_1048576ns_9-6nmres_2-5nmstep_400-800nm.mat"
    ta_file1 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/091824/p26A_120uM_PtOEP_20uM_532exc_200hz_10wp_0-5nd_1048576ns_9-6nmres_2-5nmstep_400-800nm.mat"
    ta_file2 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-Dodecane/B637_Si-dod-385-520nm.mat"
    ta_file3 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-Dodecane/B637_Si-dod_545-1200nm.mat"
    ta_file4 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Small Si QD_high loading/Si-NDI_smaller size_385-520nm_524us.mat"
    ta_file5 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Small Si QD_high loading/Si-NDI_smaller size_545-1200nm_524us.mat"
    ta_file6 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_low loading_Big Si QD/TA/B637_SiNDI_100ug_385-520nm.mat"
    ta_file7 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_low loading_Big Si QD/TA/B637_SiNDI_100ug_545-1200nm.mat"
    ta_file8 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Big Si QD_high loading/Si-NDI_bigger size_385-520nm_524us.mat"
    ta_file9 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Big Si QD_high loading/Si-NDI_bigger size_545-1200nm_524us.mat"
    ta_file10 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/090524/p27A_120uM_PtOEP_6-6uM_532exc_200hz_0wp_262144ns_9-6nmres_2-5nmstep_400-700nm.mat"
    ta_file11 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/012926/TA PtOEP_370-520nm.mat"
    ta_file12 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/PtOEPalone/09082025_PtOEP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"
    ta_file13 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"

    start_load = perf_counter_ns()
    
    # array1 = sio.loadmat(ta_file1)["data"]
    # array2 = sio.loadmat(ta_file2)["data"]
    array1 = load_mat(ta_file8, gui=True)
    array2 = load_mat(ta_file9, gui=True)
    array = array1 + array2

    end_load = perf_counter_ns()
    print(f"Data load time: {(end_load - start_load) / 1e9} s")

    start_smooth = perf_counter_ns()
    arr = array.time_zero()
    noise = arr.meta["noise_t0"]
    print(noise[0])
    tail_frac = int(0.2 * len(arr.y))
    arr = arr.downsample_time(method='linear', stride=32, aggregate="mean")
    # noise = arr.meta["noise_t0"]
    # print(noise[0])
    # arr = array.downsample_time(method='log', aggregate='mean', n_log=10000)
    # arr = arr.svd_denoise(method='e15', noise=None, weight='column')
    # noise = np.std(arr[-tail_frac:, :], axis=0)
    # signal = np.max(np.abs(arr), axis=0)
    # mean: NDArray[np.floating] = arr.mean(axis=0, keepdims=True)
    # D0: NDArray[np.floating] = arr - mean
    # arr = D0 / noise[None, :]


    end_smooth = perf_counter_ns()
    # [print(f"SNR({array.x[i]}) = {float(signal[i] / noise[i]) :.3f}") for i in range(len(array.x))]
    print(f"Data smooth time: {(end_smooth - start_smooth) / 1e9} s")
    print(f"Shape: {arr.shape}, Size: {arr.x.shape[0] * arr.y.shape[0]}")
    # residuals = array.time_zero() - arr

    # U, S, V = sp.linalg.svd(arr, full_matrices=False, lapack_driver='gesvd')
    # plt.scatter(np.arange(len(S)), S)
    # plt.plot(np.arange(len(S)), V[0, :] * noise[0] + mean[0, 0])
    # plt.plot(np.arange(len(S)), V[1, :] * noise[1] + mean[0, 1])
    # plt.yscale('log')
    # plt.show()

    # tail_frac = int(arr.shape[0] * (1 - 0.2))
    # noise = np.std(arr[tail_frac:], axis=0)
    # running_noise = np.median([np.std(arr[tail_frac+i:tail_frac+i+10], axis=0, ddof=1) for i in range(0, len(arr[tail_frac::]), 5)], axis=0)
    # print(np.asarray(running_noise).shape)
    # [print(f"Wavelength: {wavelengths[i] :.0f}, Noise: {noise[i] :.4e}, Noise ± St. Dev.: {np.mean(running_noise[:, i]) :.4e} ± {np.std(running_noise[:, i]) :.4e}, RSD: {100 * (np.std(running_noise[:, i]) / np.mean(running_noise[:, i])) :.3f}%") for i in range(len(wavelengths))]
    # [print(f"{arr.x[i]} nm: {np.mean(residuals[:, i]) * 1e6 :>6.3f} μOD ± {np.std(residuals[:, i]) * 1e6 :<6.3f} μOD, {sp.stats.skew(residuals[:, i]) * 1e3 :>6.3f}, {sp.stats.kurtosis(residuals[:, i]) * 1e3 :>6.3f}") for i in range(len(arr.x))]

    # nobs, minmax, mean, variance, skew, kurt = sp.stats.describe(residuals[:, 5])
    # minimum, maximum = minmax
    # distx = np.linspace(minimum, maximum, 1000)
    # disty = (1 / np.sqrt((2 * np.pi * variance))) * np.exp(-np.square((distx - mean)) / (2 * variance))
    # top = np.max(disty)

    # plt.hist(residuals[:, 5], bins=100, density=True, histtype='step')
    # plt.plot(distx, disty, color='k', lw=1)
    # plt.show()

    do_fit      = False

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
            noise=noise,
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
    [axes[1].plot(arr.x, np.mean(arr[t:t+20, :], axis=0), lw=2.5, color=c, alpha=0.5, label=f"{arr.y[t] :.1f} ns") for t, c in zip(idt, colors3)]
    if do_fit:
        [axes[0].plot(fit.y, fit[:, i], color=c, lw=2, ls='--', alpha=0.75, label=f"Fit {arr.x[i]} nm") for i, c in zip(idw, colors2)]
        [axes[1].plot(arr.x, result.amplitudes[:, amp] * np.sum(result.traces[idt[0], :]), color='k', lw=1, ls='--') for amp in range(result.amplitudes.shape[1])]

    axes[0].set(xlim=(np.min(arr.y[1:]), np.max(arr.y)), xscale="log")
    axes[1].set(xlim=(np.min(arr.x), np.max(arr.x)))
    axes[0].legend()
    axes[1].legend(loc="lower right")
    
    plt.show()


def analysis2():
    ta_file2 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_545_800nm_0-3.mat"
    ta_file = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/102924/TIPSAn_300uM_PtOEP_6-6uM_532exc_200hz_10wp_0-5nd_1048576ns_9-6nmres_2-5nmstep_400-800nm.mat"
    ta_file1 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/091824/p26A_120uM_PtOEP_20uM_532exc_200hz_10wp_0-5nd_1048576ns_9-6nmres_2-5nmstep_400-800nm.mat"
    ta_file2 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-Dodecane/B637_Si-dod-385-520nm.mat"
    ta_file3 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-Dodecane/B637_Si-dod_545-1200nm.mat"
    ta_file4 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Small Si QD_high loading/Si-NDI_smaller size_385-520nm_524us.mat"
    ta_file5 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Small Si QD_high loading/Si-NDI_smaller size_545-1200nm_524us.mat"
    ta_file6 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_low loading_Big Si QD/TA/B637_SiNDI_100ug_385-520nm.mat"
    ta_file7 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_low loading_Big Si QD/TA/B637_SiNDI_100ug_545-1200nm.mat"
    ta_file8 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Big Si QD_high loading/Si-NDI_bigger size_385-520nm_524us.mat"
    ta_file9 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/Si-NDI/Si-NDI_Big Si QD_high loading/Si-NDI_bigger size_545-1200nm_524us.mat"
    ta_file10 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2024/TA_Data/090524/p27A_120uM_PtOEP_6-6uM_532exc_200hz_0wp_262144ns_9-6nmres_2-5nmstep_400-700nm.mat"
    ta_file11 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TA_Data/012926/TA PtOEP_370-520nm.mat"
    ta_file12 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/PtOEPalone/09082025_PtOEP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"
    ta_file13 = "C:/Users/colec/Documents/.Most_Used/Lab_Work/2026/TAsNH/8CPP/05212025_PtOEPw8CPP_532ex_200Hz_100shot_10scan_262144ns_8bit_400-520nm_0-3.mat"

    start_load = perf_counter_ns()
    plt.style.use('./research_style.mplstyle')

    array1 = load_mat(ta_file2, gui=True)
    array2 = load_mat(ta_file3, gui=True)
    array = array1 + array2

    end_load = perf_counter_ns()
    print(f"Data load time: {(end_load - start_load) / 1e9} s")
    start_smooth = perf_counter_ns()

    arr = array.time_zero()
    # arr, info = arr.svd_denoise(method='e15', noise=None, weight='column', value_rotation='include', return_details=True)
    # print(info["r_min"])
    arr = arr.downsample_time(method='linear', stride=32, aggregate='mean')

    end_smooth = perf_counter_ns()
    print(f"Data smooth time: {(end_smooth - start_smooth) / 1e9} s")
    print(f"Shape: {arr.shape}, Size: {arr.x.shape[0] * arr.y.shape[0]}")

    from phoskhemia.kinetics.models import BiexponentialModel, TriexponentialModel, ExponentialModel
    from phoskhemia.fitting.reconstructions import reconstruct_fit
    
    # result = arr.fit_global_kinetics(
    #     kinetic_model=(SiNDIKineticModel()),
    #     beta0=np.log([1300, 130000]),
    #     noise=None
    # )
    # fit = reconstruct_fit(result=result)
    # print(result.summary(style="technical"))

    cmap = mpl.colormaps['bwr_r']
    colors = cmap(np.linspace(0, 1.0, 4))

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(arr.y, arr.trace(440), color=colors[0], lw=1.5, alpha=0.5, label="440 nm")
    # ax.plot(arr.y, arr.trace(480), color=colors[1], lw=1.5, alpha=0.75, label="480 nm")
    # ax.plot(arr.y, arr.trace(600), color=colors[2], lw=1.5, alpha=0.75, label="600 nm")
    # ax.plot(arr.y, arr.trace(760), color=colors[3], lw=1.5, alpha=0.5, label="760 nm")
    # ax.set(xscale='log')
    # ax.set(xlabel="Time Delay (ns)", ylabel=r"$\Delta A \: (mOD)$")
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y * 1e3)))
    # ax.legend()
    # plt.plot(fit.y, fit.trace(440), color='k', lw=.5)
    # plt.plot(fit.y, fit.trace(480), color='k', lw=.5)
    # plt.plot(fit.y, fit.trace(620), color='k', lw=.5)
    # plt.plot(fit.y, fit.trace(760), color='k', lw=.5)
    # plt.show()
    # [plt.plot(arr.x, result.amplitudes[:, i]) for i in range(result.amplitudes.shape[1])]
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_spectrum(arr, 10, aggregate=10, ax=ax, label="10 ns", color=colors[0])
    plot_spectrum(arr, 100, aggregate=100, ax=ax, label="100 ns", color=colors[1])
    plot_spectrum(arr, 1000, aggregate=1000, ax=ax, label="1000 ns", color=colors[2])
    plot_spectrum(arr, 10000, aggregate=10000, ax=ax, label="10000 ns", color=colors[3])
    # plot_spectrum(fit, 0, ax=ax, label="Fit Amplitudes", color='k', lw=1)
    # max_amp = np.max(fit.spectrum(0))
    # ax.plot(fit.x, result.amplitudes[:, 0] * (max_amp / np.max(result.amplitudes[:, 0])), color='k')
    # ax.plot(fit.x, result.amplitudes[:, 1] * (max_amp / np.max(result.amplitudes[:, 1])), color='k', ls='--')
    ax.legend()
    # print(result.amplitudes.shape)
    # plot_spectrum(fit, 10, ax=ax, label="10 ns")
    # plot_spectrum(fit, 100, ax=ax, label="100 ns")
    # plot_spectrum(fit, 1000, ax=ax, label="1000 ns")
    # plot_spectrum(fit, 10000, ax=ax, label="10000 ns")
    ax.set(xlabel="Wavelength (nm)", ylabel=r"$\Delta A \: (mOD)$", xlim=(400, 1200))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y * 1e3)))
    ax.set(ylim=(-0.001, 0.0025))
    ax.annotate(r"$Ex = 532 nm$", 
        xy=(532, 0),
        xycoords='data', 
        xytext=(0., -4), 
        textcoords='offset fontsize', 
        verticalalignment='top', 
        horizontalalignment='center', 
        color='k',
        weight='bold',
        size=12,
        arrowprops={
            "arrowstyle": "simple, tail_width=0.05, head_width=0.25, head_length=0.5", 
            "facecolor": "k", 
            "edgecolor": "k",
            "shrinkA": 0.5,
            "shrinkB": 2.5,
        }
    )
    plt.show()

if __name__ == "__main__":
    # analysis()
    from phoskhemia.visualization.plotting import *
    analysis2()
    # asdffasdf
