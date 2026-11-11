"""Reproducible analytical TTA fit with signed, heteroscedastic TA data.

From an installed checkout:
    python examples/sensitized_tta_global_fit.py
    python examples/sensitized_tta_global_fit.py --plot docs/figures/tta-fit.png

Times are seconds. The synthetic sigmas are known, so absolute_sigma=True.
The example does not invoke an ODE integrator or a pump-calibration helper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phoskhemia.data import TransientAbsorption
from phoskhemia.kinetics import SensitizedTTAModel
from phoskhemia.fitting.reconstructions import reconstruct_fit, reconstruct_residuals


def run_example():
    rng = np.random.default_rng(20261019)
    times_s = np.r_[0.0, np.geomspace(1e-9, 1e-2, 180)]
    wavelengths_nm = np.linspace(440, 760, 33)
    model = SensitizedTTAModel(time_unit="s")

    P, beta, gamma, q = 3.0, 6e6, 1e3, 1e9
    truth = np.array([beta, gamma, q * P / beta**2])
    a_A = (8e-4 * np.exp(-((wavelengths_nm - 625) / 65) ** 2)
           - 2e-4 * np.exp(-((wavelengths_nm - 480) / 30) ** 2))
    a_S = (-6e-4 * np.exp(-((wavelengths_nm - 515) / 48) ** 2)
           + 2e-4 * np.exp(-((wavelengths_nm - 695) / 50) ** 2))
    spectra = np.column_stack([a_A, a_S])  # wavelength x species [A, S]
    clean = model.solve(times_s, truth) @ spectra.T
    sigma_lambda = 3e-6 + 1.2e-5 * ((wavelengths_nm - 600) / 160) ** 2
    measured = clean + rng.normal(size=clean.shape) * sigma_lambda[None, :]
    data = TransientAbsorption(measured, x=wavelengths_nm, y=times_s,
                               meta={"noise_t0": sigma_lambda, "synthetic": True})

    result = data.fit_global_kinetics(
        model, beta0=np.array([5e6, 1.2e3, 1e-4]),
        bounds=([1e3, 0.0, 0.0], [1e8, 1e5, 10.0]),
        x_scale=np.array([6e6, 1e3, 1e-4]),
        maxit=200, absolute_sigma=True, ci_level=.95,
        propagate_kinetic_uncertainty=True,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    fitted = reconstruct_fit(result)
    residuals = reconstruct_residuals(result, data)
    summary = {
        "truth": dict(zip(model.param_names(), map(float, truth))),
        "fit": result.kinetics,
        "local_95_percent_intervals": result.kinetics_ci,
        "reduced_chi_square": float(result.diagnostics["chi2_red"]),
        "degrees_of_freedom": float(result.diagnostics["DoF"]),
        "total_fitted_parameters": float(result.diagnostics["n_params"]),
        "shape_time_by_wavelength": list(data.shape),
        "success": bool(result._cache["solver_info"]["success"]),
    }
    return data, fitted, residuals, result, spectra, sigma_lambda, summary


def plot_example(output, data, fitted, residuals, result, spectra, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), layout="constrained")
    fig.suptitle("Analytical sensitization and annihilation: synthetic global fit", fontsize=15)
    chosen = [7, 18, 25]
    colors = ["#1f77b4", "#d95f02", "#5c3a92"]
    for j, color in zip(chosen, colors):
        axes[0, 0].plot(data.y[1:], np.asarray(data)[1:, j] * 1e3, ".", ms=3, alpha=.45, color=color)
        axes[0, 0].plot(data.y[1:], np.asarray(fitted)[1:, j] * 1e3, color=color, lw=1.5,
                        label=f"{data.x[j]:.0f} nm")
        axes[1, 0].plot(data.y[1:], np.asarray(residuals)[1:, j] / sigma[j], ".", ms=3,
                        alpha=.7, color=color)
    axes[0, 0].set(xscale="log", xlabel="Time (s)", ylabel="Signal (mΔA)", title="Measured points and fitted curves")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].axhline(0, color="0.3", lw=.8)
    axes[1, 0].set(xscale="log", xlabel="Time (s)", ylabel="Residual / σ(λ)", title="Residuals retain the measured noise scale")
    for k, label, color in zip(range(2), ["Annihilator", "Sensitizer"], ["#d95f02", "#1f77b4"]):
        axes[0, 1].plot(data.x, spectra[:, k] * 1e3, "--", color=color, alpha=.8, label=f"{label}: true")
        axes[0, 1].plot(data.x, result.amplitudes[:, k] * 1e3, color=color, label=f"{label}: fit")
    axes[0, 1].set(xlabel="Wavelength (nm)", ylabel="Amplitude (mΔA)", title="Signed component spectra")
    axes[0, 1].legend(frameon=False, fontsize=8)
    traces = result.traces
    axes[1, 1].plot(data.y[1:], traces[1:, 0], color="#d95f02", label="Annihilator w(t)")
    axes[1, 1].plot(data.y[1:], traces[1:, 1], color="#1f77b4", label="Sensitizer exp(−βt)")
    axes[1, 1].set(xscale="log", xlabel="Time (s)", ylabel="Normalized population", title="Basis functions evaluated analytically")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="Optional PNG or PDF output path")
    args = parser.parse_args()
    data, fitted, residuals, result, spectra, sigma, summary = run_example()
    print(json.dumps(summary, indent=2))
    if args.plot:
        plot_example(args.plot, data, fitted, residuals, result, spectra, sigma)
