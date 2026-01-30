import numpy as np
from numpy.typing import NDArray
from scipy import odr

from phoskhemia.kinetics.base import KineticModel
from phoskhemia.fitting.projections import project_amplitudes
from phoskhemia.fitting.validation import compute_diagnostics
from phoskhemia.fitting.projections import propagate_kinetic_covariance
from phoskhemia.data.spectrum_handlers import TransientAbsorption

def fit_global_kinetics(
        self: TransientAbsorption,
        kinetic_model: KineticModel,
        beta0: NDArray[np.floating],
        *,
        noise: NDArray[np.floating] | float | None = None,
        lam: float = 1e-12,
        propagate_kinetic_uncertainty: bool = False,
        debug: bool = False,
    ) -> dict:
    """
    Perform a global kinetic fit using variable projection.

    Parameters
    ----------
    kinetic_model : KineticModel
        Kinetic model instance
    beta0 : ndarray
        Initial guesses for kinetic parameters (log-space)
    noise : ndarray or None
        Per-wavelength noise σ(λ)
    lam : float
        Tikhonov regularization strength
    propagate_kinetic_uncertainty : bool
        Propagate kinetic covariance into amplitude uncertainties
    debug : bool
        Print diagnostic messages on failures

    Returns
    -------
    result : dict
        Structured fit result
    """

    # Input preparation
    data: NDArray[np.floating] = np.asarray(self, dtype=float)
    times: NDArray[np.floating] = np.asarray(self.y, dtype=float)
    wl: NDArray[np.floating] = np.asarray(self.x, dtype=float)
    beta0: NDArray[np.floating] = np.asarray(beta0, dtype=float)

    n_times: int
    n_wl: int
    n_times, n_wl = data.shape

    if beta0.size != kinetic_model.n_params():
        raise ValueError("beta0 length does not match kinetic model")

    # Default to unity noise if none provided
    if noise is None:
        noise: NDArray[np.floating] = np.ones(n_wl, dtype=float)
    elif isinstance(noise, float | int):
        noise: NDArray[np.floating] = noise * np.ones(n_wl, dtype=float)
    else:
        noise: NDArray[np.floating] = np.asarray(noise, dtype=float)
        if noise.size != n_wl:
            raise ValueError("noise length must match number of wavelengths")

    # Flatten observed data (wavelength-major)
    y_obs: NDArray[np.floating] = data.T.ravel()
    t_flat: NDArray[np.floating] = np.tile(times, n_wl)

    # ODR model (kinetics only)
    def odr_model(beta, t):
        try:
            traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
            fit: NDArray[np.floating] = np.empty((n_times, n_wl))

            for i in range(n_wl):
                coeffs: NDArray[np.floating]
                coeffs, _, _ = project_amplitudes(
                    traces,
                    data[:, i],
                    noise[i],
                    lam,
                )
                fit[:, i] = traces @ coeffs

            return fit.T.ravel()

        except Exception as exc:
            if debug:
                print("ODR model failure:", exc)
            return np.full_like(y_obs, 1e30)

    # Run ODR
    odr_data: odr.RealData = odr.RealData(t_flat, y_obs)
    odr_mod: odr.Model = odr.Model(odr_model)
    odr_out: odr.Output = odr.ODR(odr_data, odr_mod, beta0=beta0).run()

    # Post-fit reconstruction
    beta: NDArray[np.floating] = odr_out.beta
    traces: NDArray[np.floating] = kinetic_model.solve(times, beta)
    n_species: int = traces.shape[1]

    amplitudes: NDArray[np.floating] = np.empty((n_wl, n_species))
    amp_errors: NDArray[np.floating] = np.empty_like(amplitudes)

    for i in range(n_wl):
        coeffs: NDArray[np.floating]
        cov: NDArray[np.floating]
        coeffs, cov, _ = project_amplitudes(
            traces,
            data[:, i],
            noise[i],
            lam,
        )

        if propagate_kinetic_uncertainty:
            cov += propagate_kinetic_covariance(
                kinetic_model,
                times,
                beta,
                coeffs,
                odr_out.cov_beta,
                data[:, i],
                noise[i],
                lam,
            )

        amplitudes[i] = coeffs
        amp_errors[i] = np.sqrt(np.diag(cov))

    # Diagnostics
    # Reconstruct fitted signal for diagnostics
    fit: NDArray[np.floating] = traces @ amplitudes.T   # (n_times, n_wl)
    y_fit: NDArray[np.floating] = fit.T.ravel()
    noise_flat: NDArray[np.floating] = np.repeat(noise, n_times)

    diagnostics: dict[str, float] = compute_diagnostics(
        y_obs=y_obs,
        y_fit=y_fit,
        noise=noise_flat,
        n_params=kinetic_model.n_params(),
    )

    # Package result
    kinetics: dict[str, float] = {
        name: np.exp(val)
        for name, val in zip(
            kinetic_model.param_names(),
            beta,
        )
    }
    kinetics_err: dict[str, float] = {
        name + "_err": np.exp(val) * err
        for name, val, err in zip(
            kinetic_model.param_names(),
            beta,
            odr_out.sd_beta,
        )
    }

    result: dict = {
        "odr": odr_out,
        "kinetics": {**kinetics, **kinetics_err},
        "amplitudes": {
            "values": amplitudes,
            "errors": amp_errors,
            "species": kinetic_model.species_names(),
            "x": wl,
        },
        "diagnostics": diagnostics,
        "_cache": {
            "kinetic_model": kinetic_model,
            "beta": beta,
            "times": times,
            "traces": traces,
        },
    }

    return result

def reconstruct_fit():
    pass

def reconstruct_species():
    pass
