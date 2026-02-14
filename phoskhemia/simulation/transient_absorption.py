from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray


from phoskhemia.data.spectrum_handlers import TransientAbsorption


SpectrumFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]
TracesFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]

def anchor_eps(
        wl_nm: NDArray[np.floating],
        shape: NDArray[np.floating],
        *,
        lambda_ref_nm: float,
        eps_ref: float,  # M^-1 cm^-1
    ) -> NDArray[np.floating]:
    wl = np.asarray(wl_nm, dtype=float).reshape(-1)
    s = np.asarray(shape, dtype=float).reshape(-1)
    if wl.size != s.size:
        raise ValueError("wl_nm and shape must have same length")

    s_ref = float(np.interp(lambda_ref_nm, wl, s))
    if not np.isfinite(s_ref) or s_ref == 0.0:
        raise ValueError("Reference shape value is zero/invalid; cannot anchor eps")

    return (eps_ref / s_ref) * s

def monoexp(times: NDArray[np.floating], tau: float) -> NDArray[np.floating]:
    t = np.asarray(times, dtype=float)
    return np.exp(-t / float(tau))

@dataclass(frozen=True)
class PumpSpec:
    lambda_pump_nm: float
    pathlength_cm: float = 0.1

    # choose one:
    f_exc: float | None = None                      # direct fraction excited (0..1)
    pulse_energy_J: float | None = None             # if computing from energy
    beam_area_cm2: float | None = None              # needed with pulse_energy_J
    quantum_yield: float = 1.0                      # effective scaling (0..1)


def eps_to_sigma_cm2(eps: float) -> float:
    # sigma = (1000 * ln(10) / N_A) * eps
    NA = 6.02214076e23
    return (1000.0 * np.log(10.0) / NA) * float(eps)

def excitation_fraction(
        *,
        eps_pump: float,          # M^-1 cm^-1
        conc_M: float,            # M
        pump: PumpSpec,
    ) -> float:
    if pump.f_exc is not None:
        f = float(pump.f_exc)
        if not (0.0 <= f <= 1.0):
            raise ValueError("f_exc must be in [0, 1]")
        return f

    if pump.pulse_energy_J is None or pump.beam_area_cm2 is None:
        raise ValueError("Provide either pump.f_exc or (pulse_energy_J and beam_area_cm2)")

    # Beer–Lambert absorbance at pump wavelength
    A = float(eps_pump) * float(conc_M) * float(pump.pathlength_cm)
    f_abs = 1.0 - 10.0 ** (-A)

    # photons/cm^2
    h = 6.62607015e-34
    c = 299792458.0
    lam_m = pump.lambda_pump_nm * 1e-9
    E_ph = h * c / lam_m
    F0 = float(pump.pulse_energy_J) / E_ph / float(pump.beam_area_cm2)

    # low-fluence thin-sample default (simple + stable for tests)
    sigma = eps_to_sigma_cm2(eps_pump)
    f_exc = (1.0 - np.exp(-sigma * F0)) * f_abs * float(pump.quantum_yield)

    return float(np.clip(f_exc, 0.0, 1.0))

@dataclass(frozen=True)
class SpeciesSpectra:
    # eps(λ) for GS and ES; bleach uses -GS, ESA uses +ES
    eps_gs: NDArray[np.floating]     # (n_wl,)
    eps_es: NDArray[np.floating]     # (n_wl,)


@dataclass(frozen=True)
class NoiseSpec:
    kind: Literal["constant", "sigma_lambda"] = "constant"
    sigma: float = 1e-4
    sigma_lambda: NDArray[np.floating] | None = None


def simulate_ta(
        *,
        times: NDArray[np.floating],
        wavelengths_nm: NDArray[np.floating],
        traces: NDArray[np.floating] | None = None,
        traces_fn: TracesFn | None = None,
        kinetic_model: Any | None = None,   # KineticModel
        beta: NDArray[np.floating] | None = None,
        spectra_dA: NDArray[np.floating] | None = None,        # (K, n_wl)
        delta_eps: NDArray[np.floating] | None = None,         # (K, n_wl) in M^-1 cm^-1
        amp_M: NDArray[np.floating] | None = None,             # (K,) concentration scaling
        pathlength_cm: float = 0.1,
        sigma_lambda: NDArray[np.floating] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[TransientAbsorption, dict[str, Any]]:

    t = np.asarray(times, dtype=float).reshape(-1)
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)

    # traces source
    if traces is None:
        if traces_fn is not None:
            traces = np.asarray(traces_fn(t), dtype=float)
        elif kinetic_model is not None and beta is not None:
            traces = np.asarray(kinetic_model.solve(t, beta), dtype=float)
        else:
            raise ValueError("Provide traces, traces_fn, or (kinetic_model and beta)")

    T = np.asarray(traces, dtype=float)
    if T.ndim != 2 or T.shape[0] != t.size:
        raise ValueError("traces must be (n_times, K)")

    K = T.shape[1]
    n_wl = wl.size

    # spectra source
    if spectra_dA is None:
        if delta_eps is None or amp_M is None:
            raise ValueError("Provide spectra_dA or (delta_eps and amp_M)")
        De = np.asarray(delta_eps, dtype=float)
        a = np.asarray(amp_M, dtype=float).reshape(-1)
        if De.shape != (K, n_wl) or a.size != K:
            raise ValueError("delta_eps must be (K,n_wl) and amp_M must be (K,)")
        S = float(pathlength_cm) * (a[:, None] * De)
    else:
        S = np.asarray(spectra_dA, dtype=float)
        if S.shape != (K, n_wl):
            raise ValueError("spectra_dA must be (K,n_wl)")

    surface = T @ S  # (n_times, n_wl)

    if sigma_lambda is None or rng is None:
        noisy = surface
    else:
        sig = np.asarray(sigma_lambda, dtype=float).reshape(-1)
        if sig.size == 1:
            noisy = surface + rng.normal(0.0, sig, size=surface.shape)
        elif sig.size != n_wl:
            raise ValueError("sigma_lambda must match wavelengths")
        else:
            noisy = surface + rng.normal(0.0, sig[None, :], size=surface.shape)

    truth = {"traces": T, "spectra": S, "surface_clean": surface, "sigma_lambda": sigma_lambda}
    return TransientAbsorption(noisy, x=wl, y=t, meta={"synthetic": True}), truth


