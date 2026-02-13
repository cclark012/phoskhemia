import numpy as np
from numpy.typing import NDArray
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics import KineticModel
from phoskhemia.data import TransientAbsorption


# ---------- spectral models (return shape s(lambda), not absolute eps) ----------

SpectrumFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]

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

# ---------- kinetics models (return trace T(t) with T(0)=1 convention) ----------

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
    conc_M: float,
    spectra: SpeciesSpectra,
    pump: PumpSpec,
    tau: float,
    rng: np.random.Generator,
    noise: NoiseSpec = NoiseSpec(),
) -> tuple[TransientAbsorption, dict[str, Any]]:
    """
    Minimal: one pumped species with GS bleach + ES absorption, monoexp decay.
    Produces ΔA(t, λ).
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)

    eps_gs = np.asarray(spectra.eps_gs, dtype=float).reshape(-1)
    eps_es = np.asarray(spectra.eps_es, dtype=float).reshape(-1)
    if eps_gs.size != wl.size or eps_es.size != wl.size:
        raise ValueError("eps arrays must match wavelengths")

    # pump wavelength eps for excitation fraction
    eps_pump = float(np.interp(pump.lambda_pump_nm, wl, eps_gs))
    f_exc = excitation_fraction(eps_pump=eps_pump, conc_M=conc_M, pump=pump)

    # populations (simple): excited decays monoexp back to ground
    T = monoexp(t - t.min(), tau=tau)  # define t0 at first sample for synthetic
    dC_es = (f_exc * conc_M) * T                # + excited concentration
    dC_gs = -(f_exc * conc_M) * T               # - ground state (bleach magnitude)

    # ΔA = l * (Δc_gs * eps_gs + Δc_es * eps_es)
    l = float(pump.pathlength_cm)
    surface = l * (dC_gs[:, None] * eps_gs[None, :] + dC_es[:, None] * eps_es[None, :])

    # noise
    if noise.kind == "constant":
        sigma = float(noise.sigma)
        sigma_lambda = np.full(wl.size, sigma, dtype=float)
    elif noise.kind == "sigma_lambda":
        if noise.sigma_lambda is None:
            raise ValueError("sigma_lambda required for kind='sigma_lambda'")
        sigma_lambda = np.asarray(noise.sigma_lambda, dtype=float).reshape(-1)
        if sigma_lambda.size != wl.size:
            raise ValueError("sigma_lambda must match wavelengths")
    else:
        raise ValueError("Unknown noise.kind")

    noisy = surface + rng.normal(0.0, sigma_lambda[None, :], size=surface.shape)

    meta = {
        "synthetic": True,
        "conc_M": float(conc_M),
        "pathlength_cm": float(pump.pathlength_cm),
        "lambda_pump_nm": float(pump.lambda_pump_nm),
        "f_exc": float(f_exc),
        "tau": float(tau),
        "noise_sigma_lambda": sigma_lambda,
    }

    truth = {
        "surface_clean": surface,
        "sigma_lambda": sigma_lambda,
        "dC_es": dC_es,
        "dC_gs": dC_gs,
        "eps_gs": eps_gs,
        "eps_es": eps_es,
        "f_exc": f_exc,
        "pump": pump,
    }

    return TransientAbsorption(noisy, x=wl, y=t, meta=meta), truth



