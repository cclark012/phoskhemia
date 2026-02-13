from typing import Literal
import numpy as np
from numpy.typing import NDArray


def sigma_lambda_from_probe(
        probe: NDArray[np.floating],
        *,
        model: Literal["inv_sqrt", "inv", "proportional"] = "inv_sqrt",
        floor: float = 0.0,
        ref: Literal["max", "median"] = "median",
        target_sigma: float | None = None,
        target_snr: float | None = None,
        signal_scale: float | None = None,
    ) -> NDArray[np.floating]:
    """
    Build σ(λ) from a probe intensity-like spectrum.

    If target_snr is set, you must provide signal_scale (e.g. expected max|surface|).
    """
    p: NDArray[np.floating] = np.asarray(probe, dtype=float).reshape(-1)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)

    # base shape
    if model == "inv_sqrt":
        s: NDArray[np.floating] = 1.0 / np.sqrt(p)
    elif model == "inv":
        s: NDArray[np.floating] = 1.0 / p
    elif model == "proportional":
        s: NDArray[np.floating] = p
    else:
        raise ValueError("model must be 'inv_sqrt', 'inv', or 'proportional'")

    # normalize shape to 1 at reference
    if ref == "max":
        s0 = np.nanmin(s)  # for inv-like models, min corresponds to max probe
    elif ref == "median":
        s0 = np.nanmedian(s)
    else:
        raise ValueError("ref must be 'max' or 'median'")
    if not np.isfinite(s0) or s0 == 0.0:
        raise ValueError("probe spectrum invalid for noise construction")

    s: NDArray[np.floating] = s / s0

    # scale to absolute sigma
    if target_sigma is not None and target_snr is not None:
        raise ValueError("Choose only one of target_sigma or target_snr")

    if target_sigma is not None:
        scale = float(target_sigma)
    elif target_snr is not None:
        if signal_scale is None:
            raise ValueError("signal_scale required when using target_snr")
        scale = float(signal_scale) / float(target_snr)
    else:
        scale = 1.0

    sigma = scale * s
    sigma = np.where(np.isfinite(sigma), sigma, np.nanmax(sigma[np.isfinite(sigma)]))
    sigma = sigma + float(floor)
    return sigma

def xenon_irradiance_emulated(
        wavelengths_nm: NDArray[np.floating],
        *,
        temperature_K: float = 6000.0,
        normalize: bool = True,
        window_nm: tuple[float, float] | None = None,
    ) -> NDArray[np.floating]:
    """
    Smooth xenon-like continuum surrogate (relative irradiance vs wavelength).

    Simulate an approximate xenon lamp spectrum with Planck's law for blackbody
    emitters: B(λ, T) = (2hc² / λ⁵) ⋅ [1 / (exp(hc / λkᵦT) - 1)]
    This is NOT a calibrated xenon lamp spectrum. Use measured / tabulated spectra
    when realism matters; this is a convenience fallback for synthetic noise.
    """
    wl_nm: NDArray[np.floating] = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    wl_m: NDArray[np.floating] = wl_nm * 1e-9

    # Planck spectral radiance vs wavelength (up to a constant factor)
    h: float = 6.62607015e-34
    c: float = 299792458.0
    kB: float = 1.380649e-23
    T: float = float(temperature_K)

    a: float = 2.0 * h * c**2
    b: float = h * c / (kB * T)
    x: NDArray[np.floating] = b / wl_m
    # avoid overflow in exp
    x = np.clip(x, 1e-9, 700.0)
    I: NDArray[np.floating] = a / (wl_m**5 * (np.exp(x) - 1.0))

    # Optional broad windowing to focus on a band (e.g., 320–850 nm)
    if window_nm is not None:
        lo, hi = map(float, window_nm)
        # smooth logistic edges
        s = 5.0  # nm edge softness
        w = 1.0 / (1.0 + np.exp(-(wl_nm - lo) / s)) * (1.0 / (1.0 + np.exp((wl_nm - hi) / s)))
        I = I * w

    I = np.where(np.isfinite(I) & (I > 0), I, 0.0)

    if normalize:
        m = float(np.max(I)) if np.max(I) > 0 else 1.0
        I = I / m

    return I.astype(float)

def sigma_lambda_from_transmitted(
        lamp_rel: NDArray[np.floating],
        *,
        A0: NDArray[np.floating] | float = 0.0,          # baseline absorbance vs λ
        counts_ref: float = 1e6,                          # sets absolute scale
        ref: Literal["max", "median"] = "max",            # where counts_ref applies
        include_I0: bool = True,                          # include reference-shot noise term
        read_noise_counts: float = 0.0,                   # additive read noise (counts rms)
        floor_abs: float = 0.0,                           # absorbance-noise floor
    ) -> NDArray[np.floating]:
    """
    Construct σ(λ) in absorbance units from lamp irradiance and sample transmission.

    lamp_rel: relative irradiance ~ proportional to expected I0(λ)
    A0: baseline absorbance vs wavelength (or scalar)
    counts_ref: absolute counts at reference point (max or median of lamp_rel)
    """
    L: NDArray[np.floating] = np.asarray(lamp_rel, dtype=float).reshape(-1)
    if np.any(~np.isfinite(L)) or np.all(L <= 0):
        raise ValueError("lamp_rel must be finite and contain positive values")

    # normalize lamp so reference point has value 1.0
    if ref == "max":
        L0: float = float(np.max(L))
    elif ref == "median":
        L0: float = float(np.median(L[L > 0]))
    else:
        raise ValueError("ref must be 'max' or 'median'")
    if L0 <= 0:
        raise ValueError("lamp_rel reference is non-positive")

    L = L / L0

    # absolute incident counts
    I0: NDArray[np.floating] = float(counts_ref) * L

    # baseline absorbance -> transmission
    if np.isscalar(A0):
        A0_arr: NDArray[np.floating] = np.full_like(I0, float(A0), dtype=float)
    else:
        A0_arr: NDArray[np.floating] = np.asarray(A0, dtype=float).reshape(-1)
        if A0_arr.size != I0.size:
            raise ValueError("A0 must be scalar or same length as lamp_rel")

    T: NDArray[np.floating] = 10.0 ** (-A0_arr)
    I: NDArray[np.floating] = I0 * T

    rn2: float = float(read_noise_counts) ** 2

    var_I: NDArray[np.floating] = I + rn2
    var_I0: NDArray[np.floating] = I0 + rn2

    # guard
    eps: float = 1e-30
    I_safe: NDArray[np.floating] = np.maximum(I, eps)
    I0_safe: NDArray[np.floating] = np.maximum(I0, eps)

    term: NDArray[np.floating] = var_I / (I_safe**2)
    if include_I0:
        term = term + (var_I0 / (I0_safe**2))

    sigma_A: NDArray[np.floating] = (1.0 / np.log(10.0)) * np.sqrt(term)
    sigma_A = sigma_A + float(floor_abs)

    return sigma_A.astype(float)
