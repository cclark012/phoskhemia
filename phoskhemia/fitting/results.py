from __future__ import annotations

from typing import Any, Literal, TypedDict, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel

SummaryStyle = Literal["brief", "technical", "journal", "verbose"]

def cov_delta_lognormal(
        beta: NDArray[np.floating],
        cov_beta: NDArray[np.floating],
    ) -> NDArray[np.floating]:
    """
    Linear approximation for covariance in natural space for p = exp(β) with β ~ 𝒩(μ, Σ).

    Parameters
    ----------
    beta : NDArray[np.floating]
        Vector of fit parameters.
    cov_beta : NDArray[np.floating]
        Covariance of beta.

    Returns
    -------
    NDArray[np.floating]
        Covariance of p in natural space.
    """

    beta: NDArray[np.floating] = np.asarray(beta, dtype=float).reshape(-1)
    S: NDArray[np.floating] = np.asarray(cov_beta, dtype=float)
    if S.shape[0] != S.shape[1] or S.shape[0] != beta.shape[0]:
        raise ValueError("Shape mismatch: beta must be (k,), cov_beta must be (k,k).")

    J: NDArray[np.floating] = np.diag(np.exp(beta))
    return J @ S @ J

def cov_lognormal(
        beta: NDArray[np.floating],
        cov_beta: NDArray[np.floating],
    ) -> NDArray[np.floating]:
    """
    Exact covariance in natural space for p = exp(β) with β ~ 𝒩(μ, Σ).
    
    Because residuals are usually assumed to follow a normal distribution,
    parameters fit in log-space follow a log-normal distribution in natural-space.
    To obtain the covariance values in natural space the log-space covariances 
    are transformed according to: 
    Σ(pᵢ, pⱼ) = exp(μᵢ + μⱼ + ½(Σᵢᵢ + Σⱼⱼ)) ⋅ (exp(Σᵢⱼ) - 1)
    Where Σ(pᵢ, pⱼ) is the covariance of parameters pᵢ with respect to pⱼ.
    μᵢ, μⱼ are the mean values of parameters βᵢ and βⱼ (the log-space values).
    Σᵢᵢ, Σⱼⱼ are the self-correlation values (variance) of βᵢ and βⱼ.
    Σᵢⱼ is the covariance of parameters βᵢ and βⱼ.

    Parameters
    ----------
    beta : (k,) ndarray
        Vector of fit parameters.
    cov_beta : (k,k) ndarray
        Covariance of beta.

    Returns
    -------
    cov_p : (k,k) ndarray
        Exact covariance of p in natural space (multivariate log-normal).

    Raises
    ------
    ValueError
        Raised if cov_beta is not of shape (k, k).
    """

    beta: NDArray[np.floating] = np.asarray(beta, dtype=float).reshape(-1)
    S: NDArray[np.floating] = np.asarray(cov_beta, dtype=float)
    if S.shape[0] != S.shape[1] or S.shape[0] != beta.shape[0]:
        raise ValueError("Shape mismatch: beta must be (k,), cov_beta must be (k,k).")

    diag: NDArray[np.floating] = np.diag(S)
    # A_ij = exp(beta_i + beta_j + 0.5*(S_ii + S_jj))
    A: NDArray[np.floating] = (
        np.exp(beta[:, None] + beta[None, :] 
        + 0.5 * (diag[:, None] + diag[None, :]))
    )
    cov_p: NDArray[np.floating] = A * (np.exp(S) - 1.0)
    return cov_p

def _fmt_float(x: float, digits: int = 3) -> str:
    """Compact numeric formatting: fixed for moderate values, scientific otherwise."""
    VALUE_RANGE: tuple[float, float] = (1e-3, 1e4)
    abs_x: float = abs(float(x))
    if abs_x == 0.0:
        return "0"
    if (abs_x < VALUE_RANGE[0]) or (abs_x >= VALUE_RANGE[1]):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"

def _normalize_str_list(
        x: str | Sequence[str] | None, 
        n: int | None = None
    ) -> list[str | None]:

    if x is None:
        return [None] * (n or 0)

    # Convert to list of length 1 if a string
    if isinstance(x, str):
        out: list[str | None] = [x]
    else:
        out = [str(value) if value is not None else None for value in x]

    # Make sure the list is the specified length, appending or truncating.
    if n is not None:
        if len(out) < n:
            out = out + [None] * (n - len(out))
        elif len(out) > n:
            out = out[:n]
    return out

def _block_header(title: str, width: int) -> str:
    # Construct a header that is centered in a given width.
    return f"|{title:-^{width-1}}|"

def _kv_line(key: str, val: str, width: int, key_w: int = 22) -> str:
    # | key.................. = value............................... |
    rhs_w = max(0, width - (key_w + 6))
    v = val if len(val) <= rhs_w else (val[: max(0, rhs_w - 3)] + "...")
    return f"| {key:<{key_w}} = {v:<{rhs_w}} |"


def _safe_float(v) -> float | None:
    """Tries to convert to float, returning None in case of an Exception."""
    try:
        return float(v)
    except Exception:
        return None

def _top_correlations(
        cov: NDArray[np.floating],
        names: list[str],
        *,
        top_n: int = 5,
    ) -> list[tuple[str, str, float]]:
    """Return top absolute off-diagonal correlations from a covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    # Return empty list if the shape of cov is unexpected or fewer than 2 rows.
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return []
    n = cov.shape[0]
    if n < 2:
        return []

    # Try calculating Pearson's correlation coefficient for each covariance.
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (sd[:, None] * sd[None, :])

    # Get pairs in upper triangle and their correlation coefficient.
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if np.isfinite(r):
                pairs.append((names[i], names[j], float(r)))

    # Sort the pairs by the magnitude of their correlation.
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[: max(0, int(top_n))]

def sd_lognormal(
        beta: NDArray[np.floating], 
        sd_beta: NDArray[np.floating]
    ) -> NDArray[np.floating]:
    """Compute the standard deviation from log-space values and uncertainties."""

    s2: NDArray[np.floating] = np.square(sd_beta)
    mean_p: NDArray[np.floating] = np.exp(beta + 0.5*s2)
    sd_p: NDArray[np.floating] = mean_p * np.sqrt(np.exp(s2) - 1.0)
    return sd_p

def _cov_for_summary(result: GlobalFitResult) -> tuple[NDArray[np.floating] | None, str]:
    covb = result.cov_beta
    if covb is None:
        return None, "unavailable"

    if result.parameterization == "linear":
        return covb, "natural (linear)"

    # log-parameterization: choose delta (fast/readable default)
    return cov_delta_lognormal(result.beta, covb), "natural (delta, lognormal)"

def format_summary(
        result: GlobalFitResult,
        *,
        style: SummaryStyle = "brief",
        digits: int = 3,
        max_params: int | None = None,
        width: int = 72,
        include_correlations: bool | None = None,
        corr_top_n: int = 5,
        include_amplitudes: bool = False,
    ) -> str:
    """
    Human-readable summary string.

    Styles:
        - brief: minimal essentials (model + parameters + minimal diagnostics)
        - technical: detailed dev-facing diagnostics and fit config
        - journal: reviewer-facing structured report
        - verbose: near-complete dump (includes backend keys and extra blocks)
    """

    style: str = str(style).casefold().strip()
    if style not in {"brief", "technical", "journal", "verbose"}:
        raise ValueError("style must be one of: 'brief', 'technical', 'journal', 'verbose'")

    if include_correlations is None:
        include_correlations = style in {"technical", "journal", "verbose"}

    # --- Basic identity / sizes ---
    km: KineticModel = result.kinetic_model
    model_name: str = type(km).__name__ if km is not None else "<unknown model>"

    # Prefer the ordering in kinetics dict (already constructed from names)
    param_names: list[str] = list(map(str, result.kinetics.keys()))
    n_params: int = len(param_names)

    # Axes / shapes
    n_times: int | None = int(result.times.size) if getattr(result, "times", None) is not None else None
    n_wl: int | None = int(result.wavelengths.size) if getattr(result, "wavelengths", None) is not None else None
    n_points: int | None = (n_times * n_wl) if (n_times is not None and n_wl is not None) else None
    n_species: int | None = len(result.species) if getattr(result, "species", None) is not None else None

    # Optional metadata from kinetic model
    units: list[str | None] = (
        _normalize_str_list(getattr(km, "param_units", lambda: None)(), n_params) 
        if km is not None else [None] * n_params
    )
    descs: list[str | None] = (
        _normalize_str_list(getattr(km, "param_descriptions", lambda: None)(), n_params) 
        if km is not None else [None] * n_params
    )

    # CI spec
    ci_sigma: float = getattr(result, "ci_sigma", None)
    ci_level: float = getattr(result, "ci_level", None)
    ci_spec: list[str] = []
    if ci_sigma is not None:
        ci_spec.append(f"z={_fmt_float(float(ci_sigma), digits)}")
    if ci_level is not None:
        ci_spec.append(f"level={_fmt_float(100.0*float(ci_level), digits)}%")
    ci_spec_s: str = ", ".join(ci_spec) if ci_spec else "unavailable"

    # Diagnostics ordering (stable)
    preferred_diag: dict[str, str] = {
        "chi2_red": "χᵥ²", 
        "chi2": "χ²", 
        "R2": "R²", 
        "rmse": "RMSE", 
        "AIC": "AIC", 
        "AICc": "AICc", 
        "BIC": "BIC"
        }

    lines: list[str] = []

    #1). ---------------- Header ----------------
    title: str = "Global Fit Summary" if style != "journal" else "Global Kinetic Fit Report"
    lines.append(_block_header(title, width))

    #2). ---------------- Overview ----------------
    lines.append(_kv_line("Model", model_name, width))
    lines.append(_kv_line("Parameterization", str(getattr(result, "parameterization", "<unknown>")), width))
    if n_species is not None:
        lines.append(_kv_line("Species", str(n_species), width))
    if n_points is not None:
        lines.append(_kv_line("Data points", f"{n_points} (n_times={n_times}, n_wl={n_wl})", width))
    else:
        if n_times is not None:
            lines.append(_kv_line("n_times", str(n_times), width))
        if n_wl is not None:
            lines.append(_kv_line("n_wavelengths", str(n_wl), width))
    lines.append(_kv_line("CI spec", ci_spec_s, width))

    # brief stops early: only one diagnostics line (if any) + params
    # technical/journal/verbose add more blocks

    #3). ---------------- Kinetics table ----------------
    lines.append(_block_header("Kinetic parameters", width))

    # Respect max_params
    n_show: int = n_params if max_params is None else min(n_params, int(max_params))
    # values
    value: float
    unit: str | None
    dsc: str | None
    unit_s: str
    base: str
    lo: float
    hi: float
    ci_s: str
    val: str
    for i, name in enumerate(param_names[:n_show]):
        value = float(result.kinetics[name])
        unit = units[i]
        dsc = descs[i]
        unit_s = f" {unit}" if unit else ""
        base = f"{_fmt_float(value, digits)}{unit_s}"

        if result.kinetics_ci is not None and name in result.kinetics_ci:
            lo, hi = result.kinetics_ci[name]
            ci_s = f"[{_fmt_float(lo, digits)}, {_fmt_float(hi, digits)}]"
            val = f"{base}  CI {ci_s}"
        else:
            val = base

        # In journal/verbose, include descriptions if available
        if style in {"journal", "verbose"} and dsc:
            val = f"{val}  — {dsc}"

        lines.append(_kv_line(name, val, width, key_w=18))

    if n_show < n_params:
        lines.append(_kv_line("…", f"{n_params - n_show} more parameters", width, key_w=18))

    #4). ---------------- Diagnostics ----------------
    if getattr(result, "diagnostics", None):
        lines.append(_block_header("Diagnostics", width))
        diag: dict[str, float] = dict(result.diagnostics)

        # brief: single compact line
        if style == "brief":
            parts: list[str] = []
            for k in ("chi2_red", "R2", "rmse"):
                if k in diag.keys():
                    fv: float | None = _safe_float(diag.get(k))
                    if fv is not None:
                        parts.append(f"{preferred_diag[k]}={_fmt_float(fv, digits)}")
            if parts:
                lines.append(_kv_line("Key metrics", ", ".join(parts), width))
        else:
            printed: set = set()
            for k in preferred_diag.keys():
                if k in diag:
                    fv: float | None = _safe_float(diag.get(k))
                    diag_str: str = preferred_diag[k]
                    if fv is not None:
                        lines.append(_kv_line(diag_str, _fmt_float(fv, digits), width))
                        printed.add(k)
            # any remaining float-like diagnostics
            for k, v in diag.items():
                if k in printed:
                    continue
                fv: float | None = _safe_float(v)
                if fv is not None:
                    lines.append(_kv_line(str(k), _fmt_float(fv, digits), width))

    #5). ---------------- Fit config (technical/journal/verbose) ----------------
    if style in {"technical", "journal", "verbose"}:
        lines.append(_block_header("Fit configuration", width))
        lam: float = result._cache.get("lam", None)
        if lam is not None:
            lines.append(_kv_line("Tikhonov λ", _fmt_float(float(lam), digits), width))
        noise: NDArray[np.floating] = result._cache.get("noise", None)
        if noise is not None:
            nz: NDArray[np.floating] = np.asarray(noise, dtype=float)
            nz = nz[np.isfinite(nz)]
            if nz.size:
                lines.append(
                    _kv_line(
                        "Noise σ(λ)",
                        f"min={_fmt_float(float(nz.min()), digits)}, "
                        f"median={_fmt_float(float(np.median(nz)), digits)}, "
                        f"max={_fmt_float(float(nz.max()), digits)}",
                        width,
                    )
                )

    #6). ---------------- Correlations (technical/journal/verbose) ----------------
    covb, space = _cov_for_summary(result)
    if include_correlations and (covb is not None) and (n_params >= 2):
        top: list[tuple[str, str, float]] = (
            _top_correlations(covb, param_names, top_n=corr_top_n)
        )
        if top: # β-space
            lines.append(_block_header(f"Top parameter correlations ({space})", width))
            for a, b, r in top:
                lines.append(_kv_line(f"{a} ↔ {b}", _fmt_float(r, digits), width, key_w=26))

    #7). ---------------- Amplitude summary (optional / verbose) ----------------
    if include_amplitudes or style == "verbose":
        A: NDArray[np.floating] | None = getattr(result, "amplitudes", None)
        if A is not None:
            lines.append(_block_header("Amplitudes", width))
            lines.append(_kv_line("Shape", f"{tuple(A.shape)} (n_wl × n_species)", width))
            if getattr(result, "amplitude_errors", None) is not None:
                lines.append(_kv_line("Errors", "available (per wavelength/species; 1σ)", width))
            if getattr(result, "species", None):
                lines.append(_kv_line("Species names", ", ".join(map(str, result.species)), width))

    #8). ---------------- Backend (verbose only) ----------------
    if style == "verbose":
        lines.append(_block_header("Backend", width))
        try:
            bk: list[Any] = list(result.backend.keys()) if isinstance(result.backend, dict) else []
            lines.append(_kv_line("Keys", ", ".join(map(str, bk)) if bk else "<none>", width))
        except Exception:
            lines.append(_kv_line("Keys", "<unavailable>", width))

    lines.append(f"|{"_" * (width-1)}|")
    return "\n".join(lines)


class FitCache(TypedDict):
    lam: float
    noise: NDArray[np.floating]
    
    kinetic_model: KineticModel
    beta: NDArray[np.floating]
    cov_beta: NDArray[np.floating] | None
    ci_sigma: float
    ci_level: float
    parameterization: str

    traces: NDArray[np.floating]


@dataclass(frozen=True)
class GlobalFitResult:
    # Public, stable
    kinetics: dict[str, float]
    kinetics_ci: dict[str, tuple[float, float]] | None

    amplitudes: NDArray[np.floating]          # (n_wl, n_species)
    amplitude_errors: NDArray[np.floating]    # (n_wl, n_species)
    species: list[str]
    wavelengths: NDArray[np.floating]
    times: NDArray[np.floating]

    diagnostics: dict[str, float]

    # Semi-public
    backend: dict[str, Any]

    # Internal (explicitly documented)
    _cache: FitCache
    
    @property
    def traces(self) -> NDArray[np.floating]:
        return self._cache["traces"]
    
    @property
    def beta(self) -> NDArray[np.floating]:
        return self._cache["beta"]

    @property
    def cov_beta(self) -> NDArray[np.floating] | None:
        return self._cache["cov_beta"]
    
    @property
    def kinetic_model(self) -> KineticModel:
        return self._cache["kinetic_model"]
    
    @property
    def parameterization(self) -> str:
        return self._cache["parameterization"]
    
    @property
    def ci_sigma(self) -> float:
        return self._cache["ci_sigma"]
    
    @property
    def ci_level(self) -> float:
        return self._cache["ci_level"]

    def cov_kinetics(
            self,
            space: Literal["natural", "log"]="natural",
            method: Literal["exact", "delta", "linear"]="exact"
        ):
        """Placeholder for covariances of kinetic parameters."""
        raise NotImplementedError

    def sample_kinetics(self):
        """Placeholder for Monte-Carlo propagation of beta for confidence bands."""
        raise NotImplementedError
    
    def summary(self, **kwargs) -> str:
        return format_summary(self, **kwargs)
