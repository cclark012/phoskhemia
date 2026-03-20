from __future__ import annotations

from typing import Any, Literal, TypedDict, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel

SummaryStyle = Literal["brief", "technical", "journal", "verbose"]
RenderFormat = Literal["terminal", "plain", "latex"]

@dataclass(frozen=True)
class ReportRow:
    key: str
    value: str

@dataclass(frozen=True)
class ReportBlock:
    title: str
    rows: list[ReportRow]

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


def sd_lognormal(
        beta: NDArray[np.floating], 
        sd_beta: NDArray[np.floating]
    ) -> NDArray[np.floating]:
    """Compute the standard deviation from log-space values and uncertainties."""

    s2: NDArray[np.floating] = np.square(sd_beta)
    mean_p: NDArray[np.floating] = np.exp(beta + 0.5*s2)
    sd_p: NDArray[np.floating] = mean_p * np.sqrt(np.exp(s2) - 1.0)
    return sd_p

def _fmt_float(x: float, digits: int = 3) -> str:
    """Compact numeric formatting: fixed for moderate values, scientific otherwise."""
    fx = float(x)
    if not np.isfinite(fx):
        return str(fx)

    VALUE_RANGE: tuple[float, float] = (1e-3, 1e4)
    abs_x: float = abs(fx)
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

def _safe_float(v) -> float | None:
    """Tries to convert to float, returning None in case of an Exception."""
    try:
        if np.isfinite(v):
            return float(v)
        else:
            return None
    except Exception:
        return None

def _block_header(title: str, width: int) -> str:
    # Construct a header that is centered in a given width.
    return f"|{title:-^{width-1}}|"

def _kv_line(key: str, val: str, width: int, key_w: int = 22) -> str:
    # | key.................. = value............................... |
    rhs_w = max(0, width - (key_w + 6))
    v = val if len(val) <= rhs_w else (val[: max(0, rhs_w - 3)] + "...")
    return f"| {key:<{key_w}} = {v:<{rhs_w}} |"

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

def _full_correlation_rows(
        cov: NDArray[np.floating],
        names: list[str],
        *,
        digits: int = 3,
        max_dim: int = 10,
    ) -> list[ReportRow]:
    """Render a full correlation matrix. Falls back to shape hint for large matrices."""

    n = cov.shape[0]
    if n > max_dim:
        return [ReportRow("Shape", f"{n}×{n} (use result.cov_beta directly)")]
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (sd[:, None] * sd[None, :])
    rows: list[ReportRow] = []
    for i, name in enumerate(names):
        vals = "  ".join(_fmt_float(corr[i, j], digits) for j in range(n))
        rows.append(ReportRow(name, vals))
    return rows

def _cov_for_summary(result: GlobalFitResult) -> tuple[NDArray[np.floating] | None, str]:
    covb = result.cov_beta
    if covb is None:
        return None, "unavailable"

    if result.parameterization == "linear":
        return covb, "natural (linear)"

    # log-parameterization: choose delta (fast/readable default)
    return cov_delta_lognormal(result.beta, covb), "natural (delta, lognormal)"

_PREFERRED_DIAG: dict[str, str] = {
    "chi2_red": "χᵥ²", 
    "chi2": "χ²", 
    "R2": "R²", 
    "rmse": "RMSE", 
    "AIC": "AIC", 
    "AICc": "AICc", 
    "BIC": "BIC",
    "DoF": "DoF",
}

def _build_param_rows(
        result: GlobalFitResult,
        *,
        km: KineticModel | None,
        param_names: list[str],
        n_show: int,
        digits: int,
        include_descriptions: bool,
    ) -> list[ReportRow]:

    n_params: int = len(param_names)
    units: list[str | None] = (
        _normalize_str_list(km.param_units(), n_params)  # type: ignore[union-attr]
        if km is not None else [None] * n_params
    )
    descs: list[str | None] = (
        _normalize_str_list(km.param_descriptions(), n_params)  # type: ignore[union-attr]
        if km is not None else [None] * n_params
    )
    rows: list[ReportRow] = []
    lo: float | None
    hi: float | None
    for i, name in enumerate(param_names[:n_show]):
        value: float = float(result.kinetics[name])
        unit_s: str = f" {units[i]}" if units[i] else ""
        base: str = f"{_fmt_float(value, digits)}{unit_s}"
        if result.kinetics_ci is not None and name in result.kinetics_ci:
            lo, hi = result.kinetics_ci[name]
            base: str = f"{base}  CI [{_fmt_float(lo, digits)}, {_fmt_float(hi, digits)}]"
        if include_descriptions and descs[i]:
            base: str = f"{base}  — {descs[i]}"
        rows.append(ReportRow(name, base))
    if n_show < n_params:
        rows.append(ReportRow("…", f"{n_params - n_show} more parameter(s)"))
    return rows

def _build_overview_block(
        result: GlobalFitResult,
        *,
        km: KineticModel | None,
        model_name: str,
        digits: int,
    ) -> ReportBlock:

    n_times: int = int(result.times.size) if result.times is not None else None
    n_wl: int = int(result.wavelengths.size) if result.wavelengths is not None else None
    n_points: int = (n_times * n_wl) if (n_times is not None and n_wl is not None) else None
    n_species: int = len(result.species) if result.species else None

    ci_sigma: float | None = result.ci_sigma
    ci_level: float | None = result.ci_level
    ci_parts: list[str] = []
    if ci_sigma is not None:
        ci_parts.append(f"z={_fmt_float(float(ci_sigma), digits)}")
    if ci_level is not None:
        ci_parts.append(f"level={_fmt_float(100.0 * float(ci_level), digits)}%")
    ci_spec: str = ", ".join(ci_parts) if ci_parts else "unavailable"

    rows: list[ReportRow] = [
        ReportRow("Model", model_name),
        ReportRow("Parameterization", str(result.parameterization)),
        ReportRow("CI spec", ci_spec),
    ]
    if n_species is not None:
        rows.append(ReportRow("Species", str(n_species)))
    if n_points is not None:
        rows.append(ReportRow("Data points", f"{n_points}  (n_times={n_times}, n_wl={n_wl})"))
    return ReportBlock("Overview", rows)

def _build_data_range_block(result: GlobalFitResult, *, digits: int) -> ReportBlock | None:
    """Build a data-range block (wavelength and time extents). Returns None if axes unavailable."""

    rows: list[ReportRow] = []
    if result.wavelengths is not None and result.wavelengths.size >= 2:
        wl: NDArray[np.floating] = np.asarray(result.wavelengths, dtype=float)
        rows.append(ReportRow(
            "λ range",
            f"{_fmt_float(float(wl.min()), digits)} – {_fmt_float(float(wl.max()), digits)} nm"
            f"  (n={wl.size})",
        ))
    if result.times is not None and result.times.size >= 2:
        t: NDArray[np.floating] = np.asarray(result.times, dtype=float)
        rows.append(ReportRow(
            "t range",
            f"{_fmt_float(float(t.min()), digits)} – {_fmt_float(float(t.max()), digits)}"
            f"  (n={t.size})",
        ))
    n_times: int = int(result.times.size) if result.times is not None else None
    n_wl: int = int(result.wavelengths.size) if result.wavelengths is not None else None
    if n_times is not None and n_wl is not None:
        rows.append(ReportRow("Total points", str(n_times * n_wl)))
    if not rows:
        return None
    return ReportBlock("Data", rows)

def _build_diagnostics_block(
        result: GlobalFitResult,
        *,
        keys: set[str] | None,
        digits: int,
    ) -> ReportBlock | None:
    """
    Build a diagnostics block.

    Parameters
    ----------
    keys
        If provided, only emit diagnostics whose keys are in this set.
        If None, emit all diagnostics in preferred order then remainder.
    """
    if not result.diagnostics:
        return None
    diag: dict[str, float] = dict(result.diagnostics)
    rows: list[ReportRow] = []
    printed: set[str] = set()
    for k, lab in _PREFERRED_DIAG.items():
        if keys is not None and k not in keys:
            continue
        if k not in diag:
            continue
        fv: float | None = _safe_float(diag[k])
        if fv is not None:
            rows.append(ReportRow(lab, _fmt_float(fv, digits)))
            printed.add(k)
    if keys is None:
        # Emit any remaining keys not in the preferred list
        for k, v in diag.items():
            if k in printed:
                continue
            fv = _safe_float(v)
            if fv is not None:
                rows.append(ReportRow(str(k), _fmt_float(fv, digits)))
    return ReportBlock("Diagnostics", rows) if rows else None

def _build_fit_config_block(result: GlobalFitResult, *, digits: int) -> ReportBlock | None:
    rows: list[ReportRow] = []
    lam: float | None = result._cache.get("lam")
    if lam is not None:
        rows.append(ReportRow("Tikhonov λ", _fmt_float(float(lam), digits)))
    noise: float | NDArray[np.floating] | None = result._cache.get("noise")
    if noise is not None:
        nz = np.asarray(noise, dtype=float)
        nz = nz[np.isfinite(nz)]
        if nz.size:
            rows.append(ReportRow(
                "Noise σ(λ)",
                f"min={_fmt_float(float(nz.min()), digits)}, "
                f"median={_fmt_float(float(np.median(nz)), digits)}, "
                f"max={_fmt_float(float(nz.max()), digits)}",
            ))
    return ReportBlock("Fit configuration", rows) if rows else None

def _build_solver_block(result: GlobalFitResult, *, digits: int) -> ReportBlock | None:
    """Extract solver diagnostics from _cache['solver_info']."""

    info = result._cache.get("solver_info")
    if not isinstance(info, dict):
        return None
    rows: list[ReportRow] = []
    backend = info.get("backend")
    if backend is not None:
        rows.append(ReportRow("Backend", str(backend)))
    stopreason = info.get("stopreason")
    if stopreason:
        rows.append(ReportRow("Stop reason", str(stopreason)))
    for key, label in [("niter", "Iterations"), ("nfev", "Fn evaluations"), ("njev", "Jac evaluations")]:
        v = info.get(key)
        if v is not None and int(v) >= 0:
            rows.append(ReportRow(label, str(int(v))))
    # res_var = result._cache.get("cov_beta")  # res_var lives on SolverResult, not propagated — use cache
    # res_var is not stored in _cache currently; emit if available via backend raw
    raw = info.get("raw")
    if raw is not None:
        rv = getattr(raw, "res_var", None)
        if rv is not None:
            try:
                rows.append(ReportRow("Residual variance", _fmt_float(float(rv), digits)))
            except Exception:
                pass
        info_code = getattr(raw, "info", None)
        if info_code is not None:
            rows.append(ReportRow("Info code", str(info_code)))
    return ReportBlock("Solver", rows) if rows else None

def _build_amplitude_block(result: GlobalFitResult, *, digits: int) -> ReportBlock:
    A: NDArray[np.floating] = result.amplitudes
    Ae: NDArray[np.floating] = result.amplitude_errors
    rows: list[ReportRow] = [
        ReportRow("Shape", f"{tuple(A.shape)}  (n_wl × n_species)"),
    ]
    for s_idx, sp in enumerate(result.species):
        col: NDArray[np.floating] = A[:, s_idx]
        col_e: NDArray[np.floating] = Ae[:, s_idx]
        rows.append(ReportRow(
            sp,
            f"mean={_fmt_float(float(col.mean()), digits)}, "
            f"std={_fmt_float(float(col.std()), digits)}, "
            f"range=[{_fmt_float(float(col.min()), digits)}, {_fmt_float(float(col.max()), digits)}]  "
            f"err_mean={_fmt_float(float(col_e.mean()), digits)}",
        ))
    return ReportBlock("Amplitudes", rows)

def _build_covariance_block(
        result: GlobalFitResult,
        param_names: list[str],
        *,
        digits: int,
    ) -> ReportBlock | None:
    if result.cov_beta is None:
        return None
    cov: NDArray[np.floating] | None
    cov_label: str
    cov, cov_label = _cov_for_summary(result)
    if cov is None:
        return None
    rows: list[ReportRow] = []
    # Variances (diagonal in natural space)
    sd_nat: NDArray[np.floating] = np.sqrt(np.diag(cov))
    for name, s in zip(param_names, sd_nat):
        rows.append(ReportRow(f"σ({name})", _fmt_float(float(s), digits)))
    # Full correlation matrix
    rows.append(ReportRow("", ""))  # blank separator
    rows.append(ReportRow("Correlation matrix", f"({cov_label})"))
    rows.extend(_full_correlation_rows(cov, param_names, digits=digits))
    return ReportBlock("Covariance", rows)

'''
def build_fit_report(
        result: "GlobalFitResult",
        *,
        style: SummaryStyle = "brief",
        digits: int = 3,
        max_params: int | None = None,
        include_correlations: bool | None = None,
        corr_top_n: int = 5,
        include_amplitudes: bool = False,
    ) -> list[ReportBlock]:
    """
    Creates a list of summary blocks for reporting/saving.

    Styles:
        - brief: minimal essentials (model + parameters + minimal diagnostics)
        - technical: detailed dev-facing diagnostics and fit config
        - journal: reviewer-facing structured report
        - verbose: near-complete dump (includes backend keys and extra blocks)
    """

    style = str(style).casefold().strip()
    if style not in {"brief", "technical", "journal", "verbose"}:
        raise ValueError("style must be one of: 'brief', 'technical', 'journal', 'verbose'")

    if include_correlations is None:
        include_correlations = style in {"technical", "journal", "verbose"}

    blocks: list[ReportBlock] = []

    #1). ---- Overview ----
    km = result.kinetic_model
    model_name = type(km).__name__ if km is not None else "<unknown model>"

    # Axes and their shapes
    n_times = int(result.times.size) if result.times is not None else None
    n_wl = int(result.wavelengths.size) if result.wavelengths is not None else None
    n_points = (n_times * n_wl) if (n_times is not None and n_wl is not None) else None
    n_species = len(result.species) if result.species is not None else None

    # CI information
    ci_sigma = getattr(result, "ci_sigma", None)
    ci_level = getattr(result, "ci_level", None)
    ci_parts: list[str] = []
    if ci_sigma is not None:
        ci_parts.append(f"z={_fmt_float(float(ci_sigma), digits)}")
    if ci_level is not None:
        ci_parts.append(f"level={_fmt_float(100.0*float(ci_level), digits)}%")
    ci_spec = ", ".join(ci_parts) if ci_parts else "unavailable"

    overview_rows = [
        ReportRow("Model", model_name),
        ReportRow("Parameterization", str(getattr(result, "parameterization", "<unknown>"))),
        ReportRow("CI spec", ci_spec),
    ]
    if n_species is not None:
        overview_rows.append(ReportRow("Species", str(n_species)))
    if n_points is not None:
        overview_rows.append(ReportRow("Data points", f"{n_points} (n_times={n_times}, n_wl={n_wl})"))
    else:
        if n_times is not None:
            overview_rows.append(ReportRow("n_times", str(n_times)))
        if n_wl is not None:
            overview_rows.append(ReportRow("n_wavelengths", str(n_wl)))

    blocks.append(ReportBlock("Overview", overview_rows))

    #2). ---- Kinetic parameters ----
    param_names = list(map(str, result.kinetics.keys()))
    n_params = len(param_names)
    n_show = n_params if max_params is None else min(n_params, int(max_params))

    # Optional metadata from kinetic model
    units: list[str | None] = (
        _normalize_str_list(getattr(km, "param_units", lambda: None)(), n_params) 
        if km is not None else [None] * n_params
    )
    descs: list[str | None] = (
        _normalize_str_list(getattr(km, "param_descriptions", lambda: None)(), n_params) 
        if km is not None else [None] * n_params
    )

    value: float
    unit: str | None
    dsc: str | None
    unit_s: str
    base: str
    lo: float
    hi: float
    fv: float | None
    rows: list[ReportRow] = []
    for i, name in enumerate(param_names[:n_show]):
        value = float(result.kinetics[name])
        unit = units[i]
        dsc = descs[i]
        unit_s = f" {unit}" if unit else ""
        base = f"{_fmt_float(value, digits)}{unit_s}"

        if result.kinetics_ci is not None and name in result.kinetics_ci:
            lo, hi = result.kinetics_ci[name]
            base = f"{base}  CI [{_fmt_float(lo, digits)}, {_fmt_float(hi, digits)}]"

        if style in {"journal", "verbose"} and dsc:
            base = f"{base}  — {dsc}"

        rows.append(ReportRow(name, base))

    if n_show < n_params:
        rows.append(ReportRow("…", f"{n_params - n_show} more parameters"))

    blocks.append(ReportBlock("Kinetic parameters", rows))

    #3). ---- Diagnostics ----
    if getattr(result, "diagnostics", None):
        diag: dict[str, float] = dict(result.diagnostics)

        diag_rows: list[ReportRow] = []
        # Keep output short for brief
        if style == "brief":
            parts: list[str] = []
            for k, lab in _PREFERRED_DIAG.items():
                if k in diag:
                    fv = _safe_float(diag.get(k))
                    if fv is not None and k in {"chi2_red", "R2", "rmse"}:
                        parts.append(f"{lab}={_fmt_float(fv, digits)}")
            if parts:
                diag_rows.append(ReportRow("Key metrics", ", ".join(parts)))
        else:
            printed = set()
            for k, lab in _PREFERRED_DIAG.items():
                if k in diag:
                    fv = _safe_float(diag.get(k))
                    if fv is not None:
                        diag_rows.append(ReportRow(lab, _fmt_float(fv, digits)))
                        printed.add(k)
            for k, v in diag.items():
                if k in printed:
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    diag_rows.append(ReportRow(str(k), _fmt_float(fv, digits)))

        blocks.append(ReportBlock("Diagnostics", diag_rows))

    #4). ---- Fit configuration ----
    if style in {"technical", "journal", "verbose"}:
        rows: list[ReportRow] = []
        lam: float | None = result._cache.get("lam", None)
        if lam is not None:
            rows.append(ReportRow("Tikhonov λ", _fmt_float(float(lam), digits)))
        noise: NDArray[np.floating] = result._cache.get("noise", None)
        if noise is not None:
            nz: NDArray[np.floating] = np.asarray(noise, dtype=float)
            nz = nz[np.isfinite(nz)]
            if nz.size:
                rows.append(
                    ReportRow(
                        "Noise σ(λ)",
                        f"min={_fmt_float(float(nz.min()), digits)}, "
                        f"median={_fmt_float(float(np.median(nz)), digits)}, "
                        f"max={_fmt_float(float(nz.max()), digits)}",
                    )
                )
        blocks.append(ReportBlock("Fit configuration", rows))

    #5). ---- Correlations ----
    if include_correlations and (result.cov_beta is not None) and (n_params >= 2) and style != "brief":
        cov, cov_label = _cov_for_summary(result)
        if cov is not None:
            top = _top_correlations(cov, param_names, top_n=corr_top_n)
            if top:
                rows = [ReportRow(f"{a} ↔ {b}", _fmt_float(r, digits)) for a, b, r in top]
                blocks.append(ReportBlock(f"Top parameter correlations ({cov_label})", rows))

    #6). ---- Amplitudes ----
    if include_amplitudes or style == "verbose":
        A: NDArray[np.floating] = result.amplitudes
        rows = [
            ReportRow("Shape", f"{tuple(A.shape)} (n_wl × n_species)"),
            ReportRow("Errors", "available (per wavelength/species; 1σ)"),
            ReportRow("Species names", ", ".join(map(str, result.species))),
        ]
        blocks.append(ReportBlock("Amplitudes", rows))

    #7). ---- Backend ----
    if style == "verbose":
        try:
            bk: list[Any] = list(result.backend.keys()) if isinstance(result.backend, dict) else []
            blocks.append(ReportBlock("Backend", [ReportRow("Keys", ", ".join(map(str, bk)) if bk else "<none>")]))
        except Exception:
            blocks.append(ReportBlock("Backend", [ReportRow("Keys", "<unavailable>")]))

    return blocks
'''

def build_fit_report(
        result: GlobalFitResult,
        *,
        style: SummaryStyle = "brief",
        digits: int = 3,
        max_params: int | None = None,
        include_correlations: bool | None = None,
        corr_top_n: int = 5,
        include_amplitudes: bool = False,
    ) -> list[ReportBlock]:
    """
    Build a list of ReportBlocks for a GlobalFitResult.

    Styles
    ------
    brief
        Single compact block: model name, kinetic parameters with CIs,
        and a one-line diagnostic summary (χᵥ², R²).
    technical
        Developer-facing: overview, all parameters, all diagnostics,
        fit configuration (λ, noise), top parameter correlations, solver info.
    journal
        Publication-facing (JACS / Nature Chem / Angew. Chem. style):
        data range, model + species, parameters with units and descriptions,
        key fit-quality metrics (χᵥ², R², AIC, AICc, BIC).
    verbose
        Near-complete dump: all technical blocks plus full covariance matrix,
        per-species amplitude statistics, and backend details.

    Parameters
    ----------
    result : GlobalFitResult
    style : {'brief', 'technical', 'journal', 'verbose'}
    digits : int
        Significant digits for numeric formatting.
    max_params : int or None
        Truncate parameter table to this many rows (None = show all).
    include_correlations : bool or None
        Override correlation block inclusion. Default: True for
        technical/journal/verbose, False for brief.
    corr_top_n : int
        Number of top correlations to display (technical style).
    include_amplitudes : bool
        Force-include amplitude block for non-verbose styles.

    Returns
    -------
    list[ReportBlock]
    """
    style = str(style).casefold().strip()
    if style not in {"brief", "technical", "journal", "verbose"}:
        raise ValueError("style must be one of: 'brief', 'technical', 'journal', 'verbose'")

    if include_correlations is None:
        include_correlations = style in {"technical", "journal", "verbose"}

    km = result.kinetic_model
    model_name = type(km).__name__ if km is not None else "<unknown model>"
    param_names = list(result.kinetics.keys())
    n_params = len(param_names)
    n_show = n_params if max_params is None else min(n_params, int(max_params))
    blocks: list[ReportBlock] = []

    # ------------------------------------------------------------------
    if style == "brief":
        rows: list[ReportRow] = [ReportRow("Model", model_name)]
        rows.extend(_build_param_rows(
            result, km=km, param_names=param_names, n_show=n_show,
            digits=digits, include_descriptions=False,
        ))
        # One-line diagnostics
        diag = result.diagnostics or {}
        parts: list[str] = []
        for k, lab in [("chi2_red", "χᵥ²"), ("R2", "R²")]:
            fv = _safe_float(diag.get(k))
            if fv is not None:
                parts.append(f"{lab}={_fmt_float(fv, digits)}")
        if parts:
            rows.append(ReportRow("Diagnostics", ", ".join(parts)))
        blocks.append(ReportBlock("Fit Summary", rows))

    # ------------------------------------------------------------------
    elif style == "technical":
        blocks.append(_build_overview_block(
            result, km=km, model_name=model_name, digits=digits,
        ))
        blocks.append(ReportBlock(
            "Kinetic parameters",
            _build_param_rows(
                result, km=km, param_names=param_names, n_show=n_show,
                digits=digits, include_descriptions=False,
            ),
        ))
        diag_block = _build_diagnostics_block(result, keys=None, digits=digits)
        if diag_block:
            blocks.append(diag_block)
        config_block = _build_fit_config_block(result, digits=digits)
        if config_block:
            blocks.append(config_block)
        # Top correlations
        if include_correlations and result.cov_beta is not None and n_params >= 2:
            cov, cov_label = _cov_for_summary(result)
            if cov is not None:
                top = _top_correlations(cov, param_names, top_n=corr_top_n)
                if top:
                    corr_rows = [
                        ReportRow(f"{a} ↔ {b}", _fmt_float(r, digits))
                        for a, b, r in top
                    ]
                    blocks.append(ReportBlock(
                        f"Top parameter correlations ({cov_label})", corr_rows,
                    ))
        solver_block = _build_solver_block(result, digits=digits)
        if solver_block:
            blocks.append(solver_block)

    # ------------------------------------------------------------------
    elif style == "journal":
        data_block = _build_data_range_block(result, digits=digits)
        if data_block:
            blocks.append(data_block)
        # Model + species
        species_str = ", ".join(str(s) for s in result.species) if result.species else "—"
        ci_sigma = result.ci_sigma
        ci_level = result.ci_level
        ci_parts = []
        if ci_sigma is not None:
            ci_parts.append(f"z={_fmt_float(float(ci_sigma), digits)}")
        if ci_level is not None:
            ci_parts.append(f"{_fmt_float(100.0 * float(ci_level), digits)}%")
        ci_spec = ", ".join(ci_parts) if ci_parts else "unavailable"
        blocks.append(ReportBlock("Model", [
            ReportRow("Model", model_name),
            ReportRow("Species", species_str),
            ReportRow("Parameterization", str(result.parameterization)),
            ReportRow("CI", ci_spec),
        ]))
        # Parameters with units and descriptions
        blocks.append(ReportBlock(
            "Kinetic parameters",
            _build_param_rows(
                result, km=km, param_names=param_names, n_show=n_show,
                digits=digits, include_descriptions=True,
            ),
        ))
        # Fit quality: key metrics only
        diag_block = _build_diagnostics_block(
            result,
            keys={"chi2_red", "R2", "AIC", "AICc", "BIC"},
            digits=digits,
        )
        if diag_block:
            blocks.append(ReportBlock("Fit quality", diag_block.rows))

    # ------------------------------------------------------------------
    elif style == "verbose":
        blocks.append(_build_overview_block(
            result, km=km, model_name=model_name, digits=digits,
        ))
        blocks.append(ReportBlock(
            "Kinetic parameters",
            _build_param_rows(
                result, km=km, param_names=param_names, n_show=n_show,
                digits=digits, include_descriptions=True,
            ),
        ))
        diag_block = _build_diagnostics_block(result, keys=None, digits=digits)
        if diag_block:
            blocks.append(diag_block)
        config_block = _build_fit_config_block(result, digits=digits)
        if config_block:
            blocks.append(config_block)
        solver_block = _build_solver_block(result, digits=digits)
        if solver_block:
            blocks.append(solver_block)
        cov_block = _build_covariance_block(result, param_names, digits=digits)
        if cov_block:
            blocks.append(cov_block)
        blocks.append(_build_amplitude_block(result, digits=digits))
        # Backend
        try:
            bk = list(result.backend.keys()) if isinstance(result.backend, dict) else []
            blocks.append(ReportBlock(
                "Backend",
                [ReportRow("Keys", ", ".join(map(str, bk)) if bk else "<none>")],
            ))
        except Exception:
            blocks.append(ReportBlock("Backend", [ReportRow("Keys", "<unavailable>")]))

    # Amplitude block for non-verbose styles when explicitly requested
    if include_amplitudes and style != "verbose":
        blocks.append(_build_amplitude_block(result, digits=digits))

    return blocks


def render_terminal_report(blocks: list[ReportBlock], *, width: int = 72) -> str:
    lines: list[str] = [_block_header("Global Fit Summary", width)]
    for block in blocks:
        lines.append(_block_header(block.title, width))
        for row in block.rows:
            lines.append(_kv_line(row.key, row.value, width))
    # bottom border: keep your preferred width
    lines.append("|" + ("_" * (width - 1)) + "|")
    return "\n".join(lines)

def render_plain_report(blocks: list[ReportBlock]) -> str:
    lines: list[str] = ["Global Fit Summary"]
    for block in blocks:
        lines.append("")
        lines.append(block.title)
        for row in block.rows:
            lines.append(f"{row.key}: {row.value}")
    return "\n".join(lines)

def render_latex_report(
        blocks: list[ReportBlock],
        *,
        caption: str = "Global kinetic fit summary.",
        label: str = "tab:globalfit",
    ) -> str:
    """
    Render blocks as a LaTeX longtable suitable for journal SI sections.

    The output is a ``longtable`` fragment intended to be embedded in an
    existing ``.tex`` document.  It is **not** a standalone compilable file.
    The host document must load the following packages in its preamble::

        \\usepackage{longtable}
        \\usepackage{booktabs}   % \\toprule, \\midrule, \\bottomrule

    Blank-key rows (used as separators in the verbose covariance block)
    are silently skipped.

    .. warning::
        Built-in kinetic models (e.g. ``ExponentialModel``, ``TTAKineticModel``)
        use Unicode characters in their parameter names (τ, β, χ, etc.) for
        terminal display.  Plain LaTeX (without ``\\usepackage{inputenc}`` /
        ``\\usepackage{fontenc}`` or a Unicode-aware engine such as XeLaTeX or
        LuaLaTeX) cannot compile these characters and will raise an error.
        A ``UserWarning`` is issued at runtime when Unicode is detected in the
        block content.  Future versions will expose ``param_names_latex()`` on
        ``KineticModel`` to provide ASCII/math-mode alternatives.
    """
    import warnings

    def _tex_escape(s: str) -> str:
        """Minimal escaping of characters that break LaTeX in tabular content."""
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("^", r"\^{}")
            .replace("~", r"\textasciitilde{}")
            .replace("<", r"\textless{}")
            .replace(">", r"\textgreater{}")
        )
    # Detect non-ASCII content across all block rows and warn once.
    unicode_found: list[str] = []
    for block in blocks:
        for row in block.rows:
            for text in (row.key, row.value):
                non_ascii = [c for c in text if ord(c) > 127]
                if non_ascii:
                    chars = "".join(dict.fromkeys(non_ascii))  # deduplicated, order-preserved
                    unicode_found.append(f"{chars!r} in {text!r}")
    if unicode_found:
        sample = "; ".join(unicode_found[:3])
        suffix = f" (and {len(unicode_found) - 3} more)" if len(unicode_found) > 3 else ""
        warnings.warn(
            f"render_latex_report: non-ASCII characters detected in report content "
            f"({sample}{suffix}). "
            "Standard pdflatex will fail to compile these. Sources include built-in "
            "diagnostic labels (χᵥ², R², λ) and any Unicode parameter names from the "
            "kinetic model. Use XeLaTeX/LuaLaTeX with \\usepackage{fontspec}, or "
            "override param_names() on your KineticModel to return ASCII/math-mode "
            "strings for LaTeX output.",
            UserWarning,
            stacklevel=2,
        )

    lines: list[str] = [
        r"\begin{longtable}{@{}ll@{}}",
        r"\caption{" + _tex_escape(caption) + r"} \label{" + label + r"} \\",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\midrule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{2}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for block in blocks:
        lines.append(
            r"\multicolumn{2}{@{}l}{\textbf{"
            + _tex_escape(block.title)
            + r"}} \\"
        )
        lines.append(r"\midrule")
        for row in block.rows:
            if not row.key.strip():
                continue  # skip blank separator rows
            lines.append(
                _tex_escape(row.key) + " & " + _tex_escape(row.value) + r" \\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\end{longtable}")
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
    solver_info: dict[str, Any]


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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the result to a dict of JSON-compatible Python types.

        Arrays are converted to nested lists. ``cov_beta`` and ``beta``
        are included only when available.  The kinetic model is represented
        by its class name; full model serialization is handled separately
        via ``KineticModel.to_config`` (T0013).

        Returns
        -------
        dict
        """

        d: dict[str, Any] = {
            "model": type(self.kinetic_model).__name__ if self.kinetic_model is not None else None,
            "parameterization": self.parameterization,
            "species": list(self.species),
            "kinetics": dict(self.kinetics),
            "kinetics_ci": (
                {k: list(v) for k, v in self.kinetics_ci.items()}
                if self.kinetics_ci is not None else None
            ),
            "ci_sigma": float(self.ci_sigma) if self.ci_sigma is not None else None,
            "ci_level": float(self.ci_level) if self.ci_level is not None else None,
            "diagnostics": {k: float(v) for k, v in self.diagnostics.items()},
            "wavelengths": self.wavelengths.tolist(),
            "times": self.times.tolist(),
            "n_wavelengths": int(self.wavelengths.size),
            "n_times": int(self.times.size),
            "n_species": len(self.species),
            "amplitudes": self.amplitudes.tolist(),          # (n_wl, n_species)
            "amplitude_errors": self.amplitude_errors.tolist(),
        }
        if self.cov_beta is not None:
            d["beta"] = self.beta.tolist()
            d["cov_beta"] = self.cov_beta.tolist()
        return d

    def summary(
            self,
            *,
            style: SummaryStyle = "brief",
            width: int = 72,
            **kwargs: Any,
        ) -> str:
        """
        Return a fixed-width terminal string summary.

        Parameters
        ----------
        style : {'brief', 'technical', 'journal', 'verbose'}
        width : int
            Column width of the terminal box.
        **kwargs
            Forwarded to ``build_fit_report``.
        """
        blocks = build_fit_report(self, style=style, **kwargs)
        return render_terminal_report(blocks, width=width)

    def to_text(
            self,
            *,
            style: SummaryStyle = "brief",
            fmt: RenderFormat = "plain",
            **kwargs: Any,
        ) -> str:
        """
        Return the report as a plain-text or LaTeX string.

        Parameters
        ----------
        style : {'brief', 'technical', 'journal', 'verbose'}
        fmt : {'plain', 'latex'}
            Output format.  Use ``'latex'`` to get a ``longtable``
            environment suitable for journal SI sections.
        **kwargs
            Forwarded to ``build_fit_report``.
        """
        blocks = build_fit_report(self, style=style, **kwargs)
        if fmt == "latex":
            return render_latex_report(blocks, **{
                k: kwargs[k] for k in ("caption", "label") if k in kwargs
            })
        return render_plain_report(blocks)
