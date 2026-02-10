from __future__ import annotations

from typing import Any, Literal, TypedDict, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel

SummaryStyle = Literal["brief", "technical", "journal", "verbose"]


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
    return f"{title:-^{width}}"

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
    
    def summary(
            self,
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
        km: KineticModel = self.kinetic_model
        model_name: str = type(km).__name__ if km is not None else "<unknown model>"

        # Prefer the ordering in kinetics dict (already constructed from names)
        param_names: list[str] = list(map(str, self.kinetics.keys()))
        n_params: int = len(param_names)

        # Axes / shapes
        n_times: int | None = int(self.times.size) if getattr(self, "times", None) is not None else None
        n_wl: int | None = int(self.wavelengths.size) if getattr(self, "wavelengths", None) is not None else None
        n_points: int | None = (n_times * n_wl) if (n_times is not None and n_wl is not None) else None
        n_species: int | None = len(self.species) if getattr(self, "species", None) is not None else None

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
        ci_sigma: float = getattr(self, "ci_sigma", None)
        ci_level: float = getattr(self, "ci_level", None)
        ci_spec: list[str] = []
        if ci_sigma is not None:
            ci_spec.append(f"z={_fmt_float(float(ci_sigma), digits)}")
        if ci_level is not None:
            ci_spec.append(f"level={_fmt_float(100.0*float(ci_level), digits)}%")
        ci_spec_s: str = ", ".join(ci_spec) if ci_spec else "unavailable"

        # Diagnostics ordering (stable)
        preferred_diag: list[str] = ["chi2_red", "chi2", "r2", "rmse", "aic", "aicc", "bic"]

        lines: list[str] = []

        # ---------------- Header ----------------
        title: str = "Global Fit Summary" if style != "journal" else "Global Kinetic Fit Report"
        lines.append(_block_header(title, width))

        # ---------------- Overview ----------------
        lines.append(_kv_line("Model", model_name, width))
        lines.append(_kv_line("Parameterization", str(getattr(self, "parameterization", "<unknown>")), width))
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

        # ---------------- Kinetics table ----------------
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
            value = float(self.kinetics[name])
            unit = units[i]
            dsc = descs[i]
            unit_s = f" {unit}" if unit else ""
            base = f"{_fmt_float(value, digits)}{unit_s}"

            if self.kinetics_ci is not None and name in self.kinetics_ci:
                lo, hi = self.kinetics_ci[name]
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

        # ---------------- Diagnostics ----------------
        if getattr(self, "diagnostics", None):
            lines.append(_block_header("Diagnostics", width))
            diag: dict[str, float] = dict(self.diagnostics)

            # brief: single compact line
            if style == "brief":
                parts: list[str] = []
                for k in ("chi2_red", "r2", "rmse"):
                    if k in diag:
                        fv: float | None = _safe_float(diag.get(k))
                        if fv is not None:
                            parts.append(f"{k}={_fmt_float(fv, digits)}")
                if parts:
                    lines.append(_kv_line("Key metrics", ", ".join(parts), width))
            else:
                printed: set = set()
                for k in preferred_diag:
                    if k in diag:
                        fv: float | None = _safe_float(diag.get(k))
                        if fv is not None:
                            lines.append(_kv_line(k, _fmt_float(fv, digits), width))
                            printed.add(k)
                # any remaining float-like diagnostics
                for k, v in diag.items():
                    if k in printed:
                        continue
                    fv: float | None = _safe_float(v)
                    if fv is not None:
                        lines.append(_kv_line(str(k), _fmt_float(fv, digits), width))

        # ---------------- Fit config (technical/journal/verbose) ----------------
        if style in {"technical", "journal", "verbose"}:
            lines.append(_block_header("Fit configuration", width))
            lam: float = self._cache.get("lam", None)
            if lam is not None:
                lines.append(_kv_line("Tikhonov λ", _fmt_float(float(lam), digits), width))
            noise: NDArray[np.floating] = self._cache.get("noise", None)
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

        # ---------------- Correlations (technical/journal/verbose) ----------------
        if include_correlations and (self.cov_beta is not None) and (n_params >= 2):
            top: list[tuple[str, str, float]] = (
                _top_correlations(self.cov_beta, param_names, top_n=corr_top_n)
            )
            if top:
                lines.append(_block_header("Top parameter correlations (β-space)", width))
                for a, b, r in top:
                    lines.append(_kv_line(f"{a} ↔ {b}", _fmt_float(r, digits), width, key_w=26))

        # ---------------- Amplitude summary (optional / verbose) ----------------
        if include_amplitudes or style == "verbose":
            A: NDArray[np.floating] | None = getattr(self, "amplitudes", None)
            if A is not None:
                lines.append(_block_header("Amplitudes", width))
                lines.append(_kv_line("Shape", f"{tuple(A.shape)} (n_wl × n_species)", width))
                if getattr(self, "amplitude_errors", None) is not None:
                    lines.append(_kv_line("Errors", "available (per wavelength/species; 1σ)", width))
                if getattr(self, "species", None):
                    lines.append(_kv_line("Species names", ", ".join(map(str, self.species)), width))

        # ---------------- Backend (verbose only) ----------------
        if style == "verbose":
            lines.append(_block_header("Backend", width))
            try:
                bk: list[Any] = list(self.backend.keys()) if isinstance(self.backend, dict) else []
                lines.append(_kv_line("Keys", ", ".join(map(str, bk)) if bk else "<none>", width))
            except Exception:
                lines.append(_kv_line("Keys", "<unavailable>", width))

        lines.append("-" * width)
        return "\n".join(lines)
