
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def _fmt(x: float, digits: int = 3) -> str:
    """Compact numeric formatting: fixed for moderate values, scientific otherwise."""
    abs_x = abs(x)
    if abs_x == 0:
        return "0"
    if (abs_x >= 1e4) or (abs_x < 1e-3):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


@dataclass(slots=True)
class GlobalFitResult:
    """
    Minimal result container for a global kinetic fit.

    Parameters are assumed to be stored in log-space (beta = log(params)),
    but summary() reports values in linear space (exp(beta)).
    """
    kinetic_model: Any  # prefer Protocol[KineticModel] if you have one
    beta: NDArray[np.floating]  # log-parameters, shape (n_params,)

    # Optional ODR outputs
    sd_beta: NDArray[np.floating] | None = None
    cov_beta: NDArray[np.floating] | None = None
    odr_output: Any | None = None

    # Data axes
    times: NDArray[np.floating] | None = None          # shape (n_times,)
    wavelengths: NDArray[np.floating] | None = None    # shape (n_wl,)

    # Fit components
    traces: NDArray[np.floating] | None = None         # shape (n_times, n_species)
    amplitudes: NDArray[np.floating] | None = None     # shape (n_wl, n_species)
    amp_errors: NDArray[np.floating] | None = None     # shape (n_wl, n_species)

    # Fit config / metadata
    lam: float | None = None
    noise: NDArray[np.floating] | None = None          # per-wavelength σ(λ), shape (n_wl,)
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def kinetics_linear(self) -> dict[str, float]:
        names: Sequence[str] = self.kinetic_model.param_names()
        return {str(n): float(np.exp(v)) for n, v in zip(names, self.beta)}

    def kinetics_linear_errors(self) -> dict[str, float] | None:
        if self.sd_beta is None:
            return None
        names: Sequence[str] = self.kinetic_model.param_names()
        vals = np.exp(self.beta)
        errs = vals * self.sd_beta  # delta(exp(beta)) ≈ exp(beta)*delta(beta)
        return {str(n): float(e) for n, e in zip(names, errs)}

    def summary(
        self,
        *,
        style: str = "brief",
        digits: int = 3,
        max_params: int | None = None,
    ) -> str:
        """
        Return a human-readable summary string.

        style:
          - "brief": journal-leaning, minimal
          - "debug": includes extra run/context info
        """

        style = str(style).lower()
        if style not in {"brief", "debug"}:
            raise ValueError("style must be 'brief' or 'debug'")

        model_name = type(self.kinetic_model).__name__
        param_names = list(map(str, self.kinetic_model.param_names()))
        n_params = len(param_names)

        # Shapes
        n_times = int(self.times.size) if self.times is not None else (int(self.traces.shape[0]) if self.traces is not None else None)
        n_wl = int(self.wavelengths.size) if self.wavelengths is not None else (int(self.amplitudes.shape[0]) if self.amplitudes is not None else None)
        n_species = int(self.traces.shape[1]) if self.traces is not None else (int(self.amplitudes.shape[1]) if self.amplitudes is not None else None)

        # Points count
        n_points = None
        if (n_times is not None) and (n_wl is not None):
            n_points = n_times * n_wl

        lines: list[str] = []
        lines.append("Global kinetic fit summary")
        lines.append(f"Model: {model_name}")
        if n_species is not None:
            lines.append(f"Species: {n_species}")
        lines.append(f"Kinetic params: {n_params} (fit in log-space; reported as exp(beta))")

        if n_points is not None:
            lines.append(f"Data: n_times={n_times}, n_wavelengths={n_wl}, n_points={n_points}")
        else:
            parts = []
            if n_times is not None:
                parts.append(f"n_times={n_times}")
            if n_wl is not None:
                parts.append(f"n_wavelengths={n_wl}")
            if parts:
                lines.append("Data: " + ", ".join(parts))

        if style == "debug":
            if self.lam is not None:
                lines.append(f"Tikhonov λ: {self.lam:g}")
            if self.noise is not None and self.noise.size:
                nz = self.noise[np.isfinite(self.noise)]
                if nz.size:
                    lines.append(
                        "Noise σ(λ): "
                        f"min={_fmt(float(nz.min()), digits)}, "
                        f"median={_fmt(float(np.median(nz)), digits)}, "
                        f"max={_fmt(float(nz.max()), digits)}"
                    )

        # ---- Kinetic parameter table ----
        vals = np.exp(self.beta)
        errs = (vals * self.sd_beta) if self.sd_beta is not None else None

        lines.append("")
        lines.append("Kinetic parameters (linear space):")

        n_show = n_params if max_params is None else min(n_params, int(max_params))
        for i in range(n_show):
            name = param_names[i]
            v = float(vals[i])
            if errs is None:
                lines.append(f"  {name:>12s} = {_fmt(v, digits)}")
            else:
                e = float(errs[i])
                rel = (e / abs(v)) if v != 0 else np.inf
                # Flag pathological uncertainties without being noisy
                flag = "  [unstable]" if rel > 1e3 else ""
                lines.append(
                    f"  {name:>12s} = {_fmt(v, digits)} ± {_fmt(e, digits)}  "
                    f"(rel {_fmt(100*rel, digits)}%){flag}"
                )

        if n_show < n_params:
            lines.append(f"  ... ({n_params - n_show} more)")

        # ---- Diagnostics ----
        if self.diagnostics:
            lines.append("")
            lines.append("Fit diagnostics:")
            # print in a stable order if present
            preferred = ["chi2_red", "r2", "rmse", "aic", "aicc", "bic"]
            printed = set()
            for k in preferred:
                if k in self.diagnostics:
                    lines.append(f"  {k:>8s} = {_fmt(float(self.diagnostics[k]), digits)}")
                    printed.add(k)
            # any remaining
            for k, v in self.diagnostics.items():
                if k not in printed:
                    try:
                        lines.append(f"  {str(k):>8s} = {_fmt(float(v), digits)}")
                    except Exception:
                        # ignore non-floats
                        pass

        # ---- Amplitudes metadata ----
        if self.amplitudes is not None:
            lines.append("")
            lines.append(f"Amplitudes: shape={tuple(self.amplitudes.shape)} (wavelengths × species)")
            try:
                sp = list(map(str, self.kinetic_model.species_names()))
                if sp:
                    lines.append("Species names: " + ", ".join(sp))
            except Exception:
                pass
            if (self.amp_errors is not None) and (self.amp_errors.shape == self.amplitudes.shape):
                lines.append("Amplitude errors: available (1σ per wavelength/species)")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GlobalFitResult:
        """
        Adapter for older dict-based fit_global_kinetics return.
        Allows you to start using GlobalFitResult.summary() immediately.
        """
        odr_out = d.get("odr", None)
        cache = d.get("_cache", {}) if isinstance(d.get("_cache", {}), dict) else {}

        kinetic_model = cache.get("kinetic_model", None)
        beta = cache.get("beta", None)
        if beta is None and odr_out is not None:
            beta = odr_out.beta

        return cls(
            kinetic_model=kinetic_model,
            beta=np.asarray(beta, dtype=float),
            sd_beta=getattr(odr_out, "sd_beta", None),
            cov_beta=getattr(odr_out, "cov_beta", None),
            odr_output=odr_out,
            times=np.asarray(cache.get("times", None), dtype=float) if cache.get("times", None) is not None else None,
            wavelengths=np.asarray(d.get("amplitudes", {}).get("x", None), dtype=float) if isinstance(d.get("amplitudes", None), dict) else None,
            traces=np.asarray(cache.get("traces", None), dtype=float) if cache.get("traces", None) is not None else None,
            amplitudes=np.asarray(d.get("amplitudes", {}).get("values", None), dtype=float) if isinstance(d.get("amplitudes", None), dict) and d["amplitudes"].get("values", None) is not None else None,
            amp_errors=np.asarray(d.get("amplitudes", {}).get("errors", None), dtype=float) if isinstance(d.get("amplitudes", None), dict) and d["amplitudes"].get("errors", None) is not None else None,
            lam=cache.get("lam", None),
            noise=cache.get("noise", None),
            diagnostics=d.get("diagnostics", {}) if isinstance(d.get("diagnostics", {}), dict) else {},
        )

@dataclass(frozen=True)
class GlobalFitResult:
    # Public, stable
    kinetics: dict[str, float]
    kinetics_errors: dict[str, float]

    amplitudes: NDArray[np.floating]          # (n_wl, n_species)
    amplitude_errors: NDArray[np.floating]    # (n_wl, n_species)
    species: list[str]
    wavelengths: NDArray[np.floating]

    diagnostics: dict[str, float]

    # Semi-public
    backend: dict[str, Any]

    # Internal (explicitly documented)
    _cache: dict[str, Any]

'''
