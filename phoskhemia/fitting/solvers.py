from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray


F64 = np.floating
ArrayF = NDArray[F64]

def weights_from_sigma(sigma: ArrayF | float) -> ArrayF | float:
    """
    Convert standard deviations σ to ODR weights w = 1/σ^2.

    Notes
    -----
    - ODRPACK/odrpack expects weights, not sigmas.
    - If σ contains zeros or non-finite values, weights are set to 0 at those entries.
    """
    if np.isscalar(sigma):
        s = float(sigma)
        if (not np.isfinite(s)) or (s <= 0.0):
            return 0.0
        return 1.0 / (s * s)

    s = np.asarray(sigma, dtype=float)
    w = np.zeros_like(s, dtype=float)
    good = np.isfinite(s) & (s > 0.0)
    w[good] = 1.0 / np.square(s[good])
    return w


def time_sigma_from_scope(
        times: ArrayF,
        *,
        ppm: float,
        jitter: float,
    ) -> ArrayF:
    """
    Estimate σ_t for oscilloscope timestamps.

    Model
    -----
    σ_t(t) = sqrt( (ppm * 1e-6 * |t|)^2 + jitter^2 )

    Parameters
    ----------
    times : array
        Time axis values (same units as jitter).
    ppm : float
        Long-term timebase stability/accuracy in ppm.
    jitter : float
        Short-term timing jitter (same units as times).

    Returns
    -------
    sigma_t : ndarray
        Per-sample time standard deviation.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    ppm = float(ppm)
    jitter = float(jitter)
    return np.sqrt(np.square(ppm * 1e-6 * np.abs(t)) + jitter * jitter)


@dataclass(frozen=True)
class SolverResult:
    """
    Backend-agnostic solver output.

    Contract
    --------
    - cov_beta is expected to be *scaled* (i.e., includes residual variance scaling if applicable).
      This matches your current scipy.odr behavior: cov_beta_scaled = cov_beta * res_var.
    """
    beta: ArrayF
    cov_beta: ArrayF | None
    sd_beta: ArrayF | None
    res_var: float | None
    info: dict[str, Any]


class Solver(Protocol):
    def fit(
        self,
        *,
        f: Callable[[ArrayF, ArrayF], ArrayF],
        xdata: ArrayF,
        ydata: ArrayF,
        beta0: ArrayF,
        weight_x: ArrayF | float | None = None,
        weight_y: ArrayF | float | None = None,
        bounds: tuple[ArrayF | None, ArrayF | None] | None = None,
        fix_beta: NDArray[np.bool_] | None = None,
        fix_x: NDArray[np.bool_] | None = None,
        maxit: int = 50,
        report: str = "none",
        diff_scheme: str = "forward",
    ) -> SolverResult: ...


def _import_odrpack():
    try:
        import odrpack  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ODRPACK backend requires optional dependency 'odrpack'. "
            "Install with: pip install phoskhemia[odr]"
        ) from e
    return odrpack


class ODRPackSolver:
    """
    ODRPACK backend using the standalone `odrpack` package.

    This uses odrpack.odr_fit(f, xdata, ydata, beta0, ...) returning OdrResult.
    See odrpack API reference. https://hugomvale.github.io/odrpack-python/reference/#odrpack.odr_fit
    """

    name: str = "odrpack"

    def fit(
        self,
        *,
        f: Callable[[ArrayF, ArrayF], ArrayF],
        xdata: ArrayF,
        ydata: ArrayF,
        beta0: ArrayF,
        weight_x: ArrayF | float | None = None,
        weight_y: ArrayF | float | None = None,
        bounds: tuple[ArrayF | None, ArrayF | None] | None = None,
        fix_beta: NDArray[np.bool_] | None = None,
        fix_x: NDArray[np.bool_] | None = None,
        maxit: int = 50,
        report: str = "none",
        diff_scheme: str = "forward",
    ) -> SolverResult:

        odrpack = _import_odrpack()

        x = np.asarray(xdata, dtype=float)
        y = np.asarray(ydata, dtype=float)
        b0 = np.asarray(beta0, dtype=float).reshape(-1)

        sol = odrpack.odr_fit(
            f,
            x,
            y,
            b0,
            weight_x=weight_x,
            weight_y=weight_y,
            bounds=bounds,
            task="explicit-ODR",
            fix_beta=fix_beta,
            fix_x=fix_x,
            diff_scheme=diff_scheme,
            report=report,
            maxit=int(maxit),
        )

        beta = np.asarray(sol.beta, dtype=float).reshape(-1)

        cov_beta_scaled = None
        sd_beta = None
        res_var = None

        # odrpack returns cov_beta and res_var (and sd_beta). 
        # https://hugomvale.github.io/odrpack-python/reference/#odrpack.OdrResult
        try:
            res_var = float(sol.res_var)
        except Exception:
            res_var = None

        try:
            cov = np.asarray(sol.cov_beta, dtype=float)
            if cov.size > 0:
                if res_var is not None and np.isfinite(res_var):
                    cov_beta_scaled = cov * res_var
                else:
                    cov_beta_scaled = cov
        except Exception:
            cov_beta_scaled = None

        try:
            sd = np.asarray(sol.sd_beta, dtype=float).reshape(-1)
            if sd.size > 0:
                sd_beta = sd
        except Exception:
            sd_beta = None

        info: dict[str, Any] = {
            "backend": self.name,
            "success": bool(getattr(sol, "success", True)),
            "stopreason": str(getattr(sol, "stopreason", "")),
            "info": int(getattr(sol, "info", -1)),
            "niter": int(getattr(sol, "niter", -1)),
            "nfev": int(getattr(sol, "nfev", -1)),
            "njev": int(getattr(sol, "njev", -1)),
            "raw": sol,  # backend-specific object retained for provenance/debugging
        }

        return SolverResult(
            beta=beta,
            cov_beta=cov_beta_scaled,
            sd_beta=sd_beta,
            res_var=res_var,
            info=info,
        )

class LeastSquaresSolver:
    pass

def get_solver(name: str):
    if name == 'auto':
        try:
            _import_odrpack()
            return ODRPackSolver()
        except ImportError:
            return LeastSquaresSolver()
    if name == 'odrpack':
        return ODRPackSolver()
