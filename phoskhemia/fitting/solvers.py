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
    - Standard deviations must be finite and strictly positive. Mask excluded
        observations before fitting rather than assigning them invalid sigmas.
    """
    s = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        w = np.square(1.0 / s)
    if np.any(w <= 0):
        raise ValueError("sigma is outside the representable weight range")
    return float(w) if w.ndim == 0 else w


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
        task: str = "explicit-ODR",
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
            task=task,
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
    """Bounded, fixed-x least squares using SciPy.

    Covariance is a local linear approximation in optimizer coordinates.
    Projected linear coefficients count toward residual degrees of freedom.
    Rank is assessed after normalizing Jacobian columns, so changing seconds
    to nanoseconds does not itself cause a spurious rank failure.
    """

    name = "least_squares"

    def fit(
        self, *, f, xdata, ydata, beta0, weight_x=None, weight_y=None,
        bounds=None, fix_beta=None, fix_x=None, maxit=50, report="none",
        diff_scheme="forward", absolute_sigma=False, n_projected=0.0,
        n_observations=None, covariance=True, **kwargs,
    ) -> SolverResult:

        from scipy.optimize import least_squares

        if weight_x is not None or (fix_x is not None and not np.all(fix_x)):
            raise ValueError("LeastSquaresSolver requires fixed x values")

        x = np.asarray(xdata, dtype=float)
        y = np.asarray(ydata, dtype=float)
        b0 = np.asarray(beta0, dtype=float)

        if b0.ndim != 1 or not np.all(np.isfinite(b0)):
            raise ValueError("beta0 must be a finite one-dimensional array")
        if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("ydata must be a nonempty finite one-dimensional array")
        if not np.all(np.isfinite(x)):
            raise ValueError("xdata must be finite")
        if int(maxit) != maxit or maxit < 1:
            raise ValueError("maxit must be a positive integer")

        fixed = np.zeros(b0.size, dtype=bool) if fix_beta is None else np.asarray(fix_beta)
        if fixed.shape != b0.shape or fixed.dtype.kind != "b":
            raise ValueError("fix_beta must be a boolean array matching beta0")

        free = ~fixed
        if bounds is None:
            lower, upper = np.full_like(b0, -np.inf), np.full_like(b0, np.inf)
        else:
            lower = (
                np.full_like(b0, -np.inf) if bounds[0] is None 
                else np.broadcast_to(np.asarray(bounds[0], dtype=float), b0.shape).copy()
            )
            upper = (
                np.full_like(b0, np.inf) if bounds[1] is None 
                else np.broadcast_to(np.asarray(bounds[1], dtype=float), b0.shape).copy()
            )

        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("invalid parameter bounds")
        if np.any(lower[free] >= upper[free]):
            raise ValueError("free parameters require ordered bounds; use fix_beta to fix a parameter")
        if np.any(b0 < lower) or np.any(b0 > upper):
            raise ValueError("beta0 lies outside parameter bounds")

        weight = (
            np.ones_like(y) if weight_y is None else 
            np.broadcast_to(np.asarray(weight_y, dtype=float), y.shape)
        )
        if np.any(~np.isfinite(weight)) or np.any(weight <= 0):
            raise ValueError("weight_y must be finite and strictly positive")

        sqrt_weight = np.sqrt(weight)
        if kwargs.get("loss", "linear") != "linear":
            raise ValueError("only loss='linear' is supported by this variable-projection solver")
        if kwargs.get("method", "trf") == "lm":
            raise ValueError("use method='trf' or 'dogbox'; model bounds are required")
        if "jac" in kwargs:
            raise ValueError("use diff_scheme rather than jac for the profiled objective")

        schemes = {"forward": "2-point", "central": "3-point", "2-point": "2-point", "3-point": "3-point"}
        if diff_scheme not in schemes:
            raise ValueError("diff_scheme must be 'forward' or 'central'")

        options = dict(kwargs)
        x_scale = options.pop("x_scale", "jac")
        if not isinstance(x_scale, str):
            x_scale = np.broadcast_to(np.asarray(x_scale, dtype=float), b0.shape)[free]

        # SciPy's default uses an absolute step floor of sqrt(eps), which is
        # large for rates expressed per nanosecond. A relative step preserves
        # the scale of each nonzero optimizer coordinate.
        default_step = (
            np.cbrt(np.finfo(float).eps) if schemes[diff_scheme] == "3-point" 
            else np.sqrt(np.finfo(float).eps)
        )
        diff_step = options.pop("diff_step", default_step)
        if np.ndim(diff_step):
            diff_step = np.broadcast_to(np.asarray(diff_step, dtype=float), b0.shape)[free]

        options["diff_step"] = diff_step
        options.setdefault("max_nfev", int(maxit))
        options.setdefault("verbose", 0 if report == "none" else 1)

        def unpack(values):
            beta = b0.copy()
            beta[free] = values
            return beta

        def residual(values):
            prediction = np.asarray(f(x, unpack(values)), dtype=float)
            if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
                raise ValueError("model must return finite predictions matching ydata")

            result = (prediction - y) * sqrt_weight
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("weighted residuals overflowed")

            return result

        if np.any(free):
            raw = least_squares(residual, b0[free], bounds=(lower[free], upper[free]),
                                jac=schemes[diff_scheme], x_scale=x_scale, **options)

            beta, resid, jac = unpack(raw.x), raw.fun, raw.jac
            success, message = bool(raw.success), str(raw.message)
            nfev, njev = int(raw.nfev), raw.njev
            active = np.zeros_like(b0, dtype=int)
            active[free] = raw.active_mask

        else:
            raw = None
            beta, resid = b0.copy(), residual(np.empty(0))
            jac = np.empty((y.size, 0))
            success, message, nfev, njev = True, "All kinetic parameters were fixed.", 1, 0
            active = np.zeros_like(b0, dtype=int)

        n_linear = float(n_projected(beta) if callable(n_projected) else n_projected)
        n_obs = y.size if n_observations is None else int(n_observations)
        if not np.isfinite(n_linear) or not 0 <= n_linear <= n_obs:
            raise ValueError("n_projected must be finite and between zero and n_observations")
        if not 0 < n_obs <= y.size or (n_observations is not None and n_obs != n_observations):
            raise ValueError("n_observations must be a positive integer no larger than ydata.size")

        n_free = int(np.count_nonzero(free))
        dof = float(n_obs - n_free - n_linear)
        rss = float(resid @ resid)
        res_var = rss / dof if dof > 0 else float("nan")
        cov = None
        rank = 0
        condition = float("nan")
        reason = None
        if n_free:
            scales = np.linalg.norm(jac, axis=0)
            safe_scales = np.where(scales > 0, scales, 1.0)
            _, singular, vt = np.linalg.svd(jac / safe_scales, full_matrices=False)
            threshold = np.finfo(float).eps * max(jac.shape) * (singular[0] if singular.size else 0)
            rank = int(np.count_nonzero(singular > threshold))

            if rank == n_free:
                condition = float(singular[0] / singular[-1])
            if not covariance:
                reason = "Covariance disabled for the penalized objective."
            elif rank < n_free:
                reason = "The kinetic Jacobian is rank deficient."
            elif dof <= 0:
                reason = "No positive residual degrees of freedom."
            else:
                inv_j = (vt.T / singular) / safe_scales[:, None]
                small_cov = inv_j @ inv_j.T
                if not absolute_sigma:
                    small_cov *= res_var
                cov = np.zeros((b0.size, b0.size))
                cov[np.ix_(free, free)] = small_cov

        else:
            cov = np.zeros((b0.size, b0.size)) if covariance else None
        if not success:
            cov = None
            reason = "Optimizer did not converge; confidence intervals are unavailable."
        info = {
            "backend": self.name, "success": success, "stopreason": message,
            "nfev": nfev, "njev": njev, "raw": raw, "n_free_kinetics": n_free,
            "n_projected": n_linear, "n_observations": n_obs, "dof": dof,
            "jacobian_rank": rank, "scaled_jacobian_condition": condition,
            "active_bounds": active, "absolute_sigma": bool(absolute_sigma),
            "covariance_unavailable_reason": reason,
        }
        sd = None if cov is None else np.sqrt(np.maximum(np.diag(cov), 0))

        return SolverResult(beta, cov, sd, res_var, info)


def get_solver(name: str) -> Solver:
    """Select a backend without making defaults depend on optional installs."""
    if name in {"auto", "least_squares", "scipy", "lsq"}:
        return LeastSquaresSolver()
    if name == "odrpack":
        return ODRPackSolver()
    raise ValueError(f"Unknown solver {name!r}; use 'least_squares' or 'odrpack'")
