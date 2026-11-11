"""Analytical evaluation of exponentially sensitized triplet populations.

The Riccati equation ``n' = P*exp(-beta*t) - gamma*n - q*n**2`` has
``n = (P/beta)*w``, where ``s = beta*t``, ``nu = gamma/beta`` and
``eta = q*P/beta**2``.  For zero initial population, the auxiliary function
``u'' + nu*u' = eta*exp(-s)*u`` gives ``w = u'/(eta*u)``.

We evaluate positive analytical series for ``u`` and ``u'/eta`` using small
matrix exponentials.  No numerical integration or late-time source cutoff is
used.  Truncating after order M has the uniform absolute error bound

    |w - w_M| <= eta**M / ((M+1)! M!)
                 / (1 - eta/((M+1)*(M+2))).

The bound applies when its denominator is positive and concerns series
truncation, not floating-point evaluation of the matrix exponential.  In
particular, requesting a smaller tolerance does not guarantee a smaller total
roundoff error.  The experimental regime eta <= 10 requires at most 18 terms
for a truncation tolerance of 1e-14.
"""

from __future__ import annotations

from math import lgamma, log, log1p
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.special import exprel

__all__ = ["analytic_shape", "triplet_population", "series_order"]


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _times(values: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError("times must be real")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be real numbers") from exc
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("times must be finite and nonnegative, relative to time zero")
    return result


def _controls(tol: float, order: int | None) -> tuple[float, int | None]:
    tol = _scalar(tol, "tol", positive=True)
    if tol >= 1:
        raise ValueError("tol must be smaller than one")
    if order is not None:
        try:
            if isinstance(order, (bool, np.bool_)):
                raise TypeError
            order = index(order)
        except TypeError as exc:
            raise ValueError("order must be a positive integer or None") from exc
        if order < 1:
            raise ValueError("order must be a positive integer or None")
    return tol, order


def _log_error_bound(eta: float, order: int) -> float:
    if eta == 0:
        return -np.inf
    ratio = eta / ((order + 1) * (order + 2))
    if ratio >= 1:
        return np.inf
    return (
        order * log(eta)
        - lgamma(order + 2)
        - lgamma(order + 1)
        - log1p(-ratio)
    )


def series_order(eta: float, tol: float = 1e-14) -> int:
    """Choose the smallest positive order satisfying the truncation bound.

    ``eta`` can be the upper bound of an intended fit domain. Passing that
    order to :func:`analytic_shape` then uses one fixed finite expression
    throughout the fit. Orders above 499 are intentionally not attempted.
    """
    eta = _scalar(eta, "eta")
    tol, _ = _controls(tol, None)
    for order in range(1, 500):
        if _log_error_bound(eta, order) <= log(tol):
            return order
    raise ValueError("eta requires too many analytical series terms")


def analytic_shape(
    times: ArrayLike,
    beta: float,
    gamma: float,
    eta: float,
    *,
    tol: float = 1e-14,
    order: int | None = None,
) -> NDArray[np.float64]:
    """Return normalized annihilator populations for zero population at t=0.

    Parameters
    ----------
    times
        Nonnegative times relative to the excitation time. Scalar and array
        inputs retain their shape; order and duplicates do not matter.
    beta, gamma
        Positive sensitizer decay rate and nonnegative annihilator decay rate,
        expressed in the reciprocal of the time unit used by ``times``.
    eta
        Nonnegative dimensionless annihilation strength ``q*P/beta**2``.
    tol
        Absolute analytical truncation tolerance for the normalized shape.
        This does not bound floating-point error in ``expm``.
    order
        Optional fixed series order. An insufficient order raises ValueError.

    Returns
    -------
    ndarray
        Normalized ``w`` with the same shape as ``times``. The dimensional
        concentration is ``(P/beta)*w``. The initial condition is always at
        physical time zero, even when the first requested time is later.
    """
    times = _times(times)
    beta = _scalar(beta, "beta", positive=True)
    gamma = _scalar(gamma, "gamma")
    eta = _scalar(eta, "eta")
    tol, order = _controls(tol, order)

    if times.size == 0:
        return np.empty_like(times)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = beta * times.ravel()
        nu = gamma / beta
    if not np.isfinite(s).all() or not np.isfinite(nu):
        raise FloatingPointError("Dimensionless time or decay rate is not representable")

    if eta == 0:
        with np.errstate(over="raise", invalid="raise"):
            result = (
                s * np.exp(-min(1.0, nu) * s)
                * exprel(-abs(1.0 - nu) * s)
            )
        return result.reshape(times.shape)

    if order is None:
        order = series_order(eta, tol)
    elif _log_error_bound(eta, order) > log(tol):
        raise ValueError("order is insufficient for the requested eta and tolerance")

    # All U states must precede all V states. Interleaving them makes T
    # triangular; a SciPy triangular shortcut loses precision for nearly
    # repeated diagonal values. The present ordering uses the general path.
    # U_j = exp(-j*s)*u_(M-j); V_j = exp(-j*s)*u'_(M-j)/eta.
    T = np.zeros((2 * order + 1, 2 * order + 1))
    j = np.arange(order + 1)
    T[j, j] = -j
    j = np.arange(order)
    T[order + 1 + j, order + 1 + j] = -(nu + j)
    T[j, order + 1 + j] = eta
    T[order + 1 + j, j + 1] = 1.0

    unique_times, inverse = np.unique(s, return_inverse=True)
    values = np.empty_like(unique_times)
    for i, si in enumerate(unique_times):
        if si == 0:
            values[i] = 0.0
            continue
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            E = expm(si * T)
            # Initial state is one in every U component, zero in every V.
            u = E[0, :order + 1].sum()
            v = E[order + 1, :order + 1].sum()
            values[i] = v / u
    if not np.isfinite(values).all():
        raise FloatingPointError("Analytical population evaluation produced nonfinite values")
    return values[inverse].reshape(times.shape)


def triplet_population(
    times: ArrayLike,
    P: float,
    beta: float,
    gamma: float,
    q: float,
    *,
    tol: float = 1e-14,
    order: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate ``n' = P*exp(-beta*t) - gamma*n - q*n**2``, ``n(0)=0``.

    With concentration in M and time in seconds, P is M/s, beta and gamma
    are 1/s, q is 1/(M*s), and the return value is M. Use consistent units;
    no implicit time or concentration conversion is performed. ``tol`` is
    a normalized-shape truncation tolerance, not a concentration tolerance.
    """
    times = _times(times)
    P = _scalar(P, "P")
    beta = _scalar(beta, "beta", positive=True)
    gamma = _scalar(gamma, "gamma")
    q = _scalar(q, "q")
    tol, order = _controls(tol, order)
    if P == 0:
        return np.zeros_like(times)
    # Divide first to avoid overflowing q*P or beta**2 unnecessarily.
    scale = P / beta
    eta = (q / beta) * scale
    if not np.isfinite(scale) or not np.isfinite(eta):
        raise FloatingPointError("The normalized concentration or eta is not representable")
    result = scale * analytic_shape(times, beta, gamma, eta, tol=tol, order=order)
    if not np.isfinite(result).all():
        raise FloatingPointError("The dimensional population is not representable")
    return result
