"""Fast, accuracy-checked sampling of the analytical sensitized-TTA solution.

The reference evaluator is an analytical finite matrix-exponential series.
This module evaluates that expression at Chebyshev nodes, interpolates within
short dimensionless-time intervals, and uses the exact source-free Riccati
solution after the exponential source has become negligible. The tail
approximation has a uniform normalized error bound of exp(-s_cut).

Interpolation is checked at independent interior points and is an
approximation, not a rigorously certified uniform error bound. Use the direct
analytical evaluator to validate demanding parameter regions.
"""

from __future__ import annotations

from math import log

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct
from scipy.special import exprel

from phoskhemia.kinetics.analytic import analytic_shape, _controls, _scalar, _times

__all__ = ["analytic_shape_fast"]


def _chebyshev_coefficients(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Chebyshev-Lobatto interpolation coefficients from node values.

    Parameters
    ----------
    values : NDArray[np.float64]
        Values ordered at cos(pi*j/degree), j=0,...,degree.

    Returns
    -------
    NDArray[np.float64]
        Coefficients for numpy.polynomial.chebyshev.chebval.
    """
    degree = values.size - 1
    coefficients = dct(values, type=1) / degree
    coefficients[0] *= 0.5
    coefficients[-1] *= 0.5

    return coefficients


def _segment_interpolant(
    left: float,
    right: float,
    beta: float,
    gamma: float,
    eta: float,
    order: int | None,
    degree: int,
    tolerance: float,
    depth: int,
    max_depth: int,
) -> list[tuple[float, float, NDArray[np.float64]]]:
    """Construct validated Chebyshev approximants on a dimensionless interval.

    Parameters
    ----------
    left, right : float
        Interval endpoints in dimensionless time, ``s=beta*t``.
    beta, gamma, eta : float
        Physical sensitizer and annihilator rates, plus nonlinear strength.
    order : int | None
        Optional fixed order of the reference analytical series.
    degree : int
        Degree of each local interpolating polynomial.
    tolerance : float
        Maximum allowed sampled validation error.
    depth, max_depth : int
        Current recursion depth and subdivision limit.

    Returns
    -------
    list[tuple[float, float, NDArray[np.float64]]]
        Intervals and their Chebyshev coefficients, in increasing time order.
    """
    indices = np.arange(degree + 1)
    nodes = np.cos(np.pi * indices / degree)
    sample_s = 0.5 * (left + right) + 0.5 * (right - left) * nodes
    sample_s = np.maximum(sample_s, 0.0)

    values = analytic_shape(
        sample_s / beta, beta, gamma, eta, order=order,
    )
    coefficients = _chebyshev_coefficients(values)

    # Independent validation points are deliberately not interpolation nodes.
    check_nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)
    check_s = 0.5 * (left + right) + 0.5 * (right - left) * check_nodes
    actual = analytic_shape(check_s / beta, beta, gamma, eta, order=order)
    predicted = np.polynomial.chebyshev.chebval(check_nodes, coefficients)
    error = float(np.max(np.abs(actual - predicted)))

    if error <= tolerance:
        return [(left, right, coefficients)]

    if depth >= max_depth:
        raise FloatingPointError(
            "Chebyshev interpolation did not meet its validation tolerance; "
            "use the direct analytical evaluator."
        )

    midpoint = 0.5 * (left + right)
    return (
        _segment_interpolant(
            left, midpoint, beta, gamma, eta, order,
            degree, tolerance, depth + 1, max_depth,
        )
        + _segment_interpolant(
            midpoint, right, beta, gamma, eta, order,
            degree, tolerance, depth + 1, max_depth,
        )
    )


def analytic_shape_fast(
    times: ArrayLike,
    beta: float,
    gamma: float,
    eta: float,
    *,
    tol: float = 1e-11,
    order: int | None = None,
    degree: int = 20,
    interval_width: float = 3.0,
    max_depth: int = 8,
) -> NDArray[np.float64]:
    """Evaluate the analytical normalized TTA population using fast sampling.

    This approximates the reference analytical solution without integrating
    the nonlinear differential equation. Chebyshev interpolation is checked
    against additional exact-series evaluations, but has no universal
    rigorous uniform-error guarantee. For ``s >= s_cut``, the omitted source
    contributes at most ``exp(-s_cut)`` to the population, for nonnegative
    decay rates and zero initial annihilator population.

    Parameters
    ----------
    times : ArrayLike
        Nonnegative times, in the same unit as reciprocal rate constants.
    beta, gamma : float
        Sensitizer and annihilator first-order decay rates, respectively.
    eta : float
        Dimensionless quadratic strength ``q*P/beta**2``.
    tol : float, default=1e-11
        Target absolute error of the normalized shape. This is a numerical
        interpolation target, not an unconditional error guarantee.
    order : int | None, default=None
        Reference analytical series order; None selects it automatically.
    degree : int, default=20
        Local Chebyshev polynomial degree.
    interval_width : float, default=3.0
        Initial width of each segment in ``s=beta*t`` coordinates.
    max_depth : int, default=8
        Maximum recursive subdivisions of a segment.

    Returns
    -------
    NDArray[np.float64]
        Approximate normalized annihilator population at all requested times.
    """
    times_array = _times(times)
    beta = _scalar(beta, "beta", positive=True)
    gamma = _scalar(gamma, "gamma")
    eta = _scalar(eta, "eta")
    tol, order = _controls(tol, order)

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 4:
        raise ValueError("degree must be an integer of at least four")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if not np.isfinite(interval_width) or interval_width <= 0:
        raise ValueError("interval_width must be finite and positive")

    if times_array.size == 0:
        return np.empty_like(times_array)

    s = beta * times_array.ravel()
    nu = gamma / beta

    if not np.isfinite(s).all() or not np.isfinite(nu):
        raise FloatingPointError("Dimensionless time or rate is not representable")

    if eta == 0:
        # No sampling or tail approximation is needed for the linear limit.
        result = (
            s * np.exp(-min(1.0, nu) * s)
            * exprel(-abs(1.0 - nu) * s)
        )
        return result.reshape(times_array.shape)

    # The remaining source after s_cut contributes at most exp(-s_cut).
    s_cut = max(0.0, -log(0.05 * tol))
    s_end = min(float(np.max(s)), s_cut)
    result = np.empty_like(s)

    # Use the reference series accuracy in addition to interpolation accuracy.
    # Restrict the initial sampling to intervals that contain a requested time.
    boundaries = np.arange(0.0, s_end, interval_width)
    boundaries = np.r_[boundaries, s_end]

    if boundaries.size == 1:
        result.fill(0.0)
    else:
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            mask = (s >= left) & (s <= right)
            if not np.any(mask):
                continue

            segments = _segment_interpolant(
                left, right, beta, gamma, eta, order,
                degree, tol * 0.2, 0, max_depth,
            )

            for segment_left, segment_right, coefficients in segments:
                selection = mask & (s >= segment_left) & (s <= segment_right)
                if not np.any(selection):
                    continue

                scaled = (
                    (2.0 * s[selection] - segment_left - segment_right)
                    / (segment_right - segment_left)
                )
                result[selection] = np.polynomial.chebyshev.chebval(scaled, coefficients)

    if np.any(s > s_end):
        w_end = float(analytic_shape(np.array([s_end / beta]), beta, gamma, eta, order=order)[0])
        mask = s > s_end
        delta = s[mask] - s_end
        exp_factor = np.exp(-nu * delta)
        denominator = 1.0 + eta * w_end * delta * exprel(-nu * delta)
        result[mask] = w_end * exp_factor / denominator

    if not np.isfinite(result).all():
        raise FloatingPointError("Accelerated analytical evaluation returned nonfinite values")

    # Roundoff in interpolation can produce tiny negative values at time zero.
    result[s == 0.0] = 0.0
    return result.reshape(times_array.shape)
