
import numpy as np
from phoskhemia.utils.typing import ArrayFloatAny

def exponential_decay(
        beta: tuple[float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    Monoexponential decay for data fitting.

    Args:
        beta (tuple[float, float, float]): 3-tuple of decay 
            time constant, amplitude, and intercept/offset.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """
    tau1: float

    tau1, a1, b = beta
    return a1 * np.exp(-t / tau1) + b

def biexponential_decay(
        beta: tuple[float, float, float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    Biexponential decay for data fitting.

    Args:
        beta (tuple[float, float, float, float, float]): 5-tuple of decay 
            time constant 1 and 2, amplitude 1 and 2, and intercept/offset.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    tau1, tau2, a1, a2, b = beta
    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) + b)

def triexponential_decay(
        beta: tuple[float, float, float, 
                    float, float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    Triexponential decay for data fitting.

    Args:
        beta (tuple[float, float, float, 
                    float, float, float, float]): 7-tuple of decay time 
                    constants 1 - 3, amplitudes 1 - 3, and intercept/offset.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    tau1, tau2, tau3, a1, a2, a3, b = beta
    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) + b)

def tetraexponential_decay(
        beta: tuple[float, float, float, 
                    float, float, float, 
                    float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    Tetraexponential decay for data fitting.

    Args:
        beta (tuple[float, float, float, 
                    float, float, float, 
                    float, float, float]): 9-tuple of decay time 
                    constants 1 - 4, amplitudes 1 - 4, and intercept/offset.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    tau1, tau2, tau3, tau4, a1, a2, a3, a4, b = beta
    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) 
            + a4 * np.exp(-t / tau4) + b)

def pentaexponential_decay(
        beta: tuple[float, float, float, float, 
                    float, float, float, float, 
                    float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    Pentaexponential decay for data fitting.

    Args:
        beta (tuple[float, float, float, float, 
                    float, float, float, float, 
                    float, float, float]): 11-tuple of decay time 
                    constants 1 - 5, amplitudes 1 - 5, and intercept/offset.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    tau1, tau2, tau3, tau4, tau5, a1, a2, a3, a4, a5, b = beta
    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) 
            + a4 * np.exp(-t / tau4)
            + a5 * np.exp(-t / tau5) + b)

def n_exponential_decay(
        beta: tuple[float, ...], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:
    """
    n-exponential decay for data fitting.

    Args:
        beta (tuple[float, ...]): n-tuple of decay time 
            constants 1 - n, amplitudes 1 - n, and intercept/offset.
            If the input list beta is an even number of values, the 
            intercept value is set to 0, while an odd number of values
            will have the last value be the intercept.
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    if len(beta) % 2 == 0:
        b: float = 0

    else:
        b: float = beta[-1]

    beta = beta[:-1]
    index: int = len(beta) // 2
    taus: tuple[float, ...] = beta[:index]
    a_vals: tuple[float, ...] = beta[index:]

    return np.sum(
        [a * np.exp(-t / tau) for a, tau in zip(a_vals, taus)], axis=0
    ) + b


