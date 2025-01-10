import numpy as np
from typing import Any

ArrayFloatAny = np.ndarray[Any, np.dtype[np.float[Any]]]

def exponential_decay(
        beta: tuple[float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:

    tau1, a1, b = beta
    return a1 * np.exp(-t / tau1) + b

def biexponential_decay(
        beta: tuple[float, float, float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:

    tau1, tau2, a1, a2, b = beta
    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) + b)

def triexponential_decay(
        beta: tuple[float, float, float, 
                    float, float, float, float], 
        t: ArrayFloatAny,
    ) -> ArrayFloatAny:

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

    if len(beta) % 2 == 0:
        b = 0

    else:
        b = beta[-1]

    index = len(beta[:-1]) // 2
    taus = beta[:index]
    a_vals = beta[index:]

    return np.sum(
        [a * np.exp(-t / tau) for a, tau in zip(a_vals, taus)], axis=0
    ) + b


