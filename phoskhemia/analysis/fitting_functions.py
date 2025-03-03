import numpy as np
from phoskhemia.utils.typing import ArrayFloatAny

def exponential_decay( 
        t: ArrayFloatAny,
        tau1: float,
        a1: float,
        b: float=0.0,
    ) -> ArrayFloatAny:
    """
    Monoexponential decay for data fitting.
    
    Args:
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.
        tau1 (float): Decay time constant. 
        a1 (float): Amplitude.
        b (float): Intercept/offset.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return a1 * np.exp(-t / tau1) + b


def biexponential_decay(
        t: ArrayFloatAny,
        tau1: float,
        tau2: float,
        a1: float,
        a2: float,
        b: float=0.0,
    ) -> ArrayFloatAny:
    """
    Biexponential decay for data fitting.

    Args:
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.
        tau1 - tau2 (float): Decay time constants 1 - 2. 
        a1 - a2 (float): Amplitudes 1 - 2.
        b (float): Intercept/offset.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) + b)

def triexponential_decay(
        t: ArrayFloatAny,
        tau1: float,
        tau2: float,
        tau3: float,
        a1: float,
        a2: float,
        a3: float,
        b: float=0.0,
    ) -> ArrayFloatAny:
    """
    Triexponential decay for data fitting.

    Args:
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.
        tau1 - tau3 (float): Decay time constants 1 - 3. 
        a1 - a3 (float): Amplitudes 1 - 3.
        b (float): Intercept/offset.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) + b)

def tetraexponential_decay(
        t: ArrayFloatAny,
        tau1: float,
        tau2: float,
        tau3: float,
        tau4: float,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
        b: float=0.0,
    ) -> ArrayFloatAny:
    """
    Tetraexponential decay for data fitting.

    Args:
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.
        tau1 - tau4 (float): Decay time constants 1 - 4. 
        a1 - a4 (float): Amplitudes 1 - 4.
        b (float): Intercept/offset.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) 
            + a4 * np.exp(-t / tau4) + b)

def pentaexponential_decay(
        t: ArrayFloatAny,
        tau1: float,
        tau2: float,
        tau3: float,
        tau4: float,
        tau5: float,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
        a5: float,
        b: float=0.0,
    ) -> ArrayFloatAny:
    """
    Pentaexponential decay for data fitting.

    Args:
        t (ArrayFloatAny): One-dimensional array of 
            values to evaluate expression at.
        tau1 - tau5 (float): Decay time constants 1 - 5. 
        a1 - a5 (float): Amplitudes 1 - 5.
        b (float): Intercept/offset.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return (a1 * np.exp(-t / tau1) 
            + a2 * np.exp(-t / tau2) 
            + a3 * np.exp(-t / tau3) 
            + a4 * np.exp(-t / tau4)
            + a5 * np.exp(-t / tau5) + b)

def n_exponential_decay(
        t: ArrayFloatAny,
        *args: float, 
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
    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            args = args[0]

    assert len(args) >= 2, "Must specify at least one amplitude and time constant"

    if len(args) % 2 == 0:
        b: float = 0

    else:
        b: float = args[-1]

    beta = args[:-1]
    index: int = len(beta) // 2
    taus: tuple[float, ...] = beta[:index]
    a_vals: tuple[float, ...] = beta[index:]

    return np.sum(
        [a * np.exp(-t / tau) for a, tau in zip(a_vals, taus)], axis=0
    ) + b

def stretched_exponential_decay(
        t: ArrayFloatAny,
        tau1: float,
        a1: float,
        b: float,
        alpha: float,
    ) -> ArrayFloatAny:
    """
    General stretched exponential decay function. While this can sometimes be 
    approximated with a triexponential decay, the stretched exponential has fewer
    fitting parameters. 

    Args:
        t (ArrayFloatAny): One-dimensional array of values to evaluate the function.
        tau1 (float): Time-dependent time coefficient. 
        a1 (float): Amplitude.
        b (float): Intercept/offset.
        alpha (float): Parameter responsible for stretching (compressing) the 
            decay when between 0 and 1 (>1). Related to the probability distribution
            of time constants.

    Returns:
        ArrayFloatAny: One-dimensional array of values.
    """

    return a1 * np.exp(-np.power(t / tau1, alpha)) + b



