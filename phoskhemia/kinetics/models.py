from __future__ import annotations

import numpy as np
import scipy as sp
from phoskhemia.utils.typing import ArrayFloatAny
from phoskhemia.kinetics.base import KineticModel
from phoskhemia.kinetics.parameterizations import simplex_weights_from_unconstrained

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

class ExponentialModel(KineticModel):
    """
    Single exponential model.
    """
    def n_params(self):
        return 2

    def param_names(self):
        return ["τ", "β"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau, b = np.exp(beta)
        a_star = np.exp(-times / tau) + b
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class BiexponentialModel(KineticModel):
    """
    Bi-exponential model.
    """
    def n_params(self):
        return 3

    def param_names(self):
        return ["τ₁", "τ₂", "β"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau1, tau2, b = np.exp(beta)
        a_star = b * np.exp(-times / tau1) + (1 - b) * np.exp(-times / tau2)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class TriexponentialModel(KineticModel):
    """
    Tri-exponential model.
    """
    def n_params(self):
        return 5

    def param_names(self):
        return ["τ₁", "τ₂", "τ₃", "α", "β"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau1, tau2, tau3 = np.exp(beta[:3])
        w1, w2, w3 = simplex_weights_from_unconstrained(beta[3:])
        
        a_star = w1 * np.exp(-times / tau1) + w2 * np.exp(-times / tau2) + w3 * np.exp(-times / tau3)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class TTAKineticModel(KineticModel):
    """
    Triplet–triplet annihilation kinetic model.
    """

    def n_params(self):
        return 3

    def param_names(self):
        return ["τᵣ", "τ₃ₐ", "k₂"]

    def species_names(self):
        return ["[³A*]", "[³S*]"]

    def solve(self, times, beta):
        tr, t3a, k2 = np.exp(beta)
        times = np.asarray(times, dtype=float)

        # Sensitizer population
        C_S = np.exp(-times / tr)

        def rhs(t, y):
            A = y[0]
            S = np.exp(-t / tr)
            dA = (1.0 / tr) * S - (1.0 / t3a) * A - k2 * A * A
            return [dA]

        sol = sp.integrate.solve_ivp(
            rhs,
            (times[0], times[-1]),
            y0=[0.0],
            t_eval=times,
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError("ODE solver failed")

        C_A = sol.y[0]

        return np.column_stack([C_A, C_S])
