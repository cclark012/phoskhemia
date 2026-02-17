from __future__ import annotations

import numpy as np
import scipy as sp
from phoskhemia.utils.typing import ArrayFloatAny
from phoskhemia.kinetics.base import KineticModel
from phoskhemia.kinetics.parameterizations import simplex_weights_from_unconstrained

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
        # tau1, tau2, b = np.exp(beta)
        tau1, tau2, b = (beta)
        a_star = b * np.exp(-times / tau1) + (1 - b) * np.exp(-times / tau2)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "linear"

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
        w1, w2, w3 = simplex_weights_from_unconstrained(beta[3:], k=3)
        
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

class StretchedExponentialModel(KineticModel):
    """
    Stretched Exponential model.
    """
    def n_params(self):
        return 3

    def param_names(self):
        return ["τ", "α", "β"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau1, a, b = np.exp(beta)
        
        a_star = (times ** (b - 1)) * np.exp(((a ** b) - (a + times / tau1) ** b))
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class PowerLawModel(KineticModel):
    """
    Stretched Exponential model.
    """
    def n_params(self):
        return 1

    def param_names(self):
        return ["τ"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau1 = np.exp(beta)
        k = 1 / tau1
        
        a_star = 1 / (k * times + 1)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"
