from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from numpy.typing import ArrayLike, NDArray

import numpy as np
from phoskhemia.kinetics.analytic import analytic_shape, _controls
from phoskhemia.kinetics.accelerated import analytic_shape_fast
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
        w1, w2, w3 = simplex_weights_from_unconstrained(beta[3:], k=3)
        
        a_star = w1 * np.exp(-times / tau1) + w2 * np.exp(-times / tau2) + w3 * np.exp(-times / tau3)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class TetraexponentialModel(KineticModel):
    """
    Tetra-exponential model.
    """
    def n_params(self):
        return 7

    def param_names(self):
        return ["τ₁", "τ₂", "τ₃", "τ₄", "α", "β", "γ"]

    def species_names(self):
        return "[³A*]"

    def solve(self, times, beta):
        tau1, tau2, tau3, tau4 = np.exp(beta[:4])
        w1, w2, w3, w4 = simplex_weights_from_unconstrained(beta[4:], k=4)
        
        a_star = w1 * np.exp(-times / tau1) + w2 * np.exp(-times / tau2) + w3 * np.exp(-times / tau3) + w4 * np.exp(-times / tau4)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

class NexponentialModel(KineticModel):
    # TODO - Allow n > 9 (even though that is likely unrealistic in real scenarios)
    def __init__(self, n: int = 2):
        n = int(n)
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if n > 9:
            raise ValueError("exponential with more than 9 components are not currently supported")

        self.n = n
        self.parameters = 2 * n - 1
    
    def n_params(self):
        return self.parameters
    
    def param_names(self):
        params = ["τ₁"]
        params.extend([("τ" + str(i)).replace(str(i), f"{chr(int(f"208{i}", 16))}") for i in range(2, self.n + 1)])
        params.extend([(str(i)).replace(str(i), f"{chr(int(f"03B{i}", 16))}") for i in range(1, self.n)])
        return params

    def species_names(self):
        return "[³A*]"
    
    def solve(self, times, beta):
        taus = np.exp(beta[:self.n])
        if self.n > 1:
            ws = simplex_weights_from_unconstrained(beta[self.n:], k=self.n)
            a_star = np.sum(ws[:, None] * np.exp(-times / taus[:, None]), axis=0)
        else:
            a_star = np.exp(-times / taus)
        return np.atleast_2d(a_star).T
    
    def parameterization(self):
        return "log"

def _tta_settings(time_unit, tolerance, order):
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("time_unit must be a nonempty unit label")
    _controls(tolerance, order)


def _tta_input(times, parameters):
    if np.iscomplexobj(times) or np.iscomplexobj(parameters):
        raise ValueError("times and kinetic parameters must be real")
    times = np.asarray(times, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must have shape (n_times,)")
    if parameters.shape != (3,):
        raise ValueError("kinetic parameters must have shape (3,)")
    if not np.isfinite(parameters).all():
        raise ValueError("kinetic parameters must be finite")
    return times, parameters


def _tta_basis(
    times: NDArray[np.float64],
    rate: float,
    gamma: float,
    eta: float,
    tolerance: float,
    order: int | None,
    evaluation: Literal["direct", "accelerated"] = "direct",
    interpolation_tolerance: float = 1e-11,
) -> NDArray[np.float64]:
    """Build the sensitizer and annihilator kinetic basis.

    Parameters
    ----------
    times : NDArray[np.float64]
        Nonnegative delays measured in a consistent time unit.
    rate, gamma : float
        Sensitizer and annihilator first-order decay rates.
    eta : float
        Dimensionless nonlinear annihilation strength.
    tolerance : float
        Direct analytical-series truncation tolerance.
    order : int | None
        Optional fixed series order.
    evaluation : Literal["direct", "accelerated"]
        Direct series evaluation or accelerated interpolation.
    interpolation_tolerance : float
        Absolute error target for interpolation validation.

    Returns
    -------
    NDArray[np.float64]
        Two-column kinetic basis, ordered [annihilator, sensitizer].
    """
    if evaluation == "accelerated":
        annihilator = analytic_shape_fast(
            times, rate, gamma, eta,
            tol=interpolation_tolerance,
            order=order,
        )

    else:
        annihilator = analytic_shape(
            times, rate, gamma, eta,
            tol=tolerance,
            order=order,
        )

    sensitizer = np.exp(-rate * times)
    return np.column_stack([annihilator, sensitizer])


@dataclass(frozen=True)
class SensitizedTTAModel(KineticModel):
    """Analytical sensitization and mixed-order triplet decay.

    ``solve(times, [beta, gamma, eta])`` accepts parameters in natural linear
    space: beta > 0, gamma >= 0, and eta >= 0. It returns columns ``[A, S]``
    with ``S = exp(-beta*t)`` and ``A = n/(P/beta)`` for the physical equation
    ``n' = P*exp(-beta*t) - gamma*n - q*n**2``, ``n(0)=0``.

    Here ``eta = q*P/beta**2`` is dimensionless. Physical ``q`` cannot be
    inferred from eta alone when the concentration and spectral amplitude
    scales are unknown. Two independent signed spectral amplitudes belong
    in the global fit, one for each returned column.

    ``time_unit`` labels parameter reports only; it does not convert values.
    The rates supplied to ``solve`` must use the reciprocal of the time unit
    of ``times``. ``series_tolerance`` controls analytical truncation only.
    An optional fixed ``series_order`` must satisfy that tolerance throughout
    the fitted domain. Unordered, repeated, and singleton time grids work;
    the initial condition remains at t=0 when a fit omits the early data.
    """

    time_unit: str = "s"
    series_tolerance: float = 1e-14
    series_order: int | None = None
    evaluation: Literal["direct", "accelerated"] = "direct"
    interpolation_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        _tta_settings(self.time_unit, self.series_tolerance, self.series_order)
        if self.evaluation not in ("direct", "accelerated"):
            raise ValueError("evaluation must be 'direct' or 'accelerated'")
        _controls(self.interpolation_tolerance, None)

    def n_params(self) -> int:
        return 3

    def param_names(self) -> list[str]:
        return ["beta", "gamma", "eta"]

    def param_units(self) -> list[str | None]:
        return [f"{self.time_unit}^-1", f"{self.time_unit}^-1", None]

    def param_descriptions(self) -> list[str]:
        return [
            "Sensitizer decay rate: k_3S + k_TET*[A]",
            "First-order annihilator triplet decay rate: k_3A",
            "Dimensionless annihilation strength: k_2*P/beta^2",
        ]

    def species_names(self) -> list[str]:
        return ["[³A*]", "[³S*]"]

    def parameterization(self) -> Literal["linear"]:
        return "linear"

    def parameter_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            np.array([np.finfo(float).tiny, 0.0, 0.0]),
            np.full(3, np.inf),
        )

    def solve(
        self,
        times: ArrayLike,
        beta: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return the normalized analytical population basis at each time.

        Parameters
        ----------
        times : ArrayLike
            Times measured relative to excitation time zero.
        beta : ArrayLike
            Three model parameters [sensitizer decay, annihilator decay, eta].

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_times, 2): annihilator and sensitizer traces.
        """
        times, parameters = _tta_input(times, beta)
        rate, gamma, eta = parameters
        return _tta_basis(
            times, rate, gamma, eta,
            self.series_tolerance, self.series_order,
            self.evaluation, self.interpolation_tolerance,
        )


@dataclass(frozen=True)
class TTAKineticModel(KineticModel):
    """Analytical TTA evaluator preserving the legacy log-parameter API.

    ``solve(times, log([tau_r, tau_A, k2_effective]))`` returns columns
    ``[A, S]``. The natural parameters remain strictly positive because the
    optimizer coordinates are logarithmic. Use :class:`SensitizedTTAModel`
    when exact zero first-order decay or annihilation must be allowed.

    The normalized annihilator obeys
    ``A' = exp(-t/tau_r)/tau_r - A/tau_A - k2_effective*A**2``.
    Consequently, the historical third parameter is an effective inverse-time
    rate ``k2_effective = physical_k2 * (P/beta)``, where ``beta=1/tau_r``.
    It is not, by itself, the physical bimolecular coefficient in 1/(M*time).

    The initial condition A(0)=0 is independent of the requested time grid.
    ``time_unit`` labels reports and does not perform unit conversion.
    """

    time_unit: str = "s"
    series_tolerance: float = 1e-14
    series_order: int | None = None

    def __post_init__(self):
        _tta_settings(self.time_unit, self.series_tolerance, self.series_order)

    def n_params(self):
        return 3

    def param_names(self):
        return ["τᵣ", "τ₃ₐ", "k₂"]

    def param_units(self):
        return [self.time_unit, self.time_unit, f"{self.time_unit}^-1"]

    def param_descriptions(self):
        return [
            "Sensitizer lifetime: 1/(k_3S + k_TET*[A])",
            "First-order annihilator triplet lifetime: 1/k_3A",
            "Effective normalized annihilation rate: physical_k_2*(P/beta)",
        ]

    def species_names(self):
        return ["[³A*]", "[³S*]"]

    def solve(self, times, beta):
        times, parameters = _tta_input(times, beta)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            rate = np.exp(-parameters[0])
            gamma = np.exp(-parameters[1])
            eta = np.exp(parameters[2] + parameters[0])
        return _tta_basis(
            times, rate, gamma, eta, self.series_tolerance, self.series_order
        )

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
