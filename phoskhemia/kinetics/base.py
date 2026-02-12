from __future__ import annotations
from typing import Literal, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


class KineticModel(ABC):
    """
    Base class for global kinetic models.
    
    Subclasses must implement the following methods:
    n_params() -> int:
        Returns the number of nonlinear/global variables.
    solve(times: NDArray[np.floating], beta: NDArray[np.floating]) -> NDArray[np.floating]:
        Solves the model and returns a column vector/matrix.
    
    Subclasses can implement the following methods for printing and fit control:
    param_names() -> str | list[str]:
        Chosen names for the nonlinear variables.
    param_units() -> str | list[str | None]:
        Chosen units for the nonlinar variables.
    param_descriptions() -> str | list[str | None]:
        Chosen descriptions for the nonlinear variables.
    species_names() -> str | list[str]:
        Chosen names for the chemical/kinetic species.
    parameterization() -> Literal["log", "linear"]:
        Whether nonlinear variables will be fit in log or linear space. 
        Must return either "log" or "linear".
    """

    @abstractmethod
    def n_params(self: KineticModel) -> int:
        """Number of nonlinear kinetic parameters."""
        pass

    @abstractmethod
    def solve(
            self, 
            times: NDArray[np.floating], 
            beta: NDArray[np.floating]
        ) -> NDArray[np.floating]:
        """
        Solve the kinetic model.

        Parameters
        ----------
        times : NDArray[np.floating], shape (n_times,)
        beta : NDArray[np.floating], shape (n_params,)

        Returns
        -------
        C : NDArray[np.floating], shape (n_times, n_species)
            Kinetic basis functions
        """
        pass

    def param_names(self) -> str | list[str]:
        """Names of kinetic parameters."""
        return [f"p{i}" for i in range(self.n_params())]

    def param_units(self) -> str | list[str | None]:
        """Units for the fit parameters. None uses no units."""
        return [None] * self.n_params()
    
    def param_descriptions(self) -> str | list[str | None]:
        """Description of parameters in order of declaration."""
        return [None] * self.n_params()

    def species_names(self) -> str | list[str]:
        """Names of kinetic species / basis functions."""
        return [f"species{i}" for i in range(self.solve(np.array([0.0]), np.zeros(self.n_params())).shape[1])]
    
    def parameterization(self) -> Literal["log", "linear"]:
        """
        Return the parameterization to be used for this model.

        "log"    : beta = log(p), p > 0
        "linear" : beta = p

        Returns
        -------
        str
            The parameterization to be used.
        """

        return "log"


@dataclass(frozen=True)
class CompositeKineticModel(KineticModel):
    models: tuple[KineticModel, ...]
    prefixes: tuple[str, ...] | None = None  # optional for namespacing

    def n_params(self) -> int:
        return sum(m.n_params() for m in self.models)

    def parameterization(self) -> str:
        # enforce same parameterization for now
        ps: set[str] = {m.parameterization() for m in self.models}
        if len(ps) != 1:
            raise ValueError(f"Composite requires matching parameterizations; got {ps}")
        return next(iter(ps))

    def solve(self, times: NDArray[np.floating], beta: NDArray[np.floating]) -> NDArray[np.floating]:
        beta: NDArray[np.floating] = np.asarray(beta, dtype=float).reshape(-1)
        parts: list[NDArray[np.floating]] = []
        idx: int = 0
        for m in self.models:
            k: int = m.n_params()
            b: NDArray[np.floating] = beta[idx:idx+k]
            idx += k
            T = m.solve(times, b)  # (n_times, n_species_m)
            parts.append(np.asarray(T, dtype=float))
        if idx != beta.size:
            raise ValueError("beta length mismatch for composite model")
        return np.concatenate(parts, axis=1)

    def species_names(self) -> list[str]:
        names: list[str] = []
        for i, m in enumerate(self.models):
            sn: list[str] | str = m.species_names()
            sn = [sn] if isinstance(sn, str) else list(sn)
            if self.prefixes:
                sn = [f"{self.prefixes[i]}:{s}" for s in sn]
            names.extend(sn)
        return names

    def param_names(self) -> list[str]:
        names: list[str] = []
        for i, m in enumerate(self.models):
            pn: list[str] | str = m.param_names()
            pn = [pn] if isinstance(pn, str) else list(pn)
            if self.prefixes:
                pn = [f"{self.prefixes[i]}:{p}" for p in pn]
            names.extend(pn)
        return names
