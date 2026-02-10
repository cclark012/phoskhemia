from __future__ import annotations

from typing import Any, Literal, TypedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phoskhemia.kinetics.base import KineticModel

class FitCache(TypedDict):
    lam: float
    noise: NDArray[np.floating]
    
    kinetic_model: KineticModel
    beta: NDArray[np.floating]
    cov_beta: NDArray[np.floating] | None
    ci_sigma: float
    ci_level: float
    parameterization: str

    traces: NDArray[np.floating]


@dataclass(frozen=True)
class GlobalFitResult:
    # Public, stable
    kinetics: dict[str, float]
    kinetics_ci: dict[str, tuple[float, float]] | None

    amplitudes: NDArray[np.floating]          # (n_wl, n_species)
    amplitude_errors: NDArray[np.floating]    # (n_wl, n_species)
    species: list[str]
    wavelengths: NDArray[np.floating]
    times: NDArray[np.floating]

    diagnostics: dict[str, float]

    # Semi-public
    backend: dict[str, Any]

    # Internal (explicitly documented)
    _cache: FitCache
    
    @property
    def traces(self) -> NDArray[np.floating]:
        return self._cache["traces"]
    
    @property
    def beta(self) -> NDArray[np.floating]:
        return self._cache["beta"]

    @property
    def cov_beta(self) -> NDArray[np.floating] | None:
        return self._cache["cov_beta"]
    
    @property
    def kinetic_model(self) -> KineticModel:
        return self._cache["kinetic_model"]
    
    @property
    def parameterization(self) -> str:
        return self._cache["parameterization"]
    
    @property
    def ci_sigma(self) -> float:
        return self._cache["ci_sigma"]
    
    @property
    def ci_level(self) -> float:
        return self._cache["ci_level"]

    def cov_kinetics(
            self,
            space: Literal["natural", "log"]="natural",
            method: Literal["exact", "delta", "linear"]="exact"
        ):
        """Placeholder for covariances of kinetic parameters."""
        raise NotImplementedError

    def sample_kinetics(self):
        """Placeholder for Monte-Carlo propagation of beta for confidence bands."""
        raise NotImplementedError
    
    def summary(
        self,
        style: Literal["brief", "technical", "journal"]="brief",
        digits: int=3,
        max_params: int | None=None
    ) -> str:
        """Placeholder for summary generation function."""
        raise NotImplementedError

        #1). Header
        #2). Model + Data
        #3). Kinetics Table
        #4). Diagnostics
        #5). Fit Config (for technical and journal)
