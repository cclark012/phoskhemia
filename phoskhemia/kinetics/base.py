from typing import Literal
from abc import ABC, abstractmethod
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

