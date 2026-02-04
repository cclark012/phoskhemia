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
    param_names() -> str | list[str] | tuple[str]:
        Chosen names for the nonlinear variables.
    species_names() -> str | list[str] | tuple[str]:
        Chosen names for the chemical/kinetic species.
    """

    @abstractmethod
    def n_params(self: KineticModel) -> int:
        """Number of nonlinear kinetic parameters."""
        pass

    @abstractmethod
    def solve(
            self: KineticModel, 
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

    @abstractmethod
    def param_names(self: KineticModel) -> str | list[str] | tuple[str, ...]:
        """Names of kinetic parameters."""
        pass

    @abstractmethod
    def species_names(self: KineticModel) -> str | list[str] | tuple[str, ...]:
        """Names of kinetic species / basis functions."""
        pass
