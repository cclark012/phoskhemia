from abc import ABC, abstractmethod

class KineticModel(ABC):
    """
    Base class for global kinetic models.
    """

    @abstractmethod
    def n_params(self):
        """Number of nonlinear kinetic parameters."""
        pass

    @abstractmethod
    def solve(self, times, beta):
        """
        Solve the kinetic model.

        Parameters
        ----------
        times : ndarray, shape (n_times,)
        beta : ndarray, shape (n_params,)

        Returns
        -------
        C : ndarray, shape (n_times, n_species)
            Kinetic basis functions
        """
        pass

    @abstractmethod
    def param_names(self):
        """Names of kinetic parameters."""
        pass

    @abstractmethod
    def species_names(self):
        """Names of kinetic species / basis functions."""
        pass
