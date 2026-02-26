from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["TransientAbsorption", "KineticModel", "fit_global_kinetics"]

if TYPE_CHECKING:
    from phoskhemia.data import TransientAbsorption as TransientAbsorption
    from phoskhemia.kinetics.base import KineticModel as KineticModel
    from phoskhemia.fitting.global_fit import fit_global_kinetics as fit_global_kinetics

def __getattr__(name: str):
    if name == "TransientAbsorption":
        from phoskhemia.data import TransientAbsorption
        return TransientAbsorption
    if name == "KineticModel":
        from phoskhemia.kinetics.base import KineticModel
        return KineticModel
    if name == "fit_global_kinetics":
        from phoskhemia.fitting.global_fit import fit_global_kinetics
        return fit_global_kinetics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
