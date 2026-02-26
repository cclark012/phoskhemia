from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["KineticModel"]

if TYPE_CHECKING:
    from .base import KineticModel as KineticModel

def __getattr__(name: str):
    if name == "KineticModel":
        from .base import KineticModel
        return KineticModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
