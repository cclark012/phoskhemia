from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["TransientAbsorption"]

if TYPE_CHECKING:
    from phoskhemia.data.spectrum_handlers import TransientAbsorption as TransientAbsorption

def __getattr__(name: str):
    if name == "TransientAbsorption":
        from phoskhemia.data import TransientAbsorption
        return TransientAbsorption
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
