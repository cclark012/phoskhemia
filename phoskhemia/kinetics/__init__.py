from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    "KineticModel",
    "SensitizedTTAModel",
    "TTAKineticModel",
    "analytic_shape",
    "analytic_shape_fast",
    "triplet_population",
    "series_order",
]

if TYPE_CHECKING:
    from .base import KineticModel as KineticModel
    from .models import SensitizedTTAModel as SensitizedTTAModel
    from .models import TTAKineticModel as TTAKineticModel
    from .analytic import analytic_shape as analytic_shape
    from .accelerated import analytic_shape_fast as analytic_shape_fast
    from .analytic import triplet_population as triplet_population
    from .analytic import series_order as series_order

def __getattr__(name: str):
    if name == "KineticModel":
        from .base import KineticModel
        return KineticModel
    if name in {"SensitizedTTAModel", "TTAKineticModel"}:
        from . import models
        return getattr(models, name)
    if name == "analytic_shape_fast":
        from .accelerated import analytic_shape_fast
        return analytic_shape_fast
    if name in {"analytic_shape", "triplet_population", "series_order"}:
        from . import analytic
        return getattr(analytic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
