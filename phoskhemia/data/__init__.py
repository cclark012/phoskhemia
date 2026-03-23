from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    "TransientAbsorption",
    "Spectrum1D",
    "FluorescenceSpectrum",
    "AbsorptionSpectrum",
    "SpectrumEntry",
    "SpectrumCollection",
    "FluorescenceCollection",
    "AbsorptionCollection",
    "load_fluorescence_spectrum",
    "load_absorption_collection",
]

if TYPE_CHECKING:
    from phoskhemia.data.spectrum2d import TransientAbsorption as TransientAbsorption

def __getattr__(name: str):
    if name == "TransientAbsorption":
        from phoskhemia.data.spectrum2d import TransientAbsorption
        return TransientAbsorption
    if name == "Spectrum1D":
        from .spectrum1d import Spectrum1D
        return Spectrum1D
    if name == "FluorescenceSpectrum":
        from .spectrum1d import FluorescenceSpectrum
        return FluorescenceSpectrum
    if name == "AbsorptionSpectrum":
        from .spectrum1d import AbsorptionSpectrum
        return AbsorptionSpectrum
    if name == "SpectrumEntry":
        from .spectrum_collections import SpectrumEntry
        return SpectrumEntry
    if name == "SpectrumCollection":
        from .spectrum_collections import SpectrumCollection
        return SpectrumCollection
    if name == "FluorescenceCollection":
        from .spectrum_collections import FluorescenceCollection
        return FluorescenceCollection
    if name == "AbsorptionCollection":
        from .spectrum_collections import AbsorptionCollection
        return AbsorptionCollection
    if name == "load_fluorescence_spectrum":
        from .spectrum1d import load_fluorescence_spectrum
        return load_fluorescence_spectrum
    # if name == "load_absorption_spectrum":
    #     from .spectrum1d import load_absorption_spectrum
    #     return load_absorption_spectrum

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
