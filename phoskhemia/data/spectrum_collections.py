from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

from phoskhemia.data.meta import MetaDict
from phoskhemia.data.spectrum1d import Spectrum1D, FluorescenceSpectrum, AbsorptionSpectrum


@dataclass
class SpectrumEntry:
    name: str
    spectrum: Spectrum1D
    kind: Literal["fluorescence", "absorption", "generic"] = "generic"
    unit: str | None = None
    is_background: bool = False
    meta: MetaDict = field(default_factory=lambda: MetaDict.coerce({}))

    @property
    def x(self):
        return self.spectrum.x

    @property
    def y(self):
        return self.spectrum

    def copy(self) -> "SpectrumEntry":
        return SpectrumEntry(
            name=str(self.name),
            spectrum=type(self.spectrum).from_arrays(
                x=self.spectrum.x.copy(),
                y=self.spectrum.copy(),
                meta=MetaDict.coerce(dict(self.spectrum.meta)),
                freeze_axis=getattr(self.spectrum, "freeze_axis", True),
                dtype=float,
            ),
            kind=self.kind,
            unit=self.unit,
            is_background=bool(self.is_background),
            meta=MetaDict.coerce(dict(self.meta)),
        )


@dataclass
class SpectrumCollection:
    entries: list[SpectrumEntry]
    meta: MetaDict = field(default_factory=lambda: MetaDict.coerce({}))

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SpectrumEntry]:
        return iter(self.entries)

    def __getitem__(self, key: int | slice) -> SpectrumEntry | list[SpectrumEntry]:
        return self.entries[key]

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    @property
    def units(self) -> list[str | None]:
        return [e.unit for e in self.entries]

    def append(self, entry: SpectrumEntry) -> None:
        self.entries.append(entry)

    def select(
        self,
        *,
        name: str | None = None,
        unit: str | None = None,
        kind: str | None = None,
        background: bool | None = None,
        name_contains: str | None = None,
    ) -> list[SpectrumEntry]:
        out: list[SpectrumEntry] = list(self.entries)

        if name is not None:
            out = [e for e in out if e.name == name]

        if unit is not None:
            out = [e for e in out if e.unit == unit]

        if kind is not None:
            out = [e for e in out if e.kind == kind]

        if background is not None:
            out = [e for e in out if e.is_background is background]

        if name_contains is not None:
            q = name_contains.casefold()
            out = [e for e in out if q in e.name.casefold()]

        return out

    def first(
        self,
        *,
        name: str | None = None,
        unit: str | None = None,
        kind: str | None = None,
        background: bool | None = None,
        name_contains: str | None = None,
    ) -> SpectrumEntry | None:
        matches = self.select(
            name=name,
            unit=unit,
            kind=kind,
            background=background,
            name_contains=name_contains,
        )
        return matches[0] if matches else None

    def by_unit(self, unit: str) -> list[SpectrumEntry]:
        return self.select(unit=unit)

    def by_kind(self, kind: str) -> list[SpectrumEntry]:
        return self.select(kind=kind)

    def backgrounds(self) -> list[SpectrumEntry]:
        return self.select(background=True)

    def copy(self) -> "SpectrumCollection":
        return type(self)(
            entries=[e.copy() for e in self.entries],
            meta=MetaDict.coerce(dict(self.meta)),
        )


@dataclass
class FluorescenceCollection(SpectrumCollection):
    entries: list[SpectrumEntry]

    def __post_init__(self) -> None:
        for e in self.entries:
            if not isinstance(e.spectrum, FluorescenceSpectrum):
                raise TypeError("FluorescenceCollection entries must contain FluorescenceSpectrum instances.")


@dataclass
class AbsorptionCollection(SpectrumCollection):
    entries: list[SpectrumEntry]

    def __post_init__(self) -> None:
        for e in self.entries:
            if not isinstance(e.spectrum, AbsorptionSpectrum):
                raise TypeError("AbsorptionCollection entries must contain AbsorptionSpectrum instances.")

    def units_present(self) -> list[str]:
        vals = [u for u in self.units if u is not None]
        return sorted(set(vals))

    def backgrounds_guess(self) -> list[SpectrumEntry]:
        return [e for e in self.entries if e.is_background]
