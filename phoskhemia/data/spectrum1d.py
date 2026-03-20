from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from phoskhemia.data.meta import MetaDict

_SPLIT_RE = re.compile(r"[\t,\s]+")  # tabs, commas, whitespace
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")

def _try_parse_two_floats(line: str, *, allow_extra_cols: bool = False) -> tuple[float, float] | None:
    s = line.strip()
    if not s:
        return None
    parts = [p for p in _SPLIT_RE.split(s) if p]
    if len(parts) < 2:
        return None
    try:
        a = float(parts[0])
        b = float(parts[1])
    except Exception:
        return None
    if (not allow_extra_cols) and len(parts) > 2:
        return None
    return a, b


def _longest_true_run(mask: list[bool]) -> tuple[int, int] | None:
    """
    Return (start, end) for the longest contiguous run of True in mask.
    end is exclusive. Returns None if no True exists.
    """

    best: tuple[int, int] | None = None
    start: int | None = None
    for i, m in enumerate(mask + [False]):  # sentinel False to flush final run
        if m and start is None:
            start = i
        if (not m) and start is not None:
            end = i
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    return best



def _parse_scalar_value(text: str) -> Any:
    """
    Parse simple scalars from metadata:
      - true/false -> bool
      - int -> int
      - float -> float
      - else -> str
    """

    t = text.strip()
    tl = t.lower()
    if tl in {"true", "false"}:
        return tl == "true"

    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except Exception:
            return t

    if _FLOAT_RE.fullmatch(t):
        try:
            return float(t)
        except Exception:
            return t

    return t


def parse_metadata_lines(lines: Iterable[str]) -> dict[str, Any]:
    """
    Parse simple key/value metadata blocks of the forms:
      - Key: Value
      - Key=Value
    Supports section headers like:
      - KY328i:
    which start a nested dict until another section header appears.
    """

    out: dict[str, Any] = {}
    current: dict[str, Any] = out

    for raw in lines:
        s = raw.strip("\n")
        st = s.strip()
        if not st:
            continue

        # tolerate leading comment markers
        if st.startswith(("#", ";")):
            out.setdefault("_comments", []).append(st)
            continue

        # section header: "SectionName:"
        if ":" in st:
            k, v = st.split(":", 1)
            key = k.strip()
            val = v.strip()
            if val == "":
                sec = out.get(key)
                if not isinstance(sec, dict):
                    sec = {}
                    out[key] = sec
                current = sec
                continue
            current[key] = _parse_scalar_value(val)
            continue

        # alt form: "Key=Value"
        if "=" in st:
            k, v = st.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key:
                current[key] = _parse_scalar_value(val)
            continue

        # otherwise preserve as unparsed line
        out.setdefault("_unparsed", []).append(st)

    return out

class Spectrum1D(np.ndarray):
    """
    1D spectrum ndarray subclass with a wavelength axis and metadata.

    Axis convention:
      - self.x is the wavelength axis, shape (n,)
      - the ndarray data is intensity-like, shape (n,)
    """
    x: NDArray[np.floating]
    meta: MetaDict
    freeze_axis: bool

    def __new__(
            cls,
            arr: NDArray[np.floating],
            *,
            x: NDArray[np.floating] | None = None,
            meta: Mapping[str, Any] | None = None,
            freeze_axis: bool = True,
            dtype: type = float,
        ) -> "Spectrum1D":
        obj = np.asarray(arr, dtype=dtype).view(cls)
        if obj.ndim != 1:
            raise ValueError(f"{cls.__name__} expects a 1D array; got shape {obj.shape}.")

        x_arr = np.arange(obj.shape[0], dtype=float) if x is None else np.asarray(x, dtype=float).reshape(-1)
        if x_arr.shape[0] != obj.shape[0]:
            raise ValueError(f"x must have length {obj.shape[0]} (got {x_arr.shape[0]}).")

        if freeze_axis:
            try:
                x_arr.flags.writeable = False
            except Exception:
                pass

        object.__setattr__(obj, "x", x_arr)
        object.__setattr__(obj, "meta", MetaDict.coerce(meta or {}))
        object.__setattr__(obj, "freeze_axis", bool(freeze_axis))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        object.__setattr__(self, "x", getattr(obj, "x", None))
        object.__setattr__(self, "freeze_axis", getattr(obj, "freeze_axis", True))
        # shallow-copy meta mapping to avoid cross-view mutation surprises
        object.__setattr__(self, "meta", MetaDict.coerce(dict(getattr(obj, "meta", {}))))

    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(out, Spectrum1D):
            x_new = np.asarray(self.x)[key]
            x_new = np.asarray(x_new, dtype=float).reshape(-1)
            if getattr(out, "freeze_axis", True):
                try:
                    x_new.flags.writeable = False
                except Exception:
                    pass
            object.__setattr__(out, "x", x_new)
        return out

    @classmethod
    def from_arrays(
            cls,
            *,
            x: NDArray[np.floating],
            y: NDArray[np.floating],
            meta: Mapping[str, Any] | None = None,
            freeze_axis: bool = True,
            dtype: type = float,
        ) -> "Spectrum1D":
        # y is the intensity array; naming mirrors TA conventions
        return cls(y, x=x, meta=meta, freeze_axis=freeze_axis, dtype=dtype)


class FluorescenceSpectrum(Spectrum1D):
    """1D fluorescence spectrum (wavelength vs intensity)."""
    pass


# Placeholder for later:
class AbsorptionSpectrum(Spectrum1D):
    """1D absorption spectrum (wavelength vs absorbance/intensity)."""
    pass

def load_fluorescence_spectrum(
        path: str | Path,
        *,
        meta: Mapping[str, Any] | None = None,
        allow_extra_cols: bool = False,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> FluorescenceSpectrum:
    """
    Load a 2-column fluorescence spectrum from txt/csv/asc.

    Behavior:
      - Finds the longest contiguous block of lines that parse as (wavelength, intensity).
      - Treats all other lines as metadata (top/bottom/both).
      - Parses simple metadata forms: 'Key: Value', 'Key=Value', and section headers 'Section:'.

    Returns FluorescenceSpectrum with:
      - spec.x = wavelength axis
      - np.asarray(spec) = intensity
      - spec.meta includes raw/parsed metadata blocks
    """

    path = Path(path)
    text = path.read_text(encoding=encoding, errors=errors)
    lines = text.splitlines()

    parsed = [_try_parse_two_floats(ln, allow_extra_cols=allow_extra_cols) for ln in lines]
    mask = [p is not None for p in parsed]
    run = _longest_true_run(mask)
    if run is None:
        raise ValueError(f"No 2-column numeric data block found in {path}.")

    i0, i1 = run
    pairs = [p for p in parsed[i0:i1] if p is not None]
    wl = np.asarray([a for a, _ in pairs], dtype=float)
    inten = np.asarray([b for _, b in pairs], dtype=float)

    meta_top = lines[:i0]
    meta_bot = lines[i1:]

    has_top = any(ln.strip() for ln in meta_top)
    has_bot = any(ln.strip() for ln in meta_bot)
    if has_top and has_bot:
        meta_pos = "both"
    elif has_top:
        meta_pos = "top"
    elif has_bot:
        meta_pos = "bottom"
    else:
        meta_pos = "none"

    m = MetaDict.coerce(meta or {})
    m.update(
        {
            "source_path": str(path),
            "spectrum_kind": "fluorescence",
            "data_columns": ["wavelength", "intensity"],
            "metadata_position": meta_pos,
            "metadata_raw_top": [ln for ln in meta_top if ln.strip()],
            "metadata_raw_bottom": [ln for ln in meta_bot if ln.strip()],
            "metadata_parsed_top": parse_metadata_lines(meta_top),
            "metadata_parsed_bottom": parse_metadata_lines(meta_bot),
        }
    )

    return FluorescenceSpectrum.from_arrays(x=wl, y=inten, meta=m, freeze_axis=True, dtype=float)
