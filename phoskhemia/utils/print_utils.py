from __future__ import annotations

from typing import Literal, Sequence
import math
import numpy as np
from numpy.typing import NDArray

SummaryStyle = Literal["brief", "technical", "journal", "verbose"]


def _fmt_float(x: float, digits: int = 3) -> str:
    """Compact numeric formatting: fixed for moderate values, scientific otherwise."""
    VALUE_RANGE: tuple[float, float] = (1e-3, 1e4)
    abs_x: float = abs(float(x))
    if abs_x == 0.0:
        return "0"
    if (abs_x < VALUE_RANGE[0]) or (abs_x >= VALUE_RANGE[1]):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"

def _normalize_str_list(
        x: str | Sequence[str] | None, 
        n: int | None = None
    ) -> list[str | None]:

    if x is None:
        return [None] * (n or 0)

    # Convert to list of length 1 if a string
    if isinstance(x, str):
        out: list[str | None] = [x]
    else:
        out = [str(value) if value is not None else None for value in x]

    # Make sure the list is the specified length, appending or truncating.
    if n is not None:
        if len(out) < n:
            out = out + [None] * (n - len(out))
        elif len(out) > n:
            out = out[:n]
    return out

def _block_header(title: str, width: int) -> str:
    # Construct a header that is centered in a given width.
    return f"{title:-^{width}}"

def _kv_line(key: str, val: str, width: int, key_w: int = 22) -> str:
    # | key.................. = value............................... |
    rhs_w = max(0, width - (key_w + 6))
    v = val if len(val) <= rhs_w else (val[: max(0, rhs_w - 3)] + "...")
    return f"| {key:<{key_w}} = {v:<{rhs_w}} |"


def _safe_float(v) -> float | None:
    """Tries to convert to float, returning None in case of an Exception."""
    try:
        return float(v)
    except Exception:
        return None
