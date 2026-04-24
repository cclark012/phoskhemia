from __future__ import annotations
from time import perf_counter_ns
import numpy as np
from numpy.typing import NDArray

def estimate_csv_bytes(n_rows: int, n_cols: int, *, chars_per_value: int = 20, delimiter: str = ",") -> int:
    """
    Rough CSV size estimate in bytes.

    - chars_per_value ~ 20 is a conservative default for "{:.17g}".
    - Includes delimiter separators and newline per row.
    """
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    if n_rows <= 0 or n_cols <= 0:
        return 0
    # values + delimiters (n_cols-1) per row + newline
    per_row = n_cols * chars_per_value + max(0, n_cols - 1) * len(delimiter) + 1
    # header row similar scale
    header = per_row
    return header + n_rows * per_row

def format_bytes(n: int) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"

def format_value(val: float, timing: bool = False) -> tuple[str, str]:
    start: int = perf_counter_ns()
    # SI metric prefixes
    prefixes: NDArray[np.str_] = np.asarray([
        "q", "r", "y", 
        "z", "a", "f", 
        "p", "n", "μ", 
        "m", " ", "k", 
        "M", "G", "T", 
        "P", "E", "Z", 
        "Y", "R", "Q"
    ])

    exponent: NDArray[np.floating] | float = np.log10(val) // 3
    idx: NDArray[np.int64] | int = (exponent + 10).astype(np.int64)
    # Values below 10⁻³⁰ or above 10³⁰ don't have standard prefixes
    # Could use combinations of prefixes, but not doing that for now

    if np.size(val) > 1:
        mask: NDArray[np.bool_] = np.logical_or(idx < 0, idx > 20)
        idx[mask] = 10
    elif idx < 0 or idx > 20:
        idx = 10

    # Normalize all values to [1, 1000)
    norm: NDArray[np.str_] = np.round(val / (10 ** (exponent * 3)), 2).astype(dtype=np.str_)
    # norm: NDArray[np.str_] = np.round(val / (10 ** (exponent * 3)), 2)
    # norm = np.asarray([f"{n :>6.2f}" for n in norm], dtype=np.str_)
    # Associated prefix for value
    prefix: NDArray[np.str_] | str = prefixes[idx]

    end: int = perf_counter_ns()
    print(f"{(end - start) * 1e-3 :.2f} μs") if timing else None

    return norm, prefix
