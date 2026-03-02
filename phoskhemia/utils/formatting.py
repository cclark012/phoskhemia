from __future__ import annotations

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
