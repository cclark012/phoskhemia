from __future__ import annotations

import os
import csv
from typing import Any, Mapping
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from phoskhemia.data.spectrum2d import TransientAbsorption
from phoskhemia.data.meta import MetaDict, meta_copy_update
from phoskhemia.data.spectrum1d import AbsorptionSpectrum
from phoskhemia.data.spectrum_collections import SpectrumEntry, AbsorptionCollection


def _select_file_dialog(
        *,
        title: str = "Select MATLAB .mat file",
        filetypes: tuple[tuple[str, str], ...] = (("MATLAB files", "*.mat"), ("All files", "*.*")),
    ) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("GUI file dialog unavailable (tkinter not installed / no display).") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise RuntimeError("No file selected.")
    return path


def load_mat(
        path: str | os.PathLike[str] | None = None,
        *,
        key: str | None = "data",
        gui: bool = False,
        meta: Mapping[str, Any] | None = None,
        store_probe_row: bool = True,
    ) -> TransientAbsorption:

    if path is None:
        if not gui:
            raise ValueError("path is required unless gui=True")
        path = _select_file_dialog()

    path = str(Path(path))
    if not os.path.exists(path):
        if gui:
            path = _select_file_dialog()
        else:
            raise FileNotFoundError(path)

    d: dict[str, Any] = _read_mat(path)

    if key is None:
        key: str = _infer_single_matrix_key(d)

    M: NDArray[np.floating] = np.asarray(d[key], dtype=float)
    if M.ndim != 2 or M.shape[0] < 3 or M.shape[1] < 2:
        raise ValueError(f"Expected 2D matrix with >=3 rows and >=2 cols; got {M.shape} for key={key!r}")

    wl: NDArray[np.floating] = np.asarray(M[0, 1:], dtype=float).reshape(-1)
    probe: NDArray[np.floating] = np.asarray(M[1, 1:], dtype=float).reshape(-1)
    times: NDArray[np.floating] = np.asarray(M[2:, 0], dtype=float).reshape(-1)
    data: NDArray[np.floating] = np.asarray(M[2:, 1:], dtype=float)

    if data.shape != (times.size, wl.size):
        raise ValueError(f"Parsed shape mismatch: data={data.shape}, times={times.size}, wl={wl.size}")

    m: MetaDict = meta_copy_update(meta, {"source_path": path, "mat_key": key,})
    if store_probe_row:
        m["probe_transmittance"] = probe

    return TransientAbsorption(data, x=wl, y=times, meta=m)


def _infer_single_matrix_key(d: Mapping[str, Any]) -> str:
    # prefer the only numeric 2D candidate if unambiguous
    candidates: list[str] = []
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim == 2 and a.size > 0:
            candidates.append(k)
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Provide 'key'. Matrix candidates: {candidates}")


def _read_mat(path: str) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
        out = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in out.items() if not k.startswith('__')}
    except (NotImplementedError, ValueError):
        return _read_mat_v73(path)


def _read_mat_v73(path: str) -> dict[str, Any]:
    import h5py
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def as_ta(
        arr: NDArray[np.floating],
        *,
        x: NDArray[np.floating] | None = None,
        y: NDArray[np.floating] | None = None,
        meta: dict[str, Any] | None = None,
        freeze_axes: bool = True,
        dtype: type = float,
    ) -> TransientAbsorption:
    return TransientAbsorption(arr, x=x, y=y, meta=meta, freeze_axes=freeze_axes, dtype=dtype)


def _clean_cell(x: str | None) -> str:
    return "" if x is None else str(x).strip()


def _is_float_text(x: str) -> bool:
    x = _clean_cell(x)
    if x == "":
        return False
    try:
        float(x)
        return True
    except Exception:
        return False


def _to_float(x: str) -> float:
    return float(_clean_cell(x))


def _looks_like_wavelength_header(x: str) -> bool:
    s = _clean_cell(x).casefold()
    return ("wavelength" in s) or ("nm" in s)


def _looks_like_measurement_header(x: str) -> bool:
    s = _clean_cell(x)
    return s in {"Abs", "Absorptivity", "%R", "%T"}


def _guess_background(name: str, unit: str | None, idx: int) -> bool:
    n = name.casefold()
    if "background" in n or "baseline" in n or "blank" in n:
        return True
    if idx == 0 and unit == "%T":
        return True
    return False


def _parse_footer_rows(rows: list[list[str]]) -> dict[str, Any]:
    """
    Conservative footer parser:
    - "Key,Value"
    - "Key: Value"
    - preserves unparsed rows
    """

    out: dict[str, Any] = {}
    unparsed: list[list[str]] = []

    for row in rows:
        cells = [_clean_cell(c) for c in row]
        cells = [c for c in cells if c != ""]
        if not cells:
            continue

        if len(cells) == 2:
            k, v = cells
            if k:
                out[k] = v
                continue

        if len(cells) == 1 and ":" in cells[0]:
            k, v = cells[0].split(":", 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
                continue

        unparsed.append(row)

    if unparsed:
        out["_unparsed_rows"] = unparsed

    return out


def load_absorption_collection(
        path: str | Path,
        *,
        meta: Mapping[str, Any] | None = None,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> AbsorptionCollection:
    """
    Load a multi-series absorption CSV.

    Expected shape
    --------------
    Row 0:
        user labels in alternating columns, often with blank columns between pairs

    Row 1:
        repeated pairs of:
            Wavelength (nm), Abs/%T/%R/Absorptivity

    Rows 2+:
        numeric wavelength/value pairs, with each pair allowed to end independently

    Footer:
        optional acquisition metadata after numeric data stop
    """

    path = Path(path)

    with path.open("r", encoding=encoding, errors=errors, newline="") as f:
        reader = csv.reader(f)
        rows: list[list[str]] = [list(r) for r in reader]

    if len(rows) < 2:
        raise ValueError(f"{path} does not contain the required header rows.")

    row0 = rows[0]
    row1 = rows[1]

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    row0 = rows[0]
    row1 = rows[1]

    pair_cols: list[tuple[int, int]] = []
    col = 0
    while col + 1 < max_cols:
        h0 = _clean_cell(row1[col])
        h1 = _clean_cell(row1[col + 1])

        if _looks_like_wavelength_header(h0) and _looks_like_measurement_header(h1):
            pair_cols.append((col, col + 1))
            col += 2
        else:
            col += 1

    if not pair_cols:
        raise ValueError(f"No wavelength/value column pairs found in {path}.")

    entries: list[SpectrumEntry] = []
    footer_start_candidates: list[int] = []

    for pair_idx, (c_wl, c_val) in enumerate(pair_cols):
        name = _clean_cell(row0[c_wl]) or _clean_cell(row0[c_val]) or f"series_{pair_idx:03d}"
        unit = _clean_cell(row1[c_val]) or None

        wl_vals: list[float] = []
        y_vals: list[float] = []

        started = False
        stop_row = len(rows)

        for r_idx in range(2, len(rows)):
            wl_txt = _clean_cell(rows[r_idx][c_wl])
            y_txt = _clean_cell(rows[r_idx][c_val])

            wl_ok = _is_float_text(wl_txt)
            y_ok = _is_float_text(y_txt)

            if wl_ok and y_ok:
                wl_vals.append(_to_float(wl_txt))
                y_vals.append(_to_float(y_txt))
                started = True
                continue

            if started:
                stop_row = r_idx
                break

            # ignore leading blanks before data start for this pair
            if not wl_ok and not y_ok:
                continue

        if len(wl_vals) == 0:
            continue

        footer_start_candidates.append(stop_row)

        wl: NDArray[np.floating] = np.asarray(wl_vals, dtype=float)
        y: NDArray[np.floating] = np.asarray(y_vals, dtype=float)

        spec_meta = MetaDict.coerce(
            {
                "source_path": str(path),
                "spectrum_kind": "absorption",
                "series_name": name,
                "unit": unit,
                "pair_index": int(pair_idx),
                "source_columns": {"wavelength": int(c_wl), "value": int(c_val)},
            }
        )

        spec = AbsorptionSpectrum.from_arrays(
            x=wl,
            y=y,
            meta=spec_meta,
            freeze_axis=True,
            dtype=float,
        )

        entries.append(
            SpectrumEntry(
                name=name,
                spectrum=spec,
                kind="absorption",
                unit=unit,
                is_background=_guess_background(name, unit, pair_idx),
                meta=MetaDict.coerce(
                    {
                        "pair_index": int(pair_idx),
                        "source_columns": {"wavelength": int(c_wl), "value": int(c_val)},
                    }
                ),
            )
        )

    if not entries:
        raise ValueError(f"No numeric absorption series could be parsed from {path}.")

    footer_start = min(footer_start_candidates) if footer_start_candidates else len(rows)
    footer_rows = rows[footer_start:] if footer_start < len(rows) else []

    coll_meta = MetaDict.coerce(meta or {})
    coll_meta.update(
        {
            "source_path": str(path),
            "collection_kind": "absorption",
            "series_count": int(len(entries)),
            "units_present": sorted({e.unit for e in entries if e.unit is not None}),
            "pair_columns": pair_cols,
            "header_row_names": row0,
            "header_row_units": row1,
            "footer_rows": footer_rows,
            "footer_parsed": _parse_footer_rows(footer_rows),
        }
    )

    return AbsorptionCollection(entries=entries, meta=coll_meta)
