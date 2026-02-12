from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from numpy.typing import NDArray

from phoskhemia.data import TransientAbsorption


def load_mat(
        path: str,
        *,
        key: str | None = None,
        meta: Mapping[str, Any] | None = None,
        store_probe_row: bool = True,
    ) -> TransientAbsorption:

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

    m = dict(meta) if meta is not None else {}
    m.update(
        {
            "source_path": path,
            "mat_key": key,
        }
    )
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
