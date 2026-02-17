from __future__ import annotations

from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray

# import your EK/E15 implementation
from phoskhemia.preprocessing.svd_ek import svd_reconstruction  # rename as needed


WeightMode = Literal["none", "column"]
CenterMode = Literal["none", "time"]  # "time" == subtract column means (mean over rows)
MethodMode = Literal["e15", "ek18"]
RotationMode = Literal["include", "exclude", "auto"]


def svd_denoise(
        arr: NDArray[np.floating],
        *,
        method: MethodMode = "ek18",
        value_rotation: RotationMode = "exclude",
        threshold: float = 0.05,
        center: CenterMode = "time",
        noise: NDArray[np.floating] | None = None,      # sigma(lambda) if known
        weight: WeightMode = "none",
        return_details: bool = False,
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], dict[str, Any]]:
    """
    Denoise a 2D array using EK18/E15 SVD reconstruction, with optional centering and column whitening.

    Parameters
    ----------
    arr
        2D array (n_times, n_wavelengths).
    center
        "time": subtract column means before denoising and add back after.
    noise
        Per-column noise sigma(lambda). Required if weight="column".
    weight
        "column": whiten by dividing each column by noise prior to denoising.
    """

    D: NDArray[np.floating] = np.asarray(arr, dtype=float)
    if D.ndim != 2:
        raise ValueError("arr must be 2D")

    if not np.all(np.isfinite(D)):
        raise ValueError("arr contains non-finite values")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")

    n_t: int
    n_w: int
    n_t, n_w = D.shape

    # Centering about column mean 
    if center == "time":
        mean: NDArray[np.floating] = D.mean(axis=0, keepdims=True)
        D0: NDArray[np.floating] = D - mean
    elif center == "none":
        mean: None = None
        D0: NDArray[np.floating] = D
    else:
        raise ValueError("center must be 'none' or 'time'")

    # Whitening / weighting 
    sigma: NDArray[np.floating] = None
    if weight == "column":
        if noise is None:
            raise ValueError("noise must be provided when weight='column'")
        sigma: NDArray[np.floating] = np.asarray(noise, dtype=float).reshape(-1)

        if isinstance(sigma, (float, int)):
            sigma: NDArray[np.floating] = np.ones(n_w, dtype=float) * sigma

        if sigma.size != n_w:
            raise ValueError("noise must have shape (n_wavelengths,)")

        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("noise must be finite and strictly positive")

        Dw: NDArray[np.floating] = D0 / sigma[None, :]

    elif weight == "none":
        Dw: NDArray[np.floating] = D0

    else:
        raise ValueError("weight must be 'none' or 'column'")

    # core denoise
    if return_details:
        Dw_hat, info_core = svd_reconstruction(
            Dw,
            method=method,
            value_rotation=value_rotation,
            threshold=threshold,
            return_details=True,
        )
    else:
        Dw_hat = svd_reconstruction(
            Dw,
            method=method,
            value_rotation=value_rotation,
            threshold=threshold,
            return_details=False,
        )
        info_core: None = None

    # unweight
    if weight == "column":
        D_hat0: NDArray[np.floating] = Dw_hat * sigma[None, :]
    else:
        D_hat0: NDArray[np.floating] = Dw_hat

    # uncenter
    D_hat: NDArray[np.floating] = (D_hat0 + mean) if mean is not None else D_hat0

    if not return_details:
        return D_hat

    # augment info
    info: dict[str, Any] = dict(info_core) if info_core is not None else {}
    info["wrapper"] = {
        "center": center,
        "weight": weight,
        "threshold": float(threshold),
        "shape": (int(n_t), int(n_w)),
        "noise_stats": None if sigma is None else {
            "min": float(np.min(sigma)),
            "median": float(np.median(sigma)),
            "max": float(np.max(sigma)),
        },
    }
    return D_hat, info
