import numpy as np
from numpy.typing import NDArray

def simplex_weights_from_unconstrained(
        u: NDArray[np.floating],
        *,
        k: int,
        method: str = "softmax",
    ) -> NDArray[np.floating]:
    """
    Convert unconstrained params to simplex weights (sum=1, all positive).

    Parameters
    ----------
    u : array-like
        Unconstrained parameters. Expected length:
          - softmax: k-1 (preferred) or k
          - stick: k-1
    k : int
        Number of weights desired.
    method : {"softmax", "stick"}
        Mapping strategy.

    Returns
    -------
    w : (k,) ndarray
        Simplex weights.
    """

    u = np.asarray(u, dtype=float).reshape(-1)
    if k < 2:
        raise ValueError("k must be >= 2")

    if method == "softmax":
        if u.size == k:
            z = u
        elif u.size == k - 1:
            z = np.concatenate([u, np.array([0.0])])
        else:
            raise ValueError(f"softmax expects len(u) in {{k-1, k}}; got {u.size}, k={k}")
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    if method == "stick":
        if u.size != k - 1:
            raise ValueError(f"stick expects len(u)=k-1; got {u.size}, k={k}")
        v = 1.0 / (1.0 + np.exp(-u))  # sigmoid
        w = np.empty(k, dtype=float)
        rem = 1.0
        for i in range(k - 1):
            w[i] = rem * v[i]
            rem *= (1.0 - v[i])
        w[k - 1] = rem
        return w

    raise ValueError(f"Unknown method: {method}")
