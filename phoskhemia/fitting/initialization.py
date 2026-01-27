import numpy as np
from numpy.typing import NDArray

def estimate_noise(arr, tail_fraction=0.2):
    data = np.asarray(arr)
    n_tail = int(data.shape[0] * tail_fraction)
    tail = data[-n_tail:, :]
    sigma = np.std(tail, axis=0, ddof=1)
    sigma[sigma == 0] = np.median(sigma)
    return sigma

def svd_initialize_kinetics(arr, n_components=2):
    D = np.asarray(arr, dtype=float)

    # Estimate noise
    sigma = estimate_noise(arr)
    W = 1.0 / sigma

    # Mean-center and weight
    D0 = D - D.mean(axis=0, keepdims=True)
    Dw = D0 * W[None, :]

    # Weighted SVD
    U, S, Vt = np.linalg.svd(Dw, full_matrices=False)

    # Unweight time components
    C = U[:, :n_components] * S[:n_components]

    t = np.asarray(arr.y, dtype=float)

    # Identify slow vs fast component (same heuristic as before)
    decay_rates = []
    for k in range(n_components):
        y = np.abs(C[:, k])
        mask = y > 0
        if mask.sum() < 5:
            decay_rates.append(np.inf)
        else:
            slope, _ = np.polyfit(t[mask], np.log(y[mask]), 1)
            decay_rates.append(-slope)

    idx_S = np.argmin(decay_rates)
    idx_A = 1 - idx_S

    C_S = C[:, idx_S]
    C_A = C[:, idx_A]

    # Initial kinetic guesses
    tr = -1.0 / np.polyfit(t[C_S > 0], np.log(C_S[C_S > 0]), 1)[0]
    t3a = -1.0 / np.polyfit(t[C_A > 0], np.log(C_A[C_A > 0]), 1)[0]
    kappa2 = 1e-4

    beta0 = np.array([
        np.log(tr),
        np.log(t3a),
        np.log(kappa2),
    ])

    return beta0
