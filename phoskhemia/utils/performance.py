from time import perf_counter_ns
from collections.abc import Callable
from typing import Any
import numpy as np

def performance_test(
        func: Callable[[Any], Any],
        n_iters: int = 20,
        **kwargs
    ) -> tuple[float, float]:
    times = []
    for i in range(n_iters):
        start = perf_counter_ns()
        func(**kwargs)
        end = perf_counter_ns()
        times.append((end - start))
    
    return float(np.mean(times)), float(np.std(times, ddof=1))
