import numpy as np
import pytest
from phoskhemia.io.spectrum_handlers import MyArray


@pytest.fixture(scope="session")
def tall_array():
    """
    Representative tall-skinny array:
    ~524k rows, ~154 columns
    524_288
    """
    rng = np.random.default_rng(0)
    data = rng.random((16384, 154), dtype=np.float64)
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    return MyArray(data, x=x, y=y)


def test_bench_1d_x_smoothing(benchmark, tall_array):
    benchmark(
        lambda: tall_array.smooth(window=5, pad="auto")
    )


def test_bench_1d_y_smoothing(benchmark, tall_array):
    benchmark(
        lambda: tall_array.smooth(window=(5, 1), pad="auto")
    )


def test_bench_separable_2d_auto(benchmark, tall_array):
    benchmark(
        lambda: tall_array.smooth(window=(5, 5), pad="auto")
    )

@pytest.mark.skip(reason="Direct 2D convolution is intentionally slow")
def test_bench_direct_2d_auto(benchmark, tall_array):
    """
    Reference only — expect this to be very slow.
    You may want to mark this xfail or skip by default.
    """
    benchmark(
        lambda: tall_array.smooth(window=(5, 5), pad="auto", separable_tol=0.0)
    )

