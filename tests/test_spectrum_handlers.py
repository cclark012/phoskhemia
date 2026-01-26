import numpy as np
import pytest
from scipy.signal import convolve

from phoskhemia.data.spectrum_handlers import MyArray


# -------------------------------------------------
# Fixtures
# -------------------------------------------------
@pytest.fixture
def small_array():
    data = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    return MyArray(data, x=x, y=y)


# -------------------------------------------------
# Basic invariants
# -------------------------------------------------
def test_shape_preserved(small_array):
    out = small_array.smooth(3)
    assert out.shape == small_array.shape


def test_coordinates_preserved(small_array):
    out = small_array.smooth(3)
    assert np.array_equal(out.x, small_array.x)
    assert np.array_equal(out.y, small_array.y)


# -------------------------------------------------
# Axis inference
# -------------------------------------------------
def test_1d_x_from_int(small_array):
    out = small_array.smooth(3)
    assert out.shape == small_array.shape


def test_1d_x_from_tuple(small_array):
    out = small_array.smooth((1, 5))
    assert out.shape == small_array.shape


def test_1d_y_from_tuple(small_array):
    out = small_array.smooth((5, 1))
    assert out.shape == small_array.shape


def test_2d_from_tuple(small_array):
    out = small_array.smooth((3, 3))
    assert out.shape == small_array.shape


# -------------------------------------------------
# SciPy parity tests
# -------------------------------------------------
def test_1d_x_matches_scipy(small_array):
    h = np.ones(5)
    h /= h.sum()

    expected = convolve(
        np.asarray(small_array),
        h[None, :],
        mode="same",
    )

    out = small_array.smooth(5)

    assert np.allclose(np.asarray(out), expected)


def test_1d_y_matches_scipy(small_array):
    v = np.ones(5)
    v /= v.sum()

    expected = convolve(
        np.asarray(small_array),
        v[:, None],
        mode="same",
    )

    out = small_array.smooth((5, 1))

    assert np.allclose(np.asarray(out), expected)


def test_2d_matches_scipy(small_array):
    W = np.ones((3, 3))
    W /= W.sum()

    expected = convolve(
        np.asarray(small_array),
        W,
        mode="same",
    )

    out = small_array.smooth((3, 3))

    assert np.allclose(np.asarray(out), expected)


# -------------------------------------------------
# SciPy keyword passthrough
# -------------------------------------------------
def test_fft_method(small_array):
    out = small_array.smooth((7, 7), method="fft")
    assert out.shape == small_array.shape


# -------------------------------------------------
# Error handling
# -------------------------------------------------
def test_invalid_window_raises(small_array):
    with pytest.raises(ValueError):
        small_array.smooth(window=(1, 2, 3))


def test_negative_window_raises(small_array):
    with pytest.raises(ValueError):
        small_array.smooth(-3)
