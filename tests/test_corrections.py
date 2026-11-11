"""Targeted regression tests for legacy correction helper mechanics."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

from phoskhemia.preprocessing.corrections import estimate_irf, find_time_zero


class CorrectionRegressionTests(unittest.TestCase):
    def test_time_zero_retains_matching_rows_for_both_shift_directions(self):
        times = np.linspace(-100.0, 100.0, 201)
        for center in (5.0, -5.0):
            with self.subTest(center=center):
                rise = expit((times - center) / 2)
                data = np.column_stack((rise, -2 * rise))
                new_times, new_data = find_time_zero(times, data)
                start = int(np.flatnonzero(times == center)[0])
                self.assertEqual(new_times.size, new_data.shape[0])
                self.assertEqual(new_times[0], 0.0)
                assert_array_equal(new_times, times[start:] - times[start])
                assert_array_equal(new_data, data[start:, :])
                assert_allclose(new_data[0], [0.5, -1.0])

    def test_irf_stack_accepts_a_finite_one_dimensional_signal(self):
        signal = np.exp(-np.arange(40.0) / 10)
        # This previously raised TypeError before computing any FFT because
        # hstack received two positional arguments instead of one sequence.
        irf = estimate_irf(signal)
        self.assertEqual(irf.ndim, 1)
        self.assertGreater(irf.size, 0)
        self.assertTrue(np.isfinite(irf).all())

    def test_irf_rejects_unsupported_signal_shapes_and_zero_tail(self):
        with self.assertRaisesRegex(ValueError, "finite 1D signal"):
            estimate_irf(np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, "starts at zero"):
            estimate_irf(np.zeros(8))


if __name__ == "__main__":
    unittest.main()
