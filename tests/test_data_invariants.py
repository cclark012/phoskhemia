"""Regression tests for the TA coordinate and measurement-noise contract."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from phoskhemia.data import Spectrum1D, TransientAbsorption
from phoskhemia.data.meta import MetaDict


class DataInvariantTests(unittest.TestCase):
    def make_data(self, noise=None):
        if noise is None:
            noise = np.array([1.0, 2.0, 3.0, 4.0])
        return TransientAbsorption(
            np.arange(12.0).reshape(3, 4),
            x=[400.0, 500.0, 600.0, 700.0],
            y=[0.0, 1e-6, 1e-3],
            meta={"noise_t0": noise, "probe_transmittance": [0.1, 0.2, 0.3, 0.4]},
        )

    def test_axes_are_owned_without_freezing_or_mutating_callers(self):
        wavelengths = np.array([400.0, 500.0])
        times = np.array([0.0, 1e-6, 1e-3])
        metadata = MetaDict({"sample": "test"})
        data = TransientAbsorption(np.zeros((3, 2)), x=wavelengths, y=times, meta=metadata)
        self.assertTrue(wavelengths.flags.writeable)
        self.assertTrue(times.flags.writeable)
        self.assertFalse(data.x.flags.writeable)
        self.assertFalse(data.y.flags.writeable)
        wavelengths[0] = 999.0
        times[0] = -1.0
        self.assertEqual(data.x[0], 400.0)
        self.assertEqual(data.y[0], 0.0)
        data.meta["sample"] = "other"
        self.assertEqual(metadata, {"sample": "test"})
        spectrum = Spectrum1D([1.0, 2.0], x=wavelengths, meta=metadata)
        wavelengths[0] = 800.0
        self.assertEqual(spectrum.x[0], 999.0)
        self.assertFalse(spectrum.x.flags.writeable)
        spectrum.meta["sample"] = "spectrum"
        self.assertEqual(metadata["sample"], "test")

    def test_rectangular_slice_subsets_axes_noise_and_probe(self):
        data = self.make_data()
        sliced = data[1:, 1:3]
        self.assertIsInstance(sliced, TransientAbsorption)
        assert_array_equal(sliced, np.asarray(data)[1:, 1:3])
        assert_array_equal(sliced.x, [500.0, 600.0])
        assert_array_equal(sliced.y, [1e-6, 1e-3])
        assert_array_equal(sliced.meta.noise_t0, [2.0, 3.0])
        assert_allclose(sliced.meta["probe_transmittance"], [0.2, 0.3])
        sliced.meta["child"] = True
        sliced.meta.noise_t0[0] = 100.0
        sliced.meta["probe_transmittance"][0] = 100.0
        self.assertNotIn("child", data.meta)
        self.assertEqual(data.meta.noise_t0[1], 2.0)
        self.assertEqual(data.meta["probe_transmittance"][1], 0.2)

    def test_scalar_noise_survives_slicing_and_trim(self):
        data = self.make_data(noise=0.25)
        selected = data.trim("wavelength", (400.0, 600.0))
        self.assertEqual(selected.meta.noise_t0, 0.25)
        assert_array_equal(selected.x, [400.0, 500.0, 600.0])
        assert_allclose(selected.meta["probe_transmittance"], [0.1, 0.2, 0.3])
        self.assertNotIn("trimmed", data.meta)

    def test_advanced_rectangular_selection_preserves_matrix_noise(self):
        noise = np.arange(1.0, 13.0).reshape(3, 4)
        data = self.make_data(noise=noise)
        index = np.ix_([0, 2], [1, 3])
        selected = data[index]
        self.assertIsInstance(selected, TransientAbsorption)
        assert_array_equal(selected, np.asarray(data)[index])
        assert_array_equal(selected.meta.noise_t0, noise[index])
        assert_array_equal(selected.x, [500.0, 700.0])
        assert_array_equal(selected.y, [0.0, 1e-3])
        assert_array_equal(data[..., 1:3].meta.noise_t0, noise[:, 1:3])
        assert_array_equal(data[[True, False, True], :].meta.noise_t0, noise[[0, 2], :])

    def test_scalar_vector_and_paired_indexing_return_numpy_types(self):
        data = self.make_data()
        self.assertTrue(np.isscalar(data[0, 0]))
        for selection in (data[0, :], data[:, 0], data[[0, 1], [1, 2]], data[data > 5]):
            self.assertIs(type(selection), np.ndarray)
            self.assertEqual(selection.ndim, 1)
        self.assertIs(type(data[:, None, :]), np.ndarray)

    def test_shape_changing_numpy_operations_do_not_keep_stale_axes(self):
        data = self.make_data()
        operations = [
            lambda: data.T,
            lambda: np.transpose(data),
            lambda: data.reshape(2, 6),
            lambda: data.mean(axis=0),
            lambda: np.mean(data, axis=0),
            lambda: data.take([0, 2], axis=0),
            lambda: data.diagonal(),
            lambda: data.repeat(2, axis=0),
            lambda: data.compress([True, False, True], axis=0),
            lambda: data.dot(np.ones(4)),
            lambda: data + np.ones((2, 3, 4)),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                self.assertIs(type(operation()), np.ndarray)
        self.assertTrue(np.isscalar(data.sum()))
        with self.assertRaises(ValueError):
            data.resize((2, 6))

    def test_nested_numpy_arguments_do_not_recurse(self):
        data = self.make_data()
        result = np.concatenate([data, data], axis=0)
        self.assertIs(type(result), np.ndarray)
        assert_array_equal(result, np.concatenate([np.asarray(data)] * 2, axis=0))
        stack = np.stack([data, data], axis=0)
        self.assertIs(type(stack), np.ndarray)
        self.assertEqual(stack.shape, (2, 3, 4))

    def test_elementwise_ufunc_keeps_compatible_axes(self):
        data = self.make_data()
        result = np.add(data, data)
        self.assertIsInstance(result, TransientAbsorption)
        assert_array_equal(result, 2 * np.asarray(data))
        assert_array_equal(result.x, data.x)
        assert_array_equal(result.y, data.y)
        incompatible = TransientAbsorption(np.asarray(data), x=data.x + 1, y=data.y)
        self.assertIs(type(np.add(data, incompatible)), np.ndarray)

    def test_trim_includes_endpoints_and_checks_real_time_bounds(self):
        data = self.make_data()
        result = data.trim("time", (0.0, 1e-3))
        self.assertEqual(result.shape, data.shape)
        assert_array_equal(result.y, data.y)
        with self.assertRaises(ValueError):
            data.trim("time", (0.0, 1.0))
        descending = data[::-1, ::-1]
        result = descending.trim("x", (400.0, 600.0))
        assert_array_equal(result.x, [600.0, 500.0, 400.0])
        assert_array_equal(result.meta.noise_t0, [3.0, 2.0, 1.0])

    def test_identical_irregular_grids_average_with_independent_noise(self):
        first = self.make_data()
        second = self.make_data(noise=2.0)
        result = first.combine(second)
        assert_array_equal(result, first)
        assert_array_equal(result.y, first.y)
        assert_allclose(result.meta.noise_t0, np.hypot([1, 2, 3, 4], 2) / 2)
        self.assertEqual(result.meta["combine_noise_assumption"], "independent input measurements")
        # Historical + semantics remain averaging; the same observation has no
        # independent-noise gain when combined with itself.
        repeated = first + first
        assert_array_equal(repeated, first)
        assert_array_equal(repeated.meta.noise_t0, first.meta.noise_t0)
        assert_array_equal(np.add(first, first), 2 * np.asarray(first))

    def test_combine_zero_overlap_and_time_unit_invariance(self):
        results = []
        for scale in (1.0, 1e9):
            first = TransientAbsorption([[1, 2], [3, 4]], x=[400, 500],
                y=np.array([-1e-6, 0]) * scale, meta={"noise_t0": [1, 2]})
            second = TransientAbsorption([[5, 6], [7, 8]], x=first.x,
                y=np.array([0, 1e-6]) * scale, meta={"noise_t0": [1, 2]})
            merged = first.combine(second)
            assert_allclose(merged.y / scale, [-1e-6, 0, 1e-6])
            assert_array_equal(merged, [[1, 2], [4, 5], [7, 8]])
            self.assertNotIn("noise_t0", merged.meta)
            self.assertIn("noise_t0", merged.meta["combine_metadata_dropped"])
            results.append(np.asarray(merged))
        assert_array_equal(*results)

    def test_combine_distinct_small_times_are_not_rounded_into_equality(self):
        for scale in (1.0, 1e9):
            first = TransientAbsorption([[1, 2], [3, 4]], x=[400, 500], y=np.array([0, 1e-6]) * scale)
            second = TransientAbsorption([[5, 6], [7, 8]], x=[400, 500], y=np.array([0, 2e-6]) * scale)
            with self.assertRaisesRegex(ValueError, "different axis spacings"):
                first.combine(second)

    def test_combined_wavelength_axes_do_not_keep_stale_probe_metadata(self):
        first = TransientAbsorption([[1, 2], [3, 4]], x=[400, 500], y=[0, 1],
            meta={"noise_t0": [1, 2], "probe_transmittance": [0.1, 0.2]})
        second = TransientAbsorption([[5, 6], [7, 8]], x=[600, 700], y=[0, 1],
            meta={"noise_t0": [2, 3], "probe_transmittance": [0.3, 0.4]})
        merged = first.combine(second)
        self.assertEqual(merged.shape, (2, 4))
        self.assertNotIn("noise_t0", merged.meta)
        self.assertNotIn("probe_transmittance", merged.meta)
        self.assertIn("probe_transmittance", merged.meta["combine_metadata_dropped"])
        self.assertEqual(first.meta["probe_transmittance"], [0.1, 0.2])

    def test_downsample_means_use_actual_bin_counts(self):
        data = TransientAbsorption(np.arange(10.0).reshape(5, 2), x=[400, 500],
            y=np.arange(5.0), meta={"noise_t0": [1.0, 2.0]})
        result = data.downsample_time(method="linear", stride=2, aggregate="mean")
        assert_array_equal(result.meta["downsample_bin_counts"], [1, 2, 2])
        assert_array_equal(result, [[0, 1], [3, 4], [7, 8]])
        self.assertNotIn("noise_t0", result.meta)
        self.assertIn("time-dependent noise", result.meta["downsample_noise_note"])
        equal_bins = data.downsample_time(method="linear", stride=1, aggregate="mean")
        assert_array_equal(equal_bins.meta.noise_t0, [1.0, 2.0])
        assert_array_equal(data.meta.noise_t0, [1.0, 2.0])

    def test_log_aggregation_with_noise_does_not_reference_unbound_scaling(self):
        data = self.make_data(noise=0.1)
        result = data.downsample_time(method="log", n_log=3, aggregate="mean")
        self.assertIsInstance(result, TransientAbsorption)
        self.assertEqual(sum(result.meta["downsample_bin_counts"]), data.shape[0])
        median = data.downsample_time(method="linear", stride=1, aggregate="median")
        self.assertNotIn("noise_t0", median.meta)
        self.assertIn("nonlinear", median.meta["downsample_noise_note"])

    def test_time_selection_subsets_observation_noise(self):
        noise = np.arange(1.0, 13.0).reshape(3, 4)
        data = self.make_data(noise=noise)
        result = data.downsample_time(method="linear", stride=2)
        assert_array_equal(result.meta.noise_t0, noise[[0, 2], :])
        assert_array_equal(result.y, data.y[[0, 2]])

    def test_default_log_indices_do_not_depend_on_time_units(self):
        from phoskhemia.preprocessing.downsampling import time_indices_log
        seconds = np.linspace(0.0, 1e-3, 1001)
        assert_array_equal(
            time_indices_log(seconds, n_log=20),
            time_indices_log(seconds * 1e9, n_log=20),
        )
        # Explicit eps is still a physical-time offset and converts with time.
        assert_array_equal(
            time_indices_log(seconds, n_log=20, eps=1e-12),
            time_indices_log(seconds * 1e9, n_log=20, eps=1e-3),
        )

    def test_direct_binned_downsampling_clears_unequal_bin_noise(self):
        from phoskhemia.preprocessing.downsampling import downsample_time_binned
        data = TransientAbsorption(np.arange(10.0).reshape(5, 2), x=[400, 500],
            y=np.arange(5.0), meta={"noise_t0": [1.0, 2.0]})
        direct = downsample_time_binned(data, np.array([0, 2, 4]))
        assert_array_equal(direct.meta["downsample_bin_counts"], [1, 2, 2])
        self.assertNotIn("noise_t0", direct.meta)
        self.assertIn("time-dependent noise", direct.meta["downsample_noise_note"])

    def test_direct_and_wrapper_mean_noise_are_scaled_once(self):
        from phoskhemia.preprocessing.downsampling import downsample_time_binned
        data = TransientAbsorption(np.arange(12.0).reshape(6, 2), x=[400, 500],
            y=np.arange(6.0), meta={"noise_t0": [3.0, 6.0]})
        direct = downsample_time_binned(data, np.array([0, 5]))
        wrapped = data.downsample_time(method="linear", stride=5, aggregate="mean")
        expected = np.array([3.0, 6.0]) / np.sqrt(3)
        assert_array_equal(direct.meta["downsample_bin_counts"], [3, 3])
        assert_allclose(direct.meta.noise_t0, expected)
        assert_allclose(wrapped.meta.noise_t0, expected)
        assert_array_equal(direct, wrapped)
        assert_array_equal(data.meta.noise_t0, [3.0, 6.0])

    def test_direct_downsampling_subsets_observation_noise(self):
        from phoskhemia.preprocessing.downsampling import downsample_time
        noise = np.arange(1.0, 13.0).reshape(3, 4)
        data = self.make_data(noise=noise)
        result = downsample_time(data, np.array([0, 2]))
        assert_array_equal(result.meta.noise_t0, noise[[0, 2], :])
        assert_array_equal(result.y, data.y[[0, 2]])

    def test_partial_overlap_rejects_irregular_and_contained_grids(self):
        first = TransientAbsorption(np.zeros((3, 2)), x=[400, 500], y=[0, 1, 3])
        second = TransientAbsorption(np.ones((3, 2)), x=[400, 500], y=[3, 4, 6])
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            first.combine(second)
        interleaved = TransientAbsorption(np.ones((3, 2)), x=[400, 500], y=[2, 4, 6])
        with self.assertRaisesRegex(ValueError, "without exact shared coordinates"):
            first.combine(interleaved)
        outer = TransientAbsorption(np.zeros((4, 2)), x=[400, 500], y=[0, 1, 2, 3])
        inner = TransientAbsorption(np.ones((2, 2)), x=[400, 500], y=[1, 2])
        with self.assertRaisesRegex(ValueError, "Contained"):
            outer.combine(inner)

    def test_supported_partial_overlap_preserves_measured_coordinates(self):
        first = TransientAbsorption(np.ones((3, 2)), x=[400, 500], y=[0, 0.1, 0.2])
        second = TransientAbsorption(np.full((3, 2), 3.0), x=[400, 500], y=[0.2, 0.3, 0.4])
        result = first.combine(second)
        assert_array_equal(result.y, np.r_[first.y, second.y[1:]])
        assert_array_equal(result, [[1, 1], [1, 1], [2, 2], [3, 3], [3, 3]])

    def test_missing_lazy_export_is_available(self):
        from phoskhemia.data import load_absorption_collection
        self.assertTrue(callable(load_absorption_collection))


if __name__ == "__main__":
    unittest.main()
