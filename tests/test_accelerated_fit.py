"""Regression coverage for optional accelerated analytical fitting."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from phoskhemia.fitting.compact_objective import (
    WeightedProjectionObjective,
    fit_compact_global,
)
from phoskhemia.fitting.projections import project_spectra
from phoskhemia.kinetics import SensitizedTTAModel
from phoskhemia.kinetics.analytic import analytic_shape
from phoskhemia.kinetics.accelerated import analytic_shape_fast


class AcceleratedKineticsTests(unittest.TestCase):
    """Compare accelerated samples with the direct analytical series."""

    def test_entire_tta_range(self) -> None:
        rng = np.random.default_rng(20261108)
        times = np.unique(
            np.r_[
                0.0,
                np.geomspace(1e-8, 1000, 60),
                rng.uniform(0.0, 40.0, 100),
            ]
        )

        for nu in (0.0, 1e-5, 0.1, 1.0, 1.000001, 10.0):
            for eta in (0.0, 1e-8, 1e-4, 0.1, 1.0, 10.0):
                with self.subTest(nu=nu, eta=eta):
                    reference = analytic_shape(times, 1.0, nu, eta)
                    accelerated = analytic_shape_fast(times, 1.0, nu, eta)

                    self.assertLess(np.max(np.abs(reference - accelerated)), 5e-10)
                    self.assertEqual(accelerated[0], 0.0)
                    self.assertTrue(np.isfinite(accelerated).all())

    def test_model_matches_reference_for_ns_and_seconds(self) -> None:
        times_ns = np.linspace(1.0, 1e6, 800)
        parameters_ns: NDArray[np.float64] = np.array([0.006, 4e-6, 1e-4])

        model = SensitizedTTAModel(
            time_unit="ns",
            evaluation="accelerated",
        )

        accelerated = model.solve(times_ns, parameters_ns)
        reference = SensitizedTTAModel().solve(
            times_ns * 1e-9,
            parameters_ns * [1e9, 1e9, 1.0],
        )

        np.testing.assert_allclose(accelerated, reference, atol=5e-10, rtol=0)

    def test_bad_acceleration_mode(self) -> None:
        with self.assertRaises(ValueError):
            SensitizedTTAModel(evaluation="invalid")


class CompactObjectiveTests(unittest.TestCase):
    """Verify compact weighted projection against full residual calculations."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20261108)
        self.times = np.linspace(1.0, 1e6, 220)
        self.model = SensitizedTTAModel(
            time_unit="ns",
            evaluation="accelerated",
        )
        self.parameters = np.array([0.006, 4e-6, 1e-4])
        self.basis = self.model.solve(self.times, self.parameters)
        self.noise = np.linspace(0.5e-6, 2e-6, 24)
        amplitudes = rng.uniform(-1e-3, 1e-3, size=(2, self.noise.size))
        self.data = (
            self.basis @ amplitudes
            + rng.normal(0, self.noise, size=(self.times.size, self.noise.size))
        )

    def test_profiled_objective_matches_full_residuals(self) -> None:
        objective = WeightedProjectionObjective(self.data, self.noise)
        projection = project_spectra(self.basis, self.data, self.noise)
        expected = float(np.sum((projection.residuals / self.noise) ** 2))
        actual = objective.chi_squared(self.basis)

        self.assertAlmostEqual(actual / expected, 1.0, places=9)

    def test_fixed_parameters_minimize_objective(self) -> None:
        objective = WeightedProjectionObjective(self.data, self.noise)
        true_value = objective.chi_squared(self.basis)
        wrong_basis = self.model.solve(self.times, [0.003, 1e-5, 0.1])

        self.assertLess(true_value, objective.chi_squared(wrong_basis))

    def test_compact_fit_produces_finite_amplitudes(self) -> None:
        result = fit_compact_global(
            self.times,
            self.data,
            self.noise,
            self.model,
            initial_parameters=[0.005, 5e-6, 1.5e-4],
            parameter_scales=[0.006, 4e-6, 1e-4],
            bounds=([0.001, 0.0, 0.0], [0.02, 1e-4, 0.01]),
            max_iterations=40,
        )

        self.assertEqual(result.amplitudes.shape, (self.noise.size, 2))
        self.assertTrue(np.isfinite(result.amplitudes).all())
        self.assertTrue(np.isfinite(result.chi_squared))


if __name__ == "__main__":
    unittest.main()
