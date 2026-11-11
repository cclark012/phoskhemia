"""End-to-end and independently calculated variable-projection regressions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar

from phoskhemia.data import TransientAbsorption
from phoskhemia.fitting.global_fit import fit_global_kinetics
from phoskhemia.fitting.projections import project_amplitudes, project_spectra
from phoskhemia.fitting.solvers import LeastSquaresSolver, ODRPackSolver, get_solver
from phoskhemia.kinetics import SensitizedTTAModel, TTAKineticModel
from phoskhemia.kinetics.base import KineticModel


class RateModel(KineticModel):
    def n_params(self):
        return 1

    def solve(self, times, beta):
        return np.exp(-beta[0] * times)[:, None]

    def parameterization(self):
        return "linear"

    def parameter_bounds(self):
        return np.array([0.0]), np.array([np.inf])

    def param_names(self):
        return ["rate"]

    def species_names(self):
        return ["population"]


class GlobalFitTests(unittest.TestCase):
    @staticmethod
    def data(values, times, noise=1.0):
        return TransientAbsorption(values, x=np.arange(values.shape[1]) + 450.0,
                                   y=times, meta={"noise_t0": noise})

    def test_recovers_analytical_triplet_parameters_and_signed_spectra(self):
        model = SensitizedTTAModel()
        times = np.r_[0.0, np.geomspace(1e-9, 1e-2, 90)]
        truth = np.array([6e6, 1e3, 1e9 * 3 / (6e6) ** 2])
        spectra = np.array([[1e-3, -2e-4], [-6e-4, 4e-4], [2e-4, -8e-4]])
        data = self.data(model.solve(times, truth) @ spectra.T, times, [1e-6, 2e-6, 5e-6])
        with patch("scipy.integrate.solve_ivp", side_effect=AssertionError("production ODE integration")):
            result = data.fit_global_kinetics(model, truth * [1.1, .9, 1.2], maxit=80, x_scale=truth)
        assert_allclose(result.beta, truth, rtol=3e-6)
        assert_allclose(result.amplitudes, spectra, rtol=3e-6)
        self.assertTrue(result._cache["solver_info"]["success"])
        self.assertEqual(result.diagnostics["n_params"], 3 + 2 * 3)
        self.assertEqual(result.diagnostics["DoF"], data.size - 9)
        self.assertEqual(result._cache["time_error_model"], "fixed")
        self.assertIsNotNone(result.cov_beta)

    def test_fit_is_invariant_under_seconds_to_nanoseconds(self):
        model = SensitizedTTAModel()
        times = np.r_[0.0, np.geomspace(1e-9, 1e-2, 70)]
        truth = np.array([6e6, 1e3, 8.333333333333333e-5])
        spectra = np.array([[1e-3, -4e-4], [-6e-4, 2e-4]])
        values = model.solve(times, truth) @ spectra.T
        ns_factor = np.array([1e-9, 1e-9, 1.0])
        result = self.data(values, times * 1e9, 1e-6).fit_global_kinetics(
            SensitizedTTAModel(time_unit="ns"), truth * ns_factor * [1.1, .9, 1.2],
            x_scale=truth * ns_factor, maxit=80,
        )
        assert_allclose(result.beta / ns_factor, truth, rtol=3e-5)
        assert_allclose(result.amplitudes, spectra, rtol=3e-5)
        self.assertEqual(result._cache["solver_info"]["jacobian_rank"], 3)

    def test_wavelength_noise_weights_the_outer_objective(self):
        times = np.linspace(0, 5, 70)
        values = 1e-3 * np.column_stack([np.exp(-.5 * times), np.exp(-2 * times)])
        sigma = np.array([1e-6, 1e-4])
        data = self.data(values, times, sigma)
        result = data.fit_global_kinetics(RateModel(), [.8], maxit=100, gtol=1e-12)

        def independent_objective(rate):
            c = np.exp(-rate * times)
            a = (c @ values) / (c @ c)
            return np.sum(((values - c[:, None] * a) / sigma) ** 2)

        reference = minimize_scalar(independent_objective, bounds=(.2, 3), method="bounded",
                                    options={"xatol": 1e-12})
        self.assertAlmostEqual(result.beta[0], reference.x, places=6)
        self.assertLess(abs(result.beta[0] - .5), 1e-3)
        unweighted = data.fit_global_kinetics(RateModel(), [.8], use_noise_as_sy=False,
                                             maxit=100, gtol=1e-14)
        self.assertGreater(unweighted.beta[0] - result.beta[0], .08)
        self.assertTrue(np.isnan(unweighted.diagnostics["AIC"]))

    def test_fixed_exact_zero_nonlinearity_and_amplitude_covariance(self):
        model = SensitizedTTAModel()
        times = np.r_[0.0, np.geomspace(1e-8, 1e-2, 50)]
        beta = np.array([6e6, 1e3, 0.0])
        basis = model.solve(times, beta)
        values = basis @ np.array([[1e-3, -3e-4], [5e-4, 2e-4]])
        data = self.data(values, times, [1e-6, 3e-6])
        result = data.fit_global_kinetics(model, beta, fix_beta=[True, True, True], absolute_sigma=True)
        assert_array_equal(result.beta, beta)
        assert_array_equal(result.cov_beta, np.zeros((3, 3)))
        self.assertEqual(result.diagnostics["n_params"], 4)
        independent_covariance = np.linalg.inv(basis.T @ basis)
        assert_allclose(result.amplitude_errors[0], 1e-6 * np.sqrt(np.diag(independent_covariance)), rtol=1e-12)
        assert_allclose(result.amplitude_errors[1], 3e-6 * np.sqrt(np.diag(independent_covariance)), rtol=1e-12)

    def test_legacy_log_coordinates_remain_compatible(self):
        times = np.geomspace(1e-9, 1e-3, 40)
        natural = np.array([1 / 6e6, 1e-3, 500.0])
        model = TTAKineticModel()
        values = model.solve(times, np.log(natural)) @ np.array([[1e-3], [-3e-4]])
        result = self.data(values, times).fit_global_kinetics(
            model, np.log(natural), fix_beta=[True, True, True],
        )
        assert_allclose(list(result.kinetics.values()), natural, rtol=1e-14)
        assert_allclose(result.traces[0], model.solve(times[:1], np.log(natural))[0])
        self.assertGreater(result.traces[0, 0], 0.0)

    def test_partial_fix_preserves_zero_and_subsets_solver_scales(self):
        times = np.r_[0.0, np.geomspace(1e-9, 1e-2, 50)]
        model = SensitizedTTAModel()
        truth = np.array([6e6, 1e3, 0.0])
        values = model.solve(times, truth) @ np.array([[1e-3, -6e-4], [-3e-4, 2e-4]])
        result = self.data(values, times, 1e-6).fit_global_kinetics(
            model, truth * [1.1, .9, 1.0], fix_beta=[False, False, True],
            x_scale=[6e6, 1e3, 1e-4], diff_step=[1e-6, 1e-6, 1e-6],
            fix_x=None, maxit=100,
        )
        assert_allclose(result.beta[:2], truth[:2], rtol=2e-6)
        self.assertEqual(result.beta[2], 0.0)
        self.assertEqual(result.diagnostics["n_params"], 6)
        assert_array_equal(result.cov_beta[2], [0, 0, 0])

    def test_kinetic_covariance_counts_projected_amplitudes(self):
        times = np.linspace(0, 4, 40)
        rng = np.random.default_rng(41)
        values = np.exp(-.7 * times)[:, None] * [1.0, -.6]
        values += .01 * rng.normal(size=values.shape)
        data = self.data(values, times, .01)
        known = data.fit_global_kinetics(RateModel(), [.8], absolute_sigma=True)
        scaled = data.fit_global_kinetics(RateModel(), [.8])
        dof = values.size - 1 - 2
        self.assertEqual(scaled._cache["solver_info"]["dof"], dof)
        assert_allclose(scaled.cov_beta, known.cov_beta * scaled.diagnostics["chi2"] / dof, rtol=1e-9)

    def test_rank_deficient_spectra_and_kinetics_are_reported(self):
        class Duplicate(RateModel):
            def solve(self, times, beta):
                c = super().solve(times, beta)
                return np.column_stack([c, c])

            def species_names(self):
                return ["a", "b"]

        times = np.linspace(0, 4, 30)
        values = np.exp(-times)[:, None]
        result = self.data(values, times).fit_global_kinetics(Duplicate(), [.9])
        self.assertEqual(result.diagnostics["amplitude_rank"], 1)
        self.assertEqual(result.diagnostics["n_amplitude_parameters"], 1)
        self.assertTrue(np.isnan(result.amplitude_errors).all())

        class Unidentified(RateModel):
            def solve(self, times, beta):
                return np.exp(-times)[:, None]

        result = self.data(values, times).fit_global_kinetics(Unidentified(), [.9])
        self.assertIsNone(result.cov_beta)
        self.assertIsNone(result.kinetics_ci)
        self.assertIn("rank deficient", result._cache["solver_info"]["covariance_unavailable_reason"])

    def test_ridge_profiles_the_penalty_in_both_objectives(self):
        times = np.linspace(0, 3, 40)
        values = np.exp(-.7 * times)[:, None]
        lam = 3.0
        data = self.data(values, times)
        result = data.fit_global_kinetics(RateModel(), [1.0], lam=lam, maxit=100,
                                         gtol=1e-12, ftol=1e-13, xtol=1e-13)

        def independent_objective(rate):
            c = np.exp(-rate * times)
            amplitude = float(c @ values[:, 0]) / (c @ c + lam)
            return np.sum((values[:, 0] - c * amplitude) ** 2) + lam * amplitude ** 2

        reference = minimize_scalar(independent_objective, bounds=(.01, 2), method="bounded",
                                    options={"xatol": 1e-12})
        self.assertAlmostEqual(result.beta[0], reference.x, places=5)
        self.assertIsNone(result.cov_beta)
        self.assertTrue(result._cache["regularized_diagnostics_approximate"])
        self.assertEqual(result._cache["solver_info"]["n_observations"], values.size)
        self.assertLess(result.diagnostics["n_amplitude_parameters"], 1)

    def test_invalid_noise_shapes_bounds_and_model_failures_are_explicit(self):
        times = np.linspace(0, 3, 12)
        data = self.data(np.exp(-times)[:, None], times)
        for sigma in (0, -1, np.nan, np.inf, [1, 2], np.ones((12, 1))):
            with self.subTest(noise=sigma), self.assertRaises(ValueError):
                data.fit_global_kinetics(RateModel(), [1.0], noise=sigma)
        with self.assertRaisesRegex(ValueError, "outside"):
            data.fit_global_kinetics(RateModel(), [.1], bounds=([.5], [2]))
        for kwargs in ({"sx": np.ones(12)}, {"scope_ppm": 10}, {"fix_x": False}):
            with self.assertRaises(NotImplementedError):
                data.fit_global_kinetics(RateModel(), [1], **kwargs)
        with self.assertRaisesRegex(ValueError, "loss"):
            data.fit_global_kinetics(RateModel(), [1], loss="soft_l1")
        class Broken(RateModel):
            def solve(self, times, beta):
                raise RuntimeError("actual model error")
        with self.assertRaisesRegex(RuntimeError, "actual model error"):
            data.fit_global_kinetics(Broken(), [1])

    def test_nonconvergence_has_no_kinetic_confidence_interval(self):
        times = np.linspace(0, 3, 12)
        data = self.data(np.exp(-times)[:, None], times)
        with self.assertWarns(RuntimeWarning):
            result = data.fit_global_kinetics(RateModel(), [2.5], maxit=1)
        self.assertFalse(result._cache["solver_info"]["success"])
        self.assertIsNone(result.kinetics_ci)

    def test_optional_odr_backend_uses_fixed_time_ols_contract(self):
        times = np.linspace(0, 2, 10)
        data = self.data(np.exp(-times)[:, None], times, .02)
        def fake_odr_fit(f, x, y, beta0, **kwargs):
            self.assertEqual(kwargs["task"], "OLS")
            self.assertTrue(np.all(kwargs["fix_x"]))
            self.assertEqual(f(x, beta0).shape, y.shape)
            return SimpleNamespace(beta=beta0, cov_beta=np.array([[2.0]]), sd_beta=np.array([1.0]),
                                   res_var=1.0, success=True, stopreason="mocked contract check")
        with patch("phoskhemia.fitting.solvers._import_odrpack", return_value=SimpleNamespace(odr_fit=fake_odr_fit)):
            result = data.fit_global_kinetics(RateModel(), [1.0], solver="odrpack")
        self.assertEqual(result._cache["solver_info"]["task"], "OLS")
        self.assertEqual(result._cache["solver_info"]["dof"], 8)


class ProjectionAndSolverTests(unittest.TestCase):
    def test_unregularized_projection_uses_lstsq_even_for_near_collinearity(self):
        t = np.linspace(-1, 1, 20)
        C = np.column_stack([np.ones(20), np.ones(20) + 1e-7 * t])
        coefficients = np.array([2.0, -1.0])
        estimate, covariance, residual = project_amplitudes(C, C @ coefficients, None, 0, absolute_sigma=True)
        assert_allclose(estimate, coefficients, atol=2e-8)
        self.assertTrue(np.isfinite(covariance).all())
        self.assertLess(np.max(abs(residual)), 1e-14)

    def test_ridge_matches_independent_augmented_least_squares(self):
        C = np.array([[1., 0.], [.7, .3], [.3, .9], [.1, .2]])
        Y = np.array([[1., -2.], [.6, -1.], [.8, .5], [.2, .1]])
        sigmas, lam = np.array([.1, .3]), 2.0
        projection = project_spectra(C, Y, sigmas, lam)
        for j, sigma in enumerate(sigmas):
            design = np.vstack([C / sigma, np.sqrt(lam) * np.eye(2)])
            observation = np.r_[Y[:, j] / sigma, [0, 0]]
            expected = np.linalg.lstsq(design, observation, rcond=None)[0]
            assert_allclose(projection.coefficients[:, j], expected, rtol=1e-13)

    def test_solver_selection_and_fixed_parameter_validation(self):
        self.assertIsInstance(get_solver("auto"), LeastSquaresSolver)
        self.assertIsInstance(get_solver("odrpack"), ODRPackSolver)
        with self.assertRaises(ValueError):
            get_solver("typo")
        with self.assertRaisesRegex(ValueError, "boolean"):
            LeastSquaresSolver().fit(f=lambda x,b: x*b[0], xdata=np.arange(3), ydata=np.arange(3),
                                     beta0=[1.], fix_beta=[1])


if __name__ == "__main__":
    unittest.main()
