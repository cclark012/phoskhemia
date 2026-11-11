"""Independent analytical checks and regressions for sensitized TTA kinetics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.special import iv, kv

from phoskhemia.kinetics import (
    SensitizedTTAModel,
    TTAKineticModel,
    analytic_shape,
    series_order,
    triplet_population,
)
from phoskhemia.kinetics.base import CompositeKineticModel
from phoskhemia.kinetics.models import ExponentialModel, NexponentialModel
from phoskhemia.kinetics.parameterizations import simplex_weights_from_unconstrained


def bessel_reference(s, nu, eta):
    """Independent exact representation, used only at safe Bessel arguments.

    Nonnegative orders avoid the extra K*K cancellation of the negative-order
    I connection formula. This expression is deliberately not used in the
    production evaluator, because very late times overflow its intermediates.
    """
    z0 = 2 * np.sqrt(eta)
    z = z0 * np.exp(-np.asarray(s) / 2)
    m = abs(nu - 1)
    ck = iv(nu + 1, z0) + (2 * nu / z0) * iv(nu, z0)
    numerator = iv(m, z0) * kv(m, z) - kv(m, z0) * iv(m, z)
    denominator = kv(m, z0) * iv(nu, z) + ck * kv(nu, z)
    return (z / (2 * eta)) * numerator / denominator


def integrated_reference(s, nu, eta):
    """Independent test-only ODE reference, accurate into the late decay.

    Integrating w directly makes an absolute ODE tolerance dominate when w
    is small. Instead write w=exp(-r*s)*v, r=min(nu, 1), and integrate
    v'=exp(-(1-r)*s)-(nu-r)*v-eta*exp(-r*s)*v**2. This equation follows
    directly from the scalar Riccati equation, independently of the matrix
    exponential or Bessel representations in the analytical checks.
    """
    s = np.asarray(s, dtype=float)
    r = min(nu, 1.0)

    def rhs(x, v):
        return np.exp(-(1 - r) * x) - (nu - r) * v - eta * np.exp(-r * x) * v**2

    def jacobian(x, v):
        return np.array([[-(nu - r) - 2 * eta * np.exp(-r * x) * v[0]]])

    # Large nu creates an unnecessary stiff linear transient in v. Radau
    # crosses it efficiently; the nonstiff cases use an independent explicit
    # Runge--Kutta implementation with high-order dense interpolation.
    method = "Radau" if (nu - r) * s[-1] > 500 else "DOP853"
    kwargs = {"jac": jacobian} if method == "Radau" else {}
    result = solve_ivp(
        rhs, (0.0, float(s[-1])), [0.0], method=method, t_eval=s,
        rtol=1e-13, atol=1e-15, **kwargs,
    )
    if not result.success:
        raise AssertionError(f"Independent ODE reference failed: {result.message}")
    return np.exp(-r * s) * result.y[0]


class AnalyticalShapeTests(unittest.TestCase):
    def test_linear_limits_include_zero_decay_and_equal_rates(self):
        beta = 2.0
        times = np.array([0.0, 1e-8, 0.1, 1.0, 10.0])
        for gamma in (0.0, beta, 5.0):
            with self.subTest(gamma=gamma):
                if gamma == beta:
                    expected = beta * times * np.exp(-beta * times)
                else:
                    expected = beta / (beta - gamma) * (
                        np.exp(-gamma * times) - np.exp(-beta * times)
                    )
                actual = analytic_shape(times, beta, gamma, 0.0)
                assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)

    def test_matches_independent_bessel_solution(self):
        times = np.array([0.1, 1.0, 5.0])
        for nu in (0.0, 1e-12, 0.01, 0.5, 1.0, 1 + 1e-11, 2.0, 10.0):
            for eta in (0.4, 10.0):
                with self.subTest(nu=nu, eta=eta):
                    expected = bessel_reference(times, nu, eta)
                    actual = analytic_shape(times, 1.0, nu, eta)
                    assert_allclose(actual, expected, rtol=2e-12, atol=3e-15)

    def test_nearly_repeated_rates_regression(self):
        # Independent high-precision analytical values calculated during
        # derivation. The interleaved triangular matrix ordering gave an
        # approximately 9.4e-6 error in the middle case.
        cases = [
            (0.0, 0.2123470451639737),
            (1e-12, 0.21234704516392477),
            (1.0, 0.16860144529246412),
        ]
        for nu, expected in cases:
            with self.subTest(nu=nu):
                actual = analytic_shape(1.0, 1.0, nu, 10.0)
                assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)

    def test_small_time_has_correct_source_slope(self):
        s = 1e-10
        for nu in (0.0, 1.0, 100.0):
            for eta in (1e-12, 10.0):
                with self.subTest(nu=nu, eta=eta):
                    expected = s - (1 + nu) * s**2 / 2
                    assert_allclose(
                        analytic_shape(s, 1.0, nu, eta),
                        expected,
                        rtol=2e-13,
                        atol=0.0,
                    )

    def test_long_time_experimental_case_has_no_cutoff(self):
        P, beta, gamma = 3.0, 6e6, 1e3
        reference_time = 1e-5
        times = np.array([2e-4, 1e-3, 1e-2])
        for q in (1e-3, 1e6, 1e9):
            with self.subTest(q=q):
                eta = q * P / beta**2
                w0 = bessel_reference(beta * reference_time, gamma / beta, eta)
                dt = times - reference_time
                decay = np.exp(-gamma * dt)
                # Exact mixed-order decay after reference_time. The omitted
                # normalized source has integral exp(-60) < 9e-27, providing
                # an independent late-time reference at double precision.
                expected_w = w0 * decay / (
                    1 + (eta * beta * w0 / gamma) * (-np.expm1(-gamma * dt))
                )
                actual = triplet_population(times, P, beta, gamma, q)
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual > 0).all())
                assert_allclose(actual, (P / beta) * expected_w, rtol=5e-11, atol=0)

    def test_stress_domain_remains_finite_and_bounded(self):
        s = np.array([0.0, 1e-9, 0.01, 1.0, 100.0, 1e4, 1e6])
        for nu in (0.0, 1e-12, 1 / 6000, 1.0, 2.0, 10.0, 100.0):
            for eta in (0.0, 1e-16, 8.333333333333333e-5, 10.0):
                with self.subTest(nu=nu, eta=eta):
                    result = analytic_shape(s, 1.0, nu, eta)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertTrue((result >= 0).all())
                    self.assertTrue((result <= 1 + 5e-10).all())
                    self.assertEqual(result[0], 0.0)

    def test_against_independent_ode_across_experimental_domain(self):
        s = np.r_[0.0, np.geomspace(1e-9, 1e6, 121)]
        for nu in (0.0, 1e-12, 1 / 6000, 1 - 1e-10, 1.0, 1 + 1e-10, 2.0, 100.0):
            for eta in (0.0, 1e-16, 1e-4, 10.0):
                with self.subTest(nu=nu, eta=eta):
                    expected = integrated_reference(s, nu, eta)
                    actual = analytic_shape(s, 1.0, nu, eta)
                    # This is a floating-point regression threshold, not the
                    # much tighter analytical series truncation tolerance.
                    # At N*=1 micromolar it is 5e-17 M absolute error.
                    assert_allclose(actual, expected, rtol=0, atol=5e-11)

    def test_physical_experimental_parameters_against_independent_ode(self):
        times = np.geomspace(1e-9, 1e-2, 101)
        P, beta, gamma = 3.0, 6e6, 1e3
        for q in (1e-3, 1e3, 1e6, 1e9, 1e10):
            with self.subTest(q=q):
                expected = (P / beta) * integrated_reference(
                    beta * times, gamma / beta, q * P / beta**2
                )
                actual = triplet_population(times, P, beta, gamma, q)
                assert_allclose(actual, expected, rtol=0, atol=1e-17)

    def test_arbitrary_array_shapes_orders_and_duplicates(self):
        times = np.array([[1e-3, 0.0], [1e-9, 1e-3]])
        actual = analytic_shape(times, 6e6, 1e3, 1e-4)
        self.assertEqual(actual.shape, times.shape)
        self.assertEqual(actual[0, 0], actual[1, 1])
        for position, time in np.ndenumerate(times):
            scalar = analytic_shape(float(time), 6e6, 1e3, 1e-4)
            self.assertEqual(scalar.shape, ())
            self.assertEqual(actual[position], scalar)
        self.assertEqual(analytic_shape([], 1.0, 1.0, 1.0).shape, (0,))

    def test_time_and_concentration_unit_invariance(self):
        times_s = np.geomspace(1e-9, 1e-2, 31)
        n_molar = triplet_population(times_s, 3.0, 6e6, 1e3, 1e9)
        n_ns = triplet_population(times_s * 1e9, 3e-9, 6e-3, 1e-6, 1.0)
        n_micromolar = triplet_population(times_s, 3e6, 6e6, 1e3, 1e3)
        assert_allclose(n_ns, n_molar, rtol=3e-12, atol=1e-21)
        assert_allclose(n_micromolar, 1e6 * n_molar, rtol=3e-12, atol=1e-15)

    def test_zero_source_is_exactly_zero(self):
        result = triplet_population([0.0, 1e-9, 1e-2], 0.0, 1e3, 0.0, 1e10)
        assert_array_equal(result, np.zeros(3))

    def test_order_selection_and_fixed_order_validation(self):
        expected = {0.0: 1, 1e-8: 2, 1e-4: 3, 0.01: 5, 1.0: 10, 10.0: 18}
        for eta, order in expected.items():
            self.assertEqual(series_order(eta), order)
        times = np.array([0.0, 0.1, 1.0, 10.0])
        assert_allclose(
            analytic_shape(times, 1.0, 0.1, 0.01),
            analytic_shape(times, 1.0, 0.1, 0.01, order=series_order(10)),
            rtol=2e-12,
            atol=2e-14,
        )
        with self.assertRaises(ValueError):
            analytic_shape(times, 1.0, 1.0, 10.0, order=1)
        with self.assertRaises(ValueError):
            series_order(1e10)

    def test_invalid_inputs_raise_explicitly(self):
        cases = [
            ([-1.0], 1.0, 1.0, 1.0),
            ([np.nan], 1.0, 1.0, 1.0),
            ([np.inf], 1.0, 1.0, 1.0),
            ([1.0 + 1j], 1.0, 1.0, 1.0),
            ([1.0], 0.0, 1.0, 1.0),
            ([1.0], -1.0, 1.0, 1.0),
            ([1.0], 1.0, -1.0, 1.0),
            ([1.0], 1.0, 1.0, -1.0),
            ([1.0], [1.0], 1.0, 1.0),
            ([1.0], 1.0, np.nan, 1.0),
            ([1.0], 1.0, 1.0, np.inf),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                analytic_shape(*args)
        for tol in (0.0, -1.0, 1.0, np.inf, np.nan):
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                analytic_shape([1.0], 1.0, 1.0, 1.0, tol=tol)
        for order in (0, -1, 1.0, True):
            with self.subTest(order=order), self.assertRaises(ValueError):
                analytic_shape([1.0], 1.0, 1.0, 1.0, order=order)
        with self.assertRaises(ValueError):
            triplet_population([1.0], -1.0, 1.0, 1.0, 1.0)


class KineticModelCompatibilityTests(unittest.TestCase):
    def test_single_component_exponential_has_unit_weight(self):
        model = NexponentialModel(1)
        times = np.array([0.0, 0.5, 1.0, 10.0])
        tau = 2.0
        self.assertEqual(model.n_params(), 1)
        assert_allclose(
            model.solve(times, np.log([tau])),
            np.exp(-times / tau)[:, None],
            rtol=2e-15, atol=0,
        )
        for method in ("softmax", "stick"):
            with self.subTest(method=method):
                assert_array_equal(
                    simplex_weights_from_unconstrained([], k=1, method=method),
                    [1.0],
                )
        assert_array_equal(simplex_weights_from_unconstrained([3.0], k=1), [1.0])
        with self.assertRaises(ValueError):
            simplex_weights_from_unconstrained([1.0, 2.0], k=1)
        with self.assertRaises(ValueError):
            simplex_weights_from_unconstrained([1.0], k=1, method="stick")
        with self.assertRaises(ValueError):
            simplex_weights_from_unconstrained([], k=1, method="unknown")
        with self.assertRaises(ValueError):
            simplex_weights_from_unconstrained([], k=0)

    def test_legacy_parameters_match_new_normalization(self):
        times = np.array([0.0, 1e-9, 1e-6, 2e-4, 1e-3])
        beta, gamma, q_effective = 6e6, 1e3, 500.0
        legacy = TTAKineticModel().solve(
            times, np.log([1 / beta, 1 / gamma, q_effective])
        )
        natural = SensitizedTTAModel().solve(times, [beta, gamma, q_effective / beta])
        assert_allclose(legacy, natural, rtol=3e-12, atol=2e-15)
        self.assertEqual(legacy.shape, (times.size, 2))
        assert_array_equal(legacy[0], [0.0, 1.0])
        self.assertEqual(TTAKineticModel().parameterization(), "log")
        self.assertEqual(SensitizedTTAModel().parameterization(), "linear")

    def test_initial_condition_does_not_move_with_first_sample(self):
        times = np.array([1e-9, 1e-6, 1e-3])
        cases = [
            (SensitizedTTAModel(), np.array([6e6, 1e3, 1e-4])),
            (TTAKineticModel(), np.log([1 / 6e6, 1e-3, 600.0])),
        ]
        for model, parameters in cases:
            with self.subTest(model=type(model).__name__):
                expected = model.solve(np.r_[0.0, times], parameters)[1:]
                actual = model.solve(times, parameters)
                assert_array_equal(actual, expected)
                self.assertGreater(actual[0, 0], 0.0)
                assert_array_equal(model.solve(times[:1], parameters), actual[:1])
                assert_array_equal(model.solve(times[::-1], parameters), actual[::-1])

    def test_zero_rates_and_species_metadata(self):
        model = SensitizedTTAModel(time_unit="ns")
        times = np.array([0.0, 1.0, 5.0])
        result = model.solve(times, [0.2, 0.0, 0.0])
        assert_allclose(result[:, 0], -np.expm1(-0.2 * times), atol=1e-15)
        assert_allclose(result[:, 1], np.exp(-0.2 * times), atol=1e-15)
        self.assertEqual(model.species_names(), ["[³A*]", "[³S*]"])
        self.assertEqual(model.param_units(), ["ns^-1", "ns^-1", None])
        self.assertEqual(model.param_names(), ["beta", "gamma", "eta"])
        self.assertEqual(len(model.param_descriptions()), 3)
        self.assertEqual(TTAKineticModel(time_unit="ns").param_units(), ["ns", "ns", "ns^-1"])

    def test_default_and_composite_bounds_use_optimizer_coordinates(self):
        log_model = ExponentialModel()
        lower, upper = log_model.parameter_bounds()
        assert_array_equal(lower, [-np.inf, -np.inf])
        assert_array_equal(upper, [np.inf, np.inf])
        model = SensitizedTTAModel()
        lower, upper = model.parameter_bounds()
        self.assertGreater(lower[0], 0.0)
        assert_array_equal(lower[1:], [0.0, 0.0])
        assert_array_equal(upper, [np.inf, np.inf, np.inf])
        compound = CompositeKineticModel((model, model))
        combined_lower, combined_upper = compound.parameter_bounds()
        assert_array_equal(combined_lower, np.tile(lower, 2))
        assert_array_equal(combined_upper, np.tile(upper, 2))

    def test_model_rejects_wrong_shapes_and_invalid_settings(self):
        model = SensitizedTTAModel()
        for times, parameters in (
            (1.0, [1.0, 1.0, 1.0]),
            ([[1.0]], [1.0, 1.0, 1.0]),
            ([1.0], [1.0, 1.0]),
            ([1.0], [[1.0, 1.0, 1.0]]),
            ([1.0], [np.nan, 1.0, 1.0]),
            ([1.0], [1.0, 1.0, -1.0]),
        ):
            with self.subTest(times=times, parameters=parameters), self.assertRaises(ValueError):
                model.solve(times, parameters)
        for kwargs in (
            {"time_unit": ""},
            {"time_unit": None},
            {"series_tolerance": 0},
            {"series_order": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SensitizedTTAModel(**kwargs)


if __name__ == "__main__":
    unittest.main()
