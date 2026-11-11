"""Regression tests for weighting, result consumers, and reconstruction."""

from dataclasses import replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from phoskhemia.data.spectrum2d import TransientAbsorption
from phoskhemia.fitting.reconstructions import (
    reconstruct_fit,
    reconstruct_residuals,
    reconstruct_species,
    reconstruct_species_sum,
    reconstruct_weighted_residuals,
)
from phoskhemia.fitting.results import (
    GlobalFitResult,
    ReportBlock,
    ReportRow,
    cov_lognormal,
    render_latex_report,
    sd_lognormal,
)
from phoskhemia.fitting.validation import (
    compare_models,
    compute_diagnostics,
    compute_residual_maps,
    cross_validate_wavelengths,
    reduced_chi_square,
)
from phoskhemia.kinetics.base import KineticModel


class TwoExponentialModel(KineticModel):
    def n_params(self):
        return 2

    def solve(self, times, beta):
        return np.exp(-np.asarray(times)[:, None] / np.exp(beta)[None, :])

    def species_names(self):
        return ["fast", "slow"]


def make_result():
    model = TwoExponentialModel()
    times = np.linspace(0, 5, 6)
    wavelengths = np.array([500.0, 550.0, 600.0])
    beta = np.log([0.5, 2.0])
    amplitudes = np.array([[1e-3, -2e-4], [-3e-4, 7e-4], [5e-4, 2e-4]])
    traces = model.solve(times, beta)
    noise = np.array([1e-6, 2e-6, 4e-6])
    prediction = traces @ amplitudes.T
    diagnostics = compute_diagnostics(
        y_obs=prediction, y_fit=prediction, noise=noise, n_params=8
    )
    return GlobalFitResult(
        kinetics={"p0": 0.5, "p1": 2.0},
        kinetics_ci=None,
        amplitudes=amplitudes,
        amplitude_errors=np.full_like(amplitudes, 1e-6),
        species=model.species_names(),
        wavelengths=wavelengths,
        times=times,
        diagnostics=diagnostics,
        backend={},
        _cache={
            "lam": 0.0,
            "noise": noise,
            "kinetic_model": model,
            "beta": beta,
            "cov_beta": np.array([[0.04, 0.01], [0.01, 0.09]]),
            "ci_sigma": 1.0,
            "ci_level": 0.6827,
            "parameterization": "log",
            "traces": traces,
            "solver_info": {},
        },
    )


class DiagnosticTests(unittest.TestCase):
    def test_per_point_standard_deviations_are_not_summed(self):
        self.assertEqual(reduced_chi_square([1, 2], [0, 0], 0, [1, 2]), 1.0)
        self.assertEqual(reduced_chi_square([1, 1], [0, 0], 0, [1, 1]), 1.0)

    def test_per_wavelength_weights_and_total_parameter_count(self):
        data = np.tile([1.0, 2.0], (3, 1))
        diag = compute_diagnostics(
            y_obs=data, y_fit=np.zeros_like(data), noise=[1, 2], n_params=2
        )
        self.assertEqual(diag["chi2"], 6.0)
        self.assertEqual(diag["DoF"], 4.0)
        self.assertEqual(diag["chi2_red"], 1.5)
        self.assertEqual(diag["n_obs"], 6)

    def test_full_heteroscedastic_gaussian_likelihood(self):
        diag = compute_diagnostics(
            y_obs=[1, 2, 0, 0], y_fit=[0, 0, 0, 0],
            noise=[1, 2, 1, 2], n_params=1,
        )
        minus_two_log_likelihood = 2 + 4 * np.log(2 * np.pi) + 4 * np.log(2)
        self.assertAlmostEqual(diag["log_likelihood"], -0.5 * minus_two_log_likelihood)
        self.assertAlmostEqual(diag["AIC"], minus_two_log_likelihood + 2)
        self.assertAlmostEqual(diag["BIC"], minus_two_log_likelihood + np.log(4))
        self.assertAlmostEqual(diag["AICc"], diag["AIC"] + 2)

    def test_fractional_effective_count_and_insufficient_dof(self):
        diag = compute_diagnostics(y_obs=[1, 2], y_fit=[0, 0], noise=[1, 2], n_params=0.5)
        self.assertEqual(diag["DoF"], 1.5)
        self.assertAlmostEqual(diag["chi2_red"], 2 / 1.5)
        for count in (2, 3):
            diag = compute_diagnostics(y_obs=[1, 2], y_fit=[0, 0], noise=1, n_params=count)
            self.assertEqual(diag["DoF"], 2 - count)
            self.assertTrue(np.isnan(diag["chi2_red"]))
            self.assertTrue(np.isnan(diag["AICc"]))
            self.assertTrue(np.isnan(reduced_chi_square([1, 2], [0, 0], count, 1)))

    def test_invalid_data_noise_shapes_and_counts_raise(self):
        bad_arguments = [
            {"y_fit": [[0], [0]]},
            {"noise": [[1], [1]]},
            {"noise": [1, 0]},
            {"noise": [1, -1]},
            {"noise": [1, np.nan]},
            {"noise": np.inf},
            {"y_obs": [1, np.nan]},
            {"y_fit": [0, np.inf]},
            {"n_params": -1},
            {"n_params": np.inf},
            {"n_params": [1]},
            {"y_obs": [], "y_fit": [], "noise": []},
        ]
        for bad in bad_arguments:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kwargs = {"y_obs": [1, 2], "y_fit": [0, 0], "noise": 1, "n_params": 0}
                kwargs.update(bad)
                compute_diagnostics(**kwargs)

    def test_constant_data_and_zero_inferred_noise_are_defined_as_nan(self):
        diag = compute_diagnostics(y_obs=[1, 1], y_fit=[1, 1], noise=1, n_params=0)
        self.assertTrue(np.isnan(diag["R2"]))
        self.assertTrue(np.isnan(reduced_chi_square([1, 1], [1, 1], 0)))


class ResultConsumerTests(unittest.TestCase):
    def setUp(self):
        self.result = make_result()

    def test_comparison_accepts_dataclass_results_and_undefined_criteria(self):
        inferior = replace(self.result, diagnostics={**self.result.diagnostics, "AICc": 100.0})
        undefined = replace(self.result, diagnostics={**self.result.diagnostics, "AICc": np.nan})
        rows = compare_models({"inferior": inferior, "best": self.result, "undefined": undefined})
        self.assertEqual([row["model"] for row in rows], ["best", "inferior", "undefined"])
        self.assertEqual(rows[0]["ΔAICc"], 0)
        self.assertTrue(np.isnan(rows[-1]["ΔAICc"]))
        self.assertEqual(compare_models({}), [])
        self.assertEqual(compare_models({"best": self.result}, "BIC")[0]["ΔBIC"], 0)

    def test_covariance_transforms_and_missing_covariance(self):
        result = self.result
        p = np.array([0.5, 2.0])
        expected_delta = result.cov_beta * p[:, None] * p[None, :]
        assert_allclose(result.cov_kinetics(method="delta"), expected_delta)
        assert_allclose(result.cov_kinetics(method="linear"), expected_delta)
        expected_exact = (
            p[:, None] * p[None, :]
            * np.exp(0.5 * (np.diag(result.cov_beta)[:, None] + np.diag(result.cov_beta)[None, :]))
            * np.expm1(result.cov_beta)
        )
        assert_allclose(result.cov_kinetics(), expected_exact)
        stored = result.cov_kinetics(space="log")
        stored[:] = 0
        self.assertGreater(result.cov_beta[0, 0], 0)
        no_cov = replace(result, _cache={**result._cache, "cov_beta": None})
        self.assertIsNone(no_cov.cov_kinetics())
        self.assertEqual(no_cov.to_dict()["beta"], result.beta.tolist())

    def test_linear_covariance_log_delta_requires_positive_parameters(self):
        result = replace(self.result, _cache={
            **self.result._cache, "parameterization": "linear", "beta": np.array([0.5, 2.0])
        })
        assert_array_equal(result.cov_kinetics(), result.cov_beta)
        assert_allclose(
            result.cov_kinetics(space="log", method="delta"),
            result.cov_beta / result.beta[:, None] / result.beta[None, :],
        )
        with self.assertRaises(ValueError):
            result.cov_kinetics(space="log")
        invalid = replace(result, _cache={**result._cache, "beta": np.array([0.0, 2.0])})
        with self.assertRaises(ValueError):
            invalid.cov_kinetics(space="log", method="delta")

    def test_small_log_covariance_does_not_cancel_to_zero(self):
        assert_allclose(cov_lognormal(np.array([0.0]), np.array([[1e-20]])), [[1e-20]], rtol=1e-15)
        assert_allclose(sd_lognormal(np.array([0.0]), np.array([1e-10])), [1e-10], rtol=1e-15)
        with self.assertRaises(ValueError):
            cov_lognormal(np.array([0.0]), np.array([1.0]))

    def test_report_renderer_options_are_not_sent_to_report_builder(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            latex = self.result.to_text(fmt="latex", caption="fit_1", label="tab:fit", digits=5)
        self.assertIn(r"\caption{fit\_1}", latex)
        self.assertIn(r"\label{tab:fit}", latex)
        self.assertIn("Global Fit Summary", self.result.to_text())
        self.assertIn("|", self.result.to_text(fmt="terminal", width=80))
        with self.assertRaises(ValueError):
            self.result.to_text(fmt="unknown")

    def test_latex_backslash_escape_does_not_escape_its_generated_braces(self):
        blocks = [ReportBlock("Path", [ReportRow("file", r"a\b_{c}")])]
        latex = render_latex_report(blocks)
        self.assertIn(r"a\textbackslash{}b\_\{c\}", latex)

    def test_species_reconstruction_and_fixed_fit_residuals(self):
        result = self.result
        fitted = reconstruct_fit(result)
        parts = np.asarray(reconstruct_species(result, "fast")) + np.asarray(reconstruct_species(result, 1))
        assert_allclose(parts, fitted)
        assert_allclose(reconstruct_species_sum(result, ["fast", 1]), fitted)
        assert_array_equal(reconstruct_species_sum(result, []), np.zeros_like(np.asarray(fitted)))
        error = np.outer(np.arange(1.0, 7.0), result._cache["noise"])
        observed = TransientAbsorption(
            np.asarray(fitted) + error, x=result.wavelengths, y=result.times,
            meta={"noise_t0": result._cache["noise"], "time_unit": "s"},
        )
        assert_allclose(reconstruct_residuals(result, observed), error, atol=1e-19)
        assert_allclose(reconstruct_weighted_residuals(result, observed), error / result._cache["noise"], atol=1e-13)
        self.assertEqual(reconstruct_residuals(result, observed).meta["time_unit"], "s")
        self.assertNotIn("noise_t0", reconstruct_weighted_residuals(result, observed).meta)

    def test_reconstruction_rejects_wrong_axes_and_invalid_species(self):
        result = self.result
        wrong_axes = TransientAbsorption(
            np.asarray(reconstruct_fit(result)), x=result.wavelengths, y=result.times + 1
        )
        with self.assertRaises(ValueError):
            reconstruct_residuals(result, wrong_axes)
        with self.assertRaises(TypeError):
            reconstruct_species(result, 0.5)
        with self.assertRaises(ValueError):
            reconstruct_species_sum(result, [0, "fast"])

    def test_residual_maps_accept_current_result_and_reproject_new_spectra(self):
        result = self.result
        observed = TransientAbsorption(
            result.traces @ (result.amplitudes * 3).T,
            x=result.wavelengths, y=result.times,
        )
        maps = compute_residual_maps(observed, result, noise=result._cache["noise"], lam=0)
        assert_allclose(maps["raw"], 0, atol=3e-18)
        assert_allclose(maps["weighted"], 0, atol=3e-12)
        legacy = {"_cache": {"model": result.kinetic_model, "beta": result.beta}}
        self.assertIsNone(compute_residual_maps(observed, legacy, lam=0)["weighted"])

    def test_wavelength_cv_counts_new_amplitudes_and_forwards_solver_options(self):
        result = self.result
        # Choose noise orthogonal to the kinetic basis. Projection therefore
        # leaves it unchanged, giving an independent check of held-out DoF.
        basis, _ = np.linalg.qr(result.traces, mode="complete")
        orthogonal = basis[:, 2] * np.sqrt(result.times.size - 2)
        observed = TransientAbsorption(
            result.traces @ result.amplitudes.T + orthogonal[:, None] * result._cache["noise"],
            x=result.wavelengths, y=result.times,
        )
        calls = []

        def fixed_training_fit(dataset, model, beta0, **kwargs):
            calls.append((dataset.shape, kwargs))
            return result

        with patch.object(TransientAbsorption, "fit_global_kinetics", fixed_training_fit):
            cv = cross_validate_wavelengths(
                observed, result.kinetic_model, result.beta,
                noise=result._cache["noise"], n_folds=2, lam=0,
                solver="least_squares", maxit=100,
            )
        assert_allclose(cv["chi2_red_folds"], [1.0, 1.0], atol=1e-12)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["solver"], "least_squares")
        self.assertEqual(calls[0][1]["maxit"], 100)


if __name__ == "__main__":
    unittest.main()
