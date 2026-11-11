"""Reproduce the numerical validation of the analytical TTA evaluator.

Run from a source checkout or unpacked source distribution::

    python examples/validate_analytic_domain.py --benchmark

The default output is docs/analytic-validation.json. The 70-case comparison
runs by default; timing 200/1000-point curves is optional. Numerical ODE
integration is an independent validation reference in this script, never
part of the production analytical evaluator. Results are observed numerical
differences, not rigorous bounds on total floating-point error.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import tomllib

import numpy as np
import scipy
from scipy.integrate import solve_ivp


SOURCE_ROOT = Path(__file__).resolve().parents[1]
# Validate the code beside this source-only script, including before an
# editable installation exists. Do not accidentally benchmark an older wheel.
sys.path.insert(0, str(SOURCE_ROOT))

from phoskhemia.kinetics import analytic_shape, series_order, triplet_population


NU_VALUES = (0.0, 1e-12, 1 / 6000, 0.1, 1 - 1e-10, 1.0, 1 + 1e-10, 2.0, 10.0, 100.0)
ETA_VALUES = (0.0, 1e-16, 1e-8, 1e-4, 0.1, 1.0, 10.0)
REFERENCE_RTOL = 1e-13
REFERENCE_ATOL = 1e-15


def ode_reference(s, nu, eta, *, method=None, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL):
    """Independently integrate a factored form of the scalar Riccati equation.

    For w'=exp(-s)-nu*w-eta*w**2, write w=exp(-r*s)*v with r=min(nu,1).
    Then v'=exp(-(1-r)*s)-(nu-r)*v-eta*exp(-r*s)*v**2. Factoring avoids
    imposing an absolute tolerance directly on an exponentially small w.
    """
    r = min(nu, 1.0)

    def rhs(x, v):
        return np.exp(-(1 - r) * x) - (nu - r) * v - eta * np.exp(-r * x) * v**2

    def jacobian(x, v):
        return np.array([[-(nu - r) - 2 * eta * np.exp(-r * x) * v[0]]])

    if method is None:
        method = "Radau" if (nu - r) * s[-1] > 500 else "DOP853"
    kwargs = {"jac": jacobian} if method == "Radau" else {}
    solution = solve_ivp(
        rhs, (0.0, float(s[-1])), [0.0], method=method,
        t_eval=s, rtol=rtol, atol=atol, **kwargs,
    )
    if not solution.success:
        raise RuntimeError(f"Independent ODE reference failed: {solution.message}")
    return np.exp(-r * s) * solution.y[0], {
        "method": method,
        "function_evaluations": solution.nfev,
    }


def compare(actual, reference, coordinates):
    """Record the complete location and values of the largest discrepancy."""
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise FloatingPointError("A validation curve contains nonfinite values")
    index = int(np.argmax(np.abs(actual - reference)))
    return {
        "max_absolute_error": float(abs(actual[index] - reference[index])),
        "at_coordinate": float(coordinates[index]),
        "analytical_value": float(actual[index]),
        "reference_value": float(reference[index]),
        "minimum_analytical_value": float(np.min(actual)),
        "maximum_analytical_value": float(np.max(actual)),
    }


def validate_domain():
    s = np.unique(np.r_[0.0, np.geomspace(1e-9, 1e6, 241), 0.01, 0.1, 1.0, 10.0, 100.0, 1e4])
    cases = []
    for eta in ETA_VALUES:
        for nu in NU_VALUES:
            reference, solver = ode_reference(s, nu, eta)
            actual = analytic_shape(s, 1.0, nu, eta)
            cases.append({
                "eta": eta, "nu": nu, "series_order": series_order(eta),
                "reference_solver": solver, **compare(actual, reference, s),
            })
    worst = max(cases, key=lambda row: row["max_absolute_error"])

    # Tighten the independent reference at the worst sampled coordinate and
    # cross-check it with two different integration algorithms.
    checks = []
    coordinate = np.array([worst["at_coordinate"]])
    for method in ("DOP853", "Radau"):
        reference, solver = ode_reference(
            coordinate, worst["nu"], worst["eta"], method=method,
            rtol=3e-14, atol=3e-16,
        )
        checks.append({
            "rtol": 3e-14, "atol": 3e-16, "solver": solver,
            "reference_value": float(reference[0]),
            "absolute_error_from_analytical": float(abs(worst["analytical_value"] - reference[0])),
        })

    return {
        "coordinate": "s = beta*t",
        "normalization": "w = n/(P/beta); nu = gamma/beta; eta = q*P/beta**2",
        "s_values": s.tolist(),
        "nu_values": list(NU_VALUES),
        "eta_values": list(ETA_VALUES),
        "number_of_cases": len(cases),
        "time_points_per_case": len(s),
        "cases": cases,
        "worst_case": worst,
        "equivalent_error_M_at_concentration_scale_1e-6_M": worst["max_absolute_error"] * 1e-6,
        "worst_case_tighter_reference_checks": checks,
    }


def validate_physical_example():
    times = np.geomspace(1e-9, 1e-2, 241)
    P, beta, gamma = 3.0, 6e6, 1e3
    q_values = (1e-3, 1.0, 1e3, 1e6, 1e9, 1e10)
    cases = []
    for q in q_values:
        eta = q * P / beta**2
        reference, solver = ode_reference(beta * times, gamma / beta, eta)
        actual = triplet_population(times, P, beta, gamma, q)
        comparison = compare(actual, (P / beta) * reference, times)
        cases.append({
            "q_M_inv_s_inv": q, "eta": eta, "reference_solver": solver,
            **comparison,
            "equivalent_normalized_error": comparison["max_absolute_error"] / (P / beta),
        })
    return {
        "coordinate_unit": "s", "concentration_unit": "M",
        "P_M_s_inv": P, "beta_s_inv": beta, "gamma_s_inv": gamma,
        "q_values_M_inv_s_inv": list(q_values),
        "times_s": times.tolist(),
        "number_of_cases": len(cases), "time_points_per_case": len(times),
        "cases": cases,
        "worst_case": max(cases, key=lambda row: row["max_absolute_error"]),
    }


def benchmark(repeats):
    results = []
    cases = (
        ("experimental", 6e6, 1e3, 1e9 * 3 / (6e6)**2, 1e-9, 1e-2),
        ("eta10_broad_window", 1.0, 1 / 6000, 10.0, 1e-9, 1e6),
    )
    for name, beta, gamma, eta, lower, upper in cases:
        for count in (200, 1000):
            times = np.geomspace(lower, upper, count)
            analytic_shape(times, beta, gamma, eta)  # one untimed warm-up
            durations = []
            for _ in range(repeats):
                start = perf_counter()
                analytic_shape(times, beta, gamma, eta)
                durations.append(perf_counter() - start)
            results.append({
                "case": name, "number_of_times": count,
                "beta": beta, "gamma": gamma, "eta": eta,
                "times": {"spacing": "geomspace", "minimum": lower, "maximum": upper},
                "series_order": series_order(eta),
                "durations_s": durations, "median_s": float(np.median(durations)),
                "minimum_s": min(durations), "maximum_s": max(durations),
            })
    return {"repeats": repeats, "warmup_runs_per_case": 1, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="also time 200/1000-point evaluations")
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions per benchmark case (default: 5)")
    parser.add_argument("--output", type=Path, default=SOURCE_ROOT / "docs/analytic-validation.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    start = perf_counter()
    configuration = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text())
    domain = validate_domain()
    physical = validate_physical_example()
    timing = benchmark(args.repeats) if args.benchmark else None
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": perf_counter() - start,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
            "package_version": configuration["project"]["version"],
            "declared_python_requirement": configuration["project"]["requires-python"],
        },
        "source_sha256": {
            name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
            for name in ("phoskhemia/kinetics/analytic.py", "examples/validate_analytic_domain.py")
        },
        "reference": {
            "equation": "v'=exp(-(1-r)*s)-(nu-r)*v-eta*exp(-r*s)*v**2; r=min(nu,1); v(0)=0; w=exp(-r*s)*v",
            "rtol": REFERENCE_RTOL, "atol": REFERENCE_ATOL,
            "method_rule": "Radau if (nu-min(nu,1))*max(s)>500; DOP853 otherwise",
        },
        "analytical_series_tolerance": 1e-14,
        "dimensionless": domain,
        "physical_example": physical,
        "benchmark": timing,
        "notes": [
            "Observed discrepancies include numerical error in both the analytical evaluator and the independent ODE reference.",
            "The analytical series tolerance bounds truncation only; it does not bound total floating-point evaluation error.",
            "Benchmarks measure one process on this machine and vary with hardware and system load.",
            "This run validates only the interpreter and library versions recorded above, not the entire declared support matrix.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Compared {domain['number_of_cases']} normalized cases at {domain['time_points_per_case']} times each.")
    print(f"Maximum observed normalized difference: {domain['worst_case']['max_absolute_error']:.6g}")
    print(f"Maximum observed physical concentration difference: {physical['worst_case']['max_absolute_error']:.6g} M")
    if timing is not None:
        for row in timing["cases"]:
            print(f"{row['case']}, {row['number_of_times']} times: {row['median_s'] * 1e3:.3f} ms median")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
