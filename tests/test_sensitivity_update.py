import itertools

import numpy as np
import pytest

from agentic_tcad_demo.parameters import ParameterSpec
from agentic_tcad_demo.sensitivity import ofat_jacobian
from agentic_tcad_demo.update import bounded_gn_step

SPECS = {"a": ParameterSpec("a", 0.0, 10.0), "b": ParameterSpec("b", 1e16, 1e20, scale="log")}


def test_normalized_coordinates_round_trip_and_clip():
    for spec, v in ((SPECS["a"], 3.7), (SPECS["b"], 2.5e18)):
        assert spec.from_x(spec.to_x(v)) == pytest.approx(v)
        assert spec.from_x(1.7) == spec.upper and spec.from_x(-0.3) == spec.lower


def test_ofat_matches_the_analytic_jacobian_of_a_linear_model():
    A = np.array([[2.0, -1.0], [0.5, 3.0], [1.0, 1.0]])

    def residual(p):
        x = np.array([SPECS["a"].to_x(p["a"]), SPECS["b"].to_x(p["b"])])
        return A @ x, np.ones(3)

    p0 = {"a": 4.0, "b": 1e18}
    r0, _ = residual(p0)
    jac, probes = ofat_jacobian(residual, p0, ("a", "b"), SPECS, r0, step=0.02)
    assert np.allclose(jac, A, atol=1e-9)
    assert all(pr["converged"] for pr in probes)


def test_ofat_steps_backwards_at_the_upper_bound_and_zeroes_failed_probes():
    p0 = {"a": 10.0, "b": 1e18}
    calls = []

    def residual(p):
        calls.append(dict(p))
        return None if p["b"] != 1e18 else (np.array([p["a"]]), np.ones(1))

    jac, probes = ofat_jacobian(residual, p0, ("a", "b"), SPECS, np.array([10.0]), step=0.02)
    assert calls[0]["a"] < 10.0 and probes[0]["dx"] < 0
    assert not probes[1]["converged"] and np.all(jac[:, 1] == 0.0)


def test_gn_step_is_the_least_squares_solution_when_constraints_are_inactive():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(6, 2))
    s_true = np.array([0.05, -0.03])
    r = -S @ s_true
    out = bounded_gn_step(r, S, np.ones(6), x=np.array([0.5, 0.5]), delta=1.0)
    assert np.allclose(out.step, s_true, atol=1e-8)
    assert not out.ball_active


def brute_force(r, S, w, x, delta, n=401):
    best, arg = np.inf, None
    for s in itertools.product(np.linspace(-1, 1, n), repeat=2):
        s = np.array(s)
        if np.any(x + s < 0) or np.any(x + s > 1) or np.linalg.norm(s) > delta:
            continue
        val = 0.5 * np.sum(w * (r + S @ s) ** 2)
        if val < best:
            best, arg = val, s
    return best, arg


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gn_step_solves_the_constrained_problem_of_eq3(seed):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(5, 2))
    r = rng.normal(size=5) * 3
    w = rng.uniform(0.2, 1.0, size=5)
    x = rng.uniform(0.05, 0.95, size=2)
    delta = 0.25
    out = bounded_gn_step(r, S, w, x, delta)
    assert np.linalg.norm(out.step) <= delta + 1e-9
    assert np.all(x + out.step >= -1e-12) and np.all(x + out.step <= 1 + 1e-12)
    model = 0.5 * np.sum(w * (r + S @ out.step) ** 2)
    best, _ = brute_force(r, S, w, x, delta)
    assert model <= best + 1e-3 * max(best, 1.0)


def test_random_gn_steps_never_leave_the_box_or_the_trust_ball():
    rng = np.random.default_rng(42)
    for _ in range(200):
        k = rng.integers(1, 4)
        S = rng.normal(size=(8, k)) * rng.uniform(0.1, 50)
        r = rng.normal(size=8) * rng.uniform(0.1, 50)
        x = rng.uniform(0, 1, size=k)
        delta = rng.uniform(0.01, 0.6)
        s = bounded_gn_step(r, S, rng.uniform(0.1, 1, size=8), x, delta).step
        assert np.linalg.norm(s) <= delta + 1e-9
        assert np.all(x + s >= -1e-12) and np.all(x + s <= 1 + 1e-12)


def test_zero_sensitivity_gives_no_step():
    out = bounded_gn_step(np.ones(4), np.zeros((4, 1)), np.ones(4), np.array([0.5]), 0.3)
    assert np.all(out.step == 0) and out.model_reduction == 0.0
