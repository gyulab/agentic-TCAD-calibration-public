import numpy as np
import pytest

from agentic_tcad_demo.metrics import crossing_voltage, extract_metrics
from agentic_tcad_demo.objective import PAPER_WEIGHTS, ObjectiveSpec, objective, residual_vector, score_curve

VG = np.linspace(-1.0, 2.0, 301)


def exponential_curve(v0, ss, i_max=1e-4):
    """Pure subthreshold exponential I = 100 nA * 10^((VG - v0)/ss), capped at i_max."""
    return np.minimum(100e-9 * 10 ** ((VG - v0) / ss), i_max)


def test_vth_and_ss_on_an_analytic_curve():
    m = extract_metrics(VG, exponential_curve(v0=0.3, ss=0.08))
    assert m.vth == pytest.approx(0.3, abs=1e-6)  # exact: log-linear interpolation of an exponential
    assert m.ss == pytest.approx(0.08, rel=1e-6)
    assert m.ion == pytest.approx(1e-4)


def test_missing_crossing_is_nan_not_extrapolated():
    assert np.isnan(crossing_voltage(VG, np.full_like(VG, 1e-9), 100e-9))
    assert np.isnan(extract_metrics(VG, np.full_like(VG, 1e-9)).vth)


def test_paper_weights_are_the_default():
    assert ObjectiveSpec().weights == {"log": 0.25, "lin": 0.10, "vth": 1.00, "ion": 0.20, "ss": 0.25}
    assert PAPER_WEIGHTS == ObjectiveSpec().weights


def test_perfect_fit_gives_zero_objective():
    meas = exponential_curve(0.3, 0.08)
    s = score_curve(VG, meas, meas, ObjectiveSpec())
    assert s.j == pytest.approx(0.0, abs=1e-12)


def test_hand_computed_objective_for_a_pure_shift():
    spec = ObjectiveSpec()
    meas = exponential_curve(0.30, 0.08)
    sim = exponential_curve(0.35, 0.08)  # +50 mV shift, same SS and same I_on cap
    s = score_curve(VG, sim, meas, spec)
    assert s.signed["vth"] == pytest.approx(0.05 / 0.1, rel=1e-6)
    assert s.terms["ss"] == pytest.approx(0.0, abs=1e-6)
    assert s.terms["ion"] == pytest.approx(0.0, abs=1e-9)
    # log-current error: -0.05/0.08 decades wherever both curves are exponential and above the floor
    mask = meas >= spec.measurement_floor_a
    expected_log = np.sqrt(np.mean((np.log10(sim[mask]) - np.log10(meas[mask])) ** 2)) / 0.1
    assert s.terms["log"] == pytest.approx(expected_log)
    expected_j = 0.25 * expected_log + 0.10 * s.terms["lin"] + 1.0 * 0.5
    assert s.j == pytest.approx(expected_j)
    assert objective([s, s]) == pytest.approx(expected_j)  # mean over biases


def test_points_below_the_measurement_floor_are_excluded():
    spec = ObjectiveSpec()
    meas = exponential_curve(0.3, 0.08)
    sim = meas.copy()
    floor = meas < spec.measurement_floor_a
    assert floor.any()
    sim[floor] *= 1e3  # huge errors, but only where the "measurement" is below 10 pA
    assert score_curve(VG, sim, meas, spec).terms["log"] == pytest.approx(0.0, abs=1e-12)


def test_residual_vector_shape_is_fixed_by_the_measurement():
    spec = ObjectiveSpec()
    meas = exponential_curve(0.3, 0.08)
    r1, w1 = residual_vector(VG, exponential_curve(0.25, 0.09), meas, spec)
    r2, w2 = residual_vector(VG, exponential_curve(0.45, 0.07), meas, spec)
    assert r1.shape == r2.shape == w1.shape
    assert np.array_equal(w1, w2) and np.all(w1 > 0)
