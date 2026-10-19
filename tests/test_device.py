import numpy as np

from agentic_tcad_demo.device import CompactTFT, Condition, synthesize_measurement

VG = np.linspace(-1.5, 2.5, 201)
P = {"phi_SD": 4.40, "phi_G": 4.35, "mu0": 15.0, "Q_f": 0.0, "N_tail": 1e18}
NONE = frozenset()


def test_current_is_monotonic_in_gate_voltage():
    res = CompactTFT().simulate(P, NONE, Condition(1.0, 1.0), VG)
    assert res.converged
    assert np.all(np.diff(res.id) > 0)


def test_drain_bias_and_channel_length_change_the_current():
    dev = CompactTFT()
    i_lin = dev.simulate(P, NONE, Condition(0.05, 1.0), VG).id
    i_sat = dev.simulate(P, NONE, Condition(1.0, 1.0), VG).id
    i_long = dev.simulate(P, NONE, Condition(1.0, 2.0), VG).id
    assert i_sat[-1] > 5 * i_lin[-1]
    # with a fixed contact resistance, doubling L reduces on-current by less than 2x
    assert 1.0 < i_sat[-1] / i_long[-1] < 2.0


def test_contact_barrier_limits_current():
    dev = CompactTFT()
    low = dev.simulate({**P, "phi_SD": 4.75}, NONE, Condition(1.0, 1.0), VG).id[-1]
    high = dev.simulate({**P, "phi_SD": 4.30}, NONE, Condition(1.0, 1.0), VG).id[-1]
    assert high > 3 * low


def test_components_only_act_when_active():
    dev = CompactTFT()
    p = {**P, "Q_f": 5e12, "N_tail": 2e19}
    base = dev.simulate(p, NONE, Condition(1.0, 1.0), VG).id
    assert np.allclose(base, dev.simulate(P, NONE, Condition(1.0, 1.0), VG).id)
    assert dev.threshold_voltage(p, frozenset({"fixed_charge"})) < dev.threshold_voltage(p, NONE)
    assert dev.ideality(p, frozenset({"tail_states"})) > dev.ideality(p, NONE)


def test_invalid_domain_reports_nonconvergence():
    res = CompactTFT().simulate({**P, "phi_SD": 5.5}, NONE, Condition(1.0, 1.0), VG)
    assert not res.converged and "validity" in res.message
    assert not CompactTFT().simulate({**P, "mu0": -1.0}, NONE, Condition(1.0, 1.0), VG).converged


def test_self_consistent_solution_satisfies_the_circuit_equation():
    dev = CompactTFT()
    cond = Condition(1.0, 1.0)
    i = dev.simulate(P, NONE, cond, VG).id
    rc = dev.contact_resistance(P)
    vt, n = dev.threshold_voltage(P, NONE), dev.ideality(P, NONE)
    rhs = dev._intrinsic(VG - 0.5 * i * rc, cond.vd - i * rc, vt, n, P["mu0"], cond.length_um)
    assert np.allclose(i, rhs, rtol=1e-5, atol=1e-25)


def test_synthetic_measurement_is_deterministic_and_order_independent():
    truth = CompactTFT(mobility_gamma=0.25)
    conds = [Condition(0.05, 1.0), Condition(1.0, 1.0), Condition(0.1, 1.0)]
    a = synthesize_measurement(truth, P, NONE, conds, VG, 7, 0.03, 2e-12)
    b = synthesize_measurement(truth, P, NONE, conds, VG, 7, 0.03, 2e-12)
    c = synthesize_measurement(truth, P, NONE, conds, VG, 8, 0.03, 2e-12)
    for k in conds:
        assert np.array_equal(a[k], b[k])
        assert not np.array_equal(a[k], c[k])
