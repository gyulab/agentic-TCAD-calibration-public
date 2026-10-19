"""The accept/reject rule: a trial is kept only if it converges and lowers J."""

import numpy as np

from agentic_tcad_demo.controller import Action, Controller, Reviewer
from agentic_tcad_demo.device import CompactTFT, SimResult, synthesize_measurement
from agentic_tcad_demo.log import ActionLog
from agentic_tcad_demo.runner import all_conditions
from agentic_tcad_demo.tools import Workspace


def make_controller(cfg, simulator, planner):
    conds = all_conditions(cfg)
    measured = synthesize_measurement(cfg.truth, cfg.truth_params, cfg.truth_active, conds, cfg.vg, cfg.seed,
                                      cfg.noise["lognormal_sigma"], cfg.noise["floor_noise_A"])
    log = ActionLog()
    ws = Workspace(simulator, measured, cfg.vg, cfg.specs, cfg.groups, cfg.objective,
                   cfg.splits["calibration"], log, cfg.ofat_step)
    return Controller(ws, cfg.components, planner, Reviewer("policy", ()), cfg.controller, log), ws, log


class OnlyGroup:
    """Planner that insists on one group once, then stops."""

    def __init__(self, group):
        self.group, self.name, self.done = group, f"only-{group}", False

    def choose(self, obs):
        if not self.done and any(g["name"] == self.group and not g["tried"] for g in obs["groups"]):
            self.done = True
            return Action("update_group", self.group, "test")
        return Action("stop", None, "test done")


class FailsFarFromStart(CompactTFT):
    """Converges for the small OFAT probes, fails for the large Gauss-Newton trial step."""

    def __init__(self, start_phi_sd):
        super().__init__()
        self.start = start_phi_sd
        self.recoveries = 0

    def recovered(self):
        self.recoveries += 1
        return self  # the recovery attempt runs but cannot fix this failure

    def simulate(self, params, active, cond, vg):
        if abs(params["phi_SD"] - self.start) > 0.05:
            return SimResult(vg, np.full_like(vg, np.nan), False, "injected nonconvergence")
        return super().simulate(params, active, cond, vg)


def test_nonconverged_trial_is_rejected_logged_and_excluded(cfg):
    sim = FailsFarFromStart(cfg.initial["phi_SD"])
    ctrl, ws, log = make_controller(cfg, sim, OnlyGroup("contact"))
    res = ctrl.run(cfg.initial)
    trials = log.of_type("trial")
    assert trials and all(t["decision"] == "rejected" for t in trials)
    assert "numerical failure" in trials[0]["reason"] and trials[0]["J_trial"] is None
    assert res.params == cfg.initial  # nothing changed
    assert res.j_history == [res.j_history[0]]
    assert ws.n_failed_simulations >= 1
    assert sim.recoveries >= 1  # the predefined numerical-recovery step was tried first


def test_trial_that_increases_j_is_rejected(cfg):
    ctrl, ws, log = make_controller(cfg, CompactTFT(), OnlyGroup("contact"))
    ctrl._rejected = []
    params, active = dict(cfg.initial), frozenset()
    ev = ws.evaluate(params, active)
    rk = ws.rank_group("contact", params, active, 0.4)
    rk.jacobian = -rk.jacobian  # deliberately wrong model: the step now points uphill
    out = ctrl._try_group("contact", rk, params, active, ev, cycle=1)
    trials = log.of_type("trial")
    assert out is None
    assert len(trials) == cfg.controller.max_step_halvings + 1
    assert all(t["decision"] == "rejected" and t["J_trial"] >= t["J_before"] for t in trials)
    assert [t["trust_radius"] for t in trials] == sorted([t["trust_radius"] for t in trials], reverse=True)


def test_every_accepted_update_lowered_j_and_respected_bounds(demo_run):
    res, log = demo_run["result"], demo_run["log"]
    assert all(b < a for a, b in zip(res.j_history, res.j_history[1:]))
    for t in log.of_type("trial"):
        for name, ch in t["changes"].items():
            spec = demo_run["workspace"].specs[name]
            assert spec.lower <= ch["to"] <= spec.upper
        if t["decision"] == "accepted":
            assert t["J_trial"] < t["J_before"]
    assert len(log.of_type("trial")) == len(res.accepted) + len(res.rejected)


def test_illegal_planner_actions_are_counted_and_stop_the_run(cfg):
    class Bad:
        name = "bad"

        def choose(self, obs):
            return Action("update_group", "no_such_group", "")

    ctrl, _, log = make_controller(cfg, CompactTFT(), Bad())
    res = ctrl.run(cfg.initial)
    assert len(log.of_type("planner_error")) == cfg.controller.max_planner_errors
    assert "illegal" in res.stop_reason and res.params == cfg.initial


def test_railed_trigger_requires_the_residual_to_push_into_the_bound(cfg):
    from agentic_tcad_demo.tools import GroupRanking

    ctrl, _, _ = make_controller(cfg, CompactTFT(), OnlyGroup("electrostatics"))
    spec = cfg.specs["phi_G"]
    r, w = np.array([1.0, 2.0]), np.ones(2)

    def ranking(sign):
        jac = sign * np.ones((2, 1))  # d(residual)/dx for phi_G
        return {"electrostatics": GroupRanking("electrostatics", ("phi_G",), jac, r, w, np.zeros(1), 1.0)}

    at_lower = {**cfg.initial, "phi_G": spec.lower}
    at_upper = {**cfg.initial, "phi_G": spec.upper}
    inside = {**cfg.initial, "phi_G": 0.5 * (spec.lower + spec.upper)}
    # gradient > 0: descent wants a smaller phi_G
    assert ctrl._pushed_into_bound("phi_G", at_lower, ranking(+1))
    assert not ctrl._pushed_into_bound("phi_G", at_upper, ranking(+1))
    assert ctrl._pushed_into_bound("phi_G", at_upper, ranking(-1))
    assert not ctrl._pushed_into_bound("phi_G", inside, ranking(+1))
