"""End-to-end properties of the default run: model activation, held-out isolation, reproducibility."""

import json

import numpy as np
import pytest

import agentic_tcad_demo.tools as tools_mod
import agentic_tcad_demo.update as update_mod
from agentic_tcad_demo.config import load_config
from agentic_tcad_demo.controller import Reviewer
from agentic_tcad_demo.evaluate import evaluate_fixed
from agentic_tcad_demo.parameters import params_hash
from agentic_tcad_demo.runner import run_demo

from .conftest import CONFIG


def test_default_run_improves_j_and_stops_for_a_logged_reason(demo_run):
    s = demo_run["summary"]
    assert s["J_final"] < s["J_initial"]
    assert s["stop_reason"]
    assert demo_run["log"].of_type("stop")[0]["reason"] == s["stop_reason"]


def test_components_are_activated_only_with_evidence_and_approval(demo_run):
    log, cfg_s = demo_run["log"], demo_run["result"]
    approvals = {e["component"]: e for e in log.of_type("approval")}
    assert cfg_s.activations == ["fixed_charge", "tail_states"]
    assert set(cfg_s.activations) <= set(approvals)
    for name in cfg_s.activations:
        ev = approvals[name]["evidence"]
        assert approvals[name]["approved"] and ev["share_of_J"] >= 0.15
    if "tail_states" in approvals:
        assert "dr_ss/dx" in approvals["tail_states"]["evidence"]["reason"]
        # SS was a large share of J for at least `persistence_cycles` diagnoses before activation
        cycle = approvals["tail_states"]["cycle"]
        shares = [d["terms"]["ss"] / sum(d["terms"].values()) for d in log.of_type("diagnosis") if d["cycle"] <= cycle]
        assert min(shares[-2:]) >= 0.15
    if "fixed_charge" in approvals:
        assert "phi_G" in approvals["fixed_charge"]["evidence"]["reason"]


def test_activation_needs_approval(cfg):
    out = run_demo(cfg, None, reviewer=Reviewer("policy", approved=()))
    assert out["result"].activations == []
    assert all(not e["approved"] for e in out["log"].of_type("approval"))
    assert out["result"].active == frozenset()


def test_heldout_data_never_influences_calibration(cfg):
    base = run_demo(cfg, None)
    held = [c for k, v in cfg.splits.items() if k != "calibration" for c in v]
    original = tools_mod.Workspace.__init__

    def corrupt(self, simulator, measured, *a, **kw):
        measured = {c: (m * 10.0 if c in held else m) for c, m in measured.items()}
        original(self, simulator, measured, *a, **kw)

    tools_mod.Workspace.__init__ = corrupt
    try:
        other = run_demo(cfg, None)
    finally:
        tools_mod.Workspace.__init__ = original
    assert other["summary"]["J_history"] == base["summary"]["J_history"]
    assert other["summary"]["final_params"] == base["summary"]["final_params"]
    assert other["summary"]["evaluation"]["heldout_drain"] != base["summary"]["evaluation"]["heldout_drain"]


def test_heldout_evaluation_does_not_retune(demo_run, monkeypatch):
    ws, res = demo_run["workspace"], demo_run["result"]

    def forbidden(*a, **k):
        raise AssertionError("an update was attempted during held-out evaluation")

    monkeypatch.setattr(update_mod, "bounded_gn_step", forbidden)
    monkeypatch.setattr(tools_mod, "bounded_gn_step", forbidden)
    before = params_hash(res.params, res.active)
    conds = demo_run["workspace"].calibration[:0] + [c for c in ws.measured if c not in ws.calibration]
    n_before = ws.n_simulations
    out = evaluate_fixed(ws, res.params, res.active, conds)
    assert ws.n_simulations - n_before == len(conds)  # one forward simulation per condition, nothing else
    assert out["params_hash"] == before == params_hash(res.params, res.active)
    assert {r["condition"] for r in out["rows"]} == {c.label for c in conds}


def test_splits_follow_the_paper_structure(cfg):
    cal = cfg.splits["calibration"]
    assert {c.vd for c in cal} == {0.05, 1.0} and {c.length_um for c in cal} == {1.0}
    assert {c.vd for c in cfg.splits["heldout_drain"]} == {0.1, 0.2, 0.5}
    assert {c.length_um for c in cfg.splits["heldout_length"]} == {2.0}
    assert not set(cal) & set(cfg.splits["heldout_drain"] + cfg.splits["heldout_length"])


def test_two_runs_are_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_demo(load_config(CONFIG), a)
    run_demo(load_config(CONFIG), b)
    assert (a / "summary.json").read_bytes() == (b / "summary.json").read_bytes()
    assert (a / "action_log.jsonl").read_bytes() == (b / "action_log.jsonl").read_bytes()
    log = [json.loads(line) for line in (a / "action_log.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "run_start" and log[-1]["event"] == "evaluation"


def test_different_seed_changes_the_data_but_not_the_workflow(cfg, tmp_path):
    other = load_config(CONFIG)
    other.seed = cfg.seed + 1
    s = run_demo(other, None)["summary"]
    assert s["J_final"] < s["J_initial"]
    assert s["J_history"] != run_demo(cfg, None)["summary"]["J_history"]


@pytest.mark.parametrize("name", ["fig1_initial_vs_target.png", "fig5_heldout_evaluation.png"])
def test_figures_are_written(demo_run, tmp_path, name):
    from agentic_tcad_demo.plots import write_figures

    paths = write_figures(demo_run, tmp_path)
    assert (tmp_path / name).exists() and len(paths) == 5
    assert np.all([p.stat().st_size > 10_000 for p in paths])


@pytest.mark.parametrize("trust_radius", [0.3, 0.5, 0.7])
def test_outcome_is_robust_to_the_trust_radius(trust_radius):
    cfg = load_config(CONFIG)
    cfg.controller.trust_radius = trust_radius
    s = run_demo(cfg, None)["summary"]
    assert set(s["activations"]) == {"fixed_charge", "tail_states"}
    assert s["J_final"] < 0.05 * s["J_initial"]
    assert s["evaluation"]["heldout_length"]["final"]["max_abs_dVth_mV"] < 50.0


@pytest.mark.parametrize("truth_components", [(), ("tail_states",), ("fixed_charge",)])
def test_activation_gate_is_selective(truth_components):
    """The candidate components share the truth model's functional forms, so the key check is the negative
    one: a component is activated only if the synthetic truth actually contains that physics."""
    cfg = load_config(CONFIG)
    cfg.truth_active = frozenset(truth_components)
    s = run_demo(cfg, None)["summary"]
    assert set(s["activations"]) == set(truth_components)


def test_premature_activation_is_refused(cfg):
    from agentic_tcad_demo.controller import Action

    class Eager:
        name = "eager"

        def choose(self, obs):
            return Action("activate_component", "tail_states", "activate immediately")

    out = run_demo(cfg, None, planner=Eager())
    assert out["result"].activations == []
    assert len(out["log"].of_type("planner_error")) == cfg.controller.max_planner_errors
    assert out["log"].of_type("approval") == []


@pytest.mark.parametrize("seed", [2600, 2603, 2606])
def test_workflow_is_robust_to_the_noise_seed(seed):
    cfg = load_config(CONFIG)
    cfg.seed = seed
    s = run_demo(cfg, None)["summary"]
    assert s["activations"] == ["fixed_charge", "tail_states"]
    assert s["J_final"] < 0.05 * s["J_initial"]


def test_readme_numbers_match_a_fresh_run(demo_run):
    """Keeps the README honest: its result table must be what the code actually produces."""
    import re

    from .conftest import ROOT

    text = (ROOT / "README.md").read_text()
    m = re.search(r"\| Objective J, initial → final \| ([\d.]+) → ([\d.]+) \((\d+) accepted updates, (\d+) rejected", text)
    assert m, "result row not found in README"
    s = demo_run["summary"]
    assert float(m.group(1)) == pytest.approx(s["J_initial"], abs=0.006)
    assert float(m.group(2)) == pytest.approx(s["J_final"], abs=0.0006)
    assert (int(m.group(3)), int(m.group(4))) == (len(s["accepted_updates"]), s["rejected_trials"])
    for split in ("heldout_drain", "heldout_length"):
        assert f"{s['evaluation'][split]['final']['max_abs_dVth_mV']:.1f} mV" in text
