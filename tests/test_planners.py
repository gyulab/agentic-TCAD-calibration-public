"""External (file-exchange) planner and deterministic replay of recorded decisions."""

import json
import threading
import time

from agentic_tcad_demo.external import ExternalPlanner, ReplayPlanner
from agentic_tcad_demo.runner import run_demo

from .conftest import ROOT

OPUS = ROOT / "examples" / "planner_runs" / "claude-opus-5-5"


def answer_later(directory, n, text, delay=0.3):
    def write():
        time.sleep(delay)
        (directory / f"act_{n:03d}.json").write_text(text)

    threading.Thread(target=write, daemon=True).start()


def test_external_planner_exchanges_files(tmp_path):
    planner = ExternalPlanner(tmp_path, "test-agent", poll_s=0.05, timeout_s=10)
    answer_later(tmp_path, 0, '{"action": "update_group", "target": "contact", "rationale": "because"}')
    action = planner.choose({"cycle": 1, "J": 1.0})
    assert (action.kind, action.target, action.rationale) == ("update_group", "contact", "because")
    assert json.loads((tmp_path / "obs_000.json").read_text())["cycle"] == 1
    assert (tmp_path / "INSTRUCTIONS.md").exists()
    assert planner.name == "external(test-agent)"


def test_external_planner_rejects_malformed_answers(tmp_path):
    planner = ExternalPlanner(tmp_path, "test-agent", poll_s=0.05, timeout_s=10)
    answer_later(tmp_path, 0, '{"action": "update_group", "target": ["contact"]}')
    assert planner.choose({"cycle": 1}).kind == "invalid"


def test_recorded_opus_run_replays_exactly(cfg):
    recorded = json.loads((OPUS / "summary.json").read_text())
    assert recorded["planner"].startswith("external(claude-opus-5-5")
    out = run_demo(cfg, None, planner=ReplayPlanner(OPUS / "action_log.jsonl"))
    s = out["summary"]
    assert s["planner"] == "replay(" + recorded["planner"] + ")"
    assert s["J_history"] == recorded["J_history"]
    assert s["final_params"] == recorded["final_params"]
    assert s["evaluation"] == recorded["evaluation"]
    assert out["log"].of_type("planner_error") == []


def test_replay_detects_divergence():
    rp = ReplayPlanner(OPUS / "action_log.jsonl")
    assert rp.choose({"cycle": 99}).kind == "invalid"  # recorded decision belongs to cycle 1
