"""Step-wise CalibrationSession (used by tool-calling agent frameworks such as NeMo Agent Toolkit)."""

from agentic_tcad_demo.controller import RulePlanner
from agentic_tcad_demo.session import CalibrationSession

from .conftest import CONFIG


def test_session_driven_step_by_step_matches_the_rule_planner_run(demo_run):
    session = CalibrationSession(CONFIG, None, "session(rule)")
    first = session.observe()
    assert first["finished"] is False
    assert {"J", "weighted_terms", "groups", "eligible_components", "legal_actions"} <= set(first["observation"])
    rule = RulePlanner()
    while not session.finished:
        a = rule.choose(session._current)
        reply = session.act(a.kind, a.target, a.rationale)
        assert "outcome" in reply
    s = session.result["summary"]
    assert s["planner"] == "session(rule)"
    assert s["J_history"] == demo_run["summary"]["J_history"]
    assert session.observe()["finished"] is True


def test_session_close_stops_and_reports(tmp_path):
    session = CalibrationSession(CONFIG, tmp_path, "session(test)")
    reply = session.act("update_group", "electrostatics", "test")
    assert reply["outcome"][0]["decision"] == "accepted"
    session.close()
    assert session.finished and "agent session ended" in session.summary()["stop_reason"]
    assert (tmp_path / "action_log.jsonl").exists() and (tmp_path / "summary.json").exists()
    assert session.act("stop", None, "again")["note"] == "calibration already finished"
