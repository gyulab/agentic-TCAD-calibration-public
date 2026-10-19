"""The optional LLM planner: never required, never silently replaced by the rule planner."""

import json

import pytest

from agentic_tcad_demo.cli import main
from agentic_tcad_demo.controller import RulePlanner
from agentic_tcad_demo.llm import LLMConfigError, LLMPlanner
from agentic_tcad_demo.runner import run_demo

from .conftest import CONFIG

ENV = ("AGENTIC_TCAD_LLM_BASE_URL", "AGENTIC_TCAD_LLM_MODEL", "AGENTIC_TCAD_LLM_API_KEY")


@pytest.fixture
def no_llm_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def test_missing_configuration_is_a_clear_error(no_llm_env):
    with pytest.raises(LLMConfigError, match="AGENTIC_TCAD_LLM_BASE_URL"):
        LLMPlanner.from_env()


def test_cli_exits_nonzero_without_llm_configuration(no_llm_env, tmp_path, capsys):
    assert main(["run", "--config", str(CONFIG), "--planner", "llm", "--out", str(tmp_path), "--no-figures"]) == 2
    assert "--planner rule" in capsys.readouterr().err
    assert not (tmp_path / "summary.json").exists()


def scripted_transport(policy):
    """Fake chat-completions endpoint whose 'model' answers with `policy(observation)`."""
    calls = []

    def transport(url, payload, headers, timeout):
        obs = json.loads(payload["messages"][1]["content"])
        calls.append({"url": url, "headers": headers, "obs": obs})
        return {"choices": [{"message": {"content": policy(obs)}}]}

    return transport, calls


def test_llm_planner_drives_the_same_tools(cfg):
    rule = RulePlanner()

    def policy(obs):  # a well-behaved "LLM" that happens to agree with the rule policy
        a = rule.choose(obs)
        return json.dumps({"action": a.kind, "target": a.target, "rationale": "scripted"})

    transport, calls = scripted_transport(policy)
    planner = LLMPlanner("http://llm.invalid/v1", "scripted-model", "dummy-key", transport)
    out = run_demo(cfg, None, planner=planner)
    assert out["summary"]["planner"] == "llm(model=scripted-model)"
    assert calls[0]["url"] == "http://llm.invalid/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer dummy-key"
    assert "legal_actions" in calls[0]["obs"] and "eligible_components" in calls[0]["obs"]
    assert out["summary"]["J_history"] == run_demo(cfg, None)["summary"]["J_history"]


def test_malformed_llm_targets_do_not_crash_the_run(cfg):
    replies = iter(['{"action": "update_group", "target": ["contact"]}',
                    '{"action": "delete_everything", "target": null}',
                    '{"action": "update_group", "target": 3}'])
    transport, _ = scripted_transport(lambda obs: next(replies))
    out = run_demo(cfg, None, planner=LLMPlanner("http://llm.invalid/v1", "odd", None, transport))
    assert len(out["log"].of_type("planner_error")) == 3
    assert "illegal" in out["summary"]["stop_reason"]


def test_invalid_llm_output_is_logged_and_never_falls_back(cfg):
    transport, _ = scripted_transport(lambda obs: "sure! I think you should lower the work function")
    out = run_demo(cfg, None, planner=LLMPlanner("http://llm.invalid/v1", "chatty", None, transport))
    errors = out["log"].of_type("planner_error")
    assert len(errors) == cfg.controller.max_planner_errors
    assert out["summary"]["accepted_updates"] == []
    assert all(p["planner"].startswith("llm(") for p in out["log"].of_type("proposal"))
