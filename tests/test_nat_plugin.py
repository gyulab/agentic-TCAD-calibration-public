"""NVIDIA NeMo Agent Toolkit integration. Runs only where `pip install -e ".[nat]"` was done."""

import json
import shutil
import subprocess

import pytest

from agentic_tcad_demo.external import ReplayPlanner
from agentic_tcad_demo.runner import run_demo

from .conftest import ROOT

pytest.importorskip("nat")
NAT = shutil.which("nat")
FUNCS = ("tcad_observe", "tcad_update_group", "tcad_activate_component", "tcad_stop")


def test_plugin_registers_the_tools():
    from nat.cli.type_registry import GlobalTypeRegistry

    import agentic_tcad_demo.nat_plugin  # noqa: F401  (entry-point module)

    names = {f.config_type.static_type() for f in GlobalTypeRegistry.get().get_registered_functions()}
    assert set(FUNCS) <= names


@pytest.mark.skipif(NAT is None, reason="nat CLI not on PATH")
def test_offline_react_agent_drives_the_controller(cfg, tmp_path):
    overrides = []
    for f in FUNCS:
        overrides += ["--override", f"functions.{f}.out_dir", str(tmp_path)]
    proc = subprocess.run([NAT, "run", "--config_file", str(ROOT / "examples/nat/offline_mock.yml"),
                           "--input", "Calibrate the synthetic TFT.", *overrides],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert " ERROR " not in proc.stdout + proc.stderr
    events = [json.loads(x) for x in (tmp_path / "action_log.jsonl").read_text().splitlines()]
    proposals = [(e["action"], e["target"]) for e in events if e["event"] == "proposal"]
    assert proposals == [("update_group", "electrostatics"), ("update_group", "contact"),
                         ("update_group", "electrostatics"), ("activate_component", "fixed_charge"),
                         ("stop", None)]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["planner"] == "nat(react_agent/tcad_scripted_llm)"
    assert summary["activations"] == ["fixed_charge"] and summary["J_final"] < summary["J_initial"]
    replay = run_demo(cfg, None, planner=ReplayPlanner(tmp_path / "action_log.jsonl"))["summary"]
    assert replay["J_history"] == summary["J_history"]
