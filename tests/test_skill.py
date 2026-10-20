"""The Agent Skills package (agentskills.io format) stays valid and points at things that exist."""

import re

import yaml

from .conftest import ROOT

SKILL = ROOT / "skills" / "agentic-tcad-calibration"


def frontmatter():
    return yaml.safe_load((SKILL / "SKILL.md").read_text().split("---")[1])


def test_frontmatter_follows_the_agent_skills_spec():
    fm = frontmatter()
    assert set(fm) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fm["name"]) and len(fm["name"]) <= 64
    assert fm["name"] == SKILL.name
    assert 0 < len(fm["description"]) <= 1024 and len(fm.get("compatibility", "")) <= 500
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in fm["metadata"].items())
    assert "not NVIDIA-verified" in fm["metadata"]["verified"]


def test_skill_references_existing_files_and_cli_options():
    text = (SKILL / "SKILL.md").read_text()
    for rel in re.findall(r"\]\(([^)#]+)\)", text):
        assert (SKILL / rel).exists(), rel
    for path in ("examples/nat/offline_mock.yml", "examples/nat/nim_react.yml", "examples/nat/task.md",
                 "configs/synthetic_demo.yaml", "requirements-lock.txt"):
        assert path in text or path.split("/")[-1] in text
        assert (ROOT / path).exists(), path
    cli = (ROOT / "src" / "agentic_tcad_demo" / "cli.py").read_text()
    for flag in re.findall(r"(--[a-z][a-z-]+)", text):
        if flag not in ("--config_file", "--input", "--no-deps"):
            assert flag in cli, flag
