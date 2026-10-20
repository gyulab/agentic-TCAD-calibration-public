# NVIDIA NeMo Agent Toolkit and Agent Skills

The paper ran its workflow through the NVIDIA NeMo Agent Toolkit (NAT) and Agent Skills. This repository
provides compatible entry points for both. They are optional: the default demo needs neither.

| Piece | Where | What it does |
|---|---|---|
| NAT plugin | `src/agentic_tcad_demo/nat_plugin.py`, entry point `nat.plugins` → `agentic_tcad = agentic_tcad_demo.nat_plugin` | Registers the tools `tcad_observe`, `tcad_update_group`, `tcad_activate_component`, `tcad_stop`, and an offline LLM `tcad_scripted_llm` |
| NAT workflows | `examples/nat/offline_mock.yml`, `nim_react.yml`, `openai_compatible.yml`, `task.md` | `react_agent` configurations: offline scripted, NVIDIA NIM, any OpenAI-compatible endpoint |
| Session layer | `src/agentic_tcad_demo/session.py` | Runs the unchanged controller in a thread so that a tool-calling agent can drive it one action at a time |
| Agent Skill | `skills/agentic-tcad-calibration/SKILL.md` (+ `references/decision-rules.md`) | Instructions for coding agents (Codex, Claude Code, …) in the Agent Skills format |
| Exact versions | `requirements-nat-lock.txt` | Environment used for the NAT tests (Python 3.11, `nvidia-nat` 1.9.0) |

## How the pieces fit

1. The NAT agent (an LLM in a ReAct loop) calls `tcad_observe`.
2. It then calls one of the three action tools with a target and a one-sentence rationale.
3. Each tool call passes one action to the same controller used by every other planner. The controller
   runs OFAT and the bounded Gauss–Newton trial, applies the accept-if-J-decreases rule and the
   activation eligibility and approval rules, and returns the outcome and the next observation as JSON.
4. When the agent calls `tcad_stop`, the model is held fixed and evaluated on the held-out conditions,
   and `action_log.jsonl`, `summary.json` and `final_params.yaml` are written to the configured `out_dir`.

If the agent ends without calling `tcad_stop`, the plugin stops the session itself and still writes the
outputs. The LLM never sets a parameter value.

## Install

The NAT integration needs Python 3.11–3.13. It uses a separate environment so the default demo stays light.

```bash
python3.11 -m venv .venv-nat && source .venv-nat/bin/activate
pip install -r requirements-nat-lock.txt     # exact versions tested here (nvidia-nat 1.9.0)
pip install -e . --no-deps                   # registers the nat.plugins entry point
nat info components -t function | grep tcad  # should list the four tcad_* tools
```

Alternatively, run `pip install -e ".[nat,dev]"`. It resolves `nvidia-nat[langchain]==1.9.0` instead of
using the full lock file.

## Run

**Offline check.** No network or API key. The scripted LLM replies with fixed ReAct steps, so this tests
the integration, not reasoning:

```bash
nat run --config_file examples/nat/offline_mock.yml --input "Calibrate the synthetic TFT."
# outputs/nat_offline/{action_log.jsonl, summary.json, final_params.yaml}
```

**Live LLM through NVIDIA NIM.** Requires a key from build.nvidia.com:

```bash
export NVIDIA_API_KEY=...
nat run --config_file examples/nat/nim_react.yml --input "$(cat examples/nat/task.md)"
# outputs/nat_nim/...
```

**Any OpenAI-compatible server** (OpenAI, vLLM, Ollama). Edit `model_name`, `base_url` and
`planner_label` in `examples/nat/openai_compatible.yml`, then run it the same way.

The `planner_label` field is written into the action log as `nat(<label>)`. Put the real model id there,
so every decision stays attributable.

## Reproduce an LLM run without the LLM

LLM outputs are not deterministic, even at temperature 0, and hosted models change. The reproducible
artifact is therefore the action log. Anyone can replay it in the default environment, with no NAT, LLM or
API key:

```bash
python -m agentic_tcad_demo run --planner replay --replay-log outputs/nat_nim/action_log.jsonl --out outputs/replay
```

The replay reproduces the J trajectory, the final parameters and the held-out tables exactly. It is
labelled `replay(nat(...))`. Commit the action log and `summary.json` of any run you report, as in
`examples/planner_runs/`.

## Agent Skill for coding agents

The skill follows the Agent Skills format (`SKILL.md` with `name`, `description`, `license`,
`compatibility`, `metadata`). The `skills` CLI installs it from this repository (Node.js ≥ 22):

```bash
npx skills add gyulab/agentic-TCAD-calibration-public --list
npx skills add gyulab/agentic-TCAD-calibration-public --skill agentic-tcad-calibration --agent codex
npx skills add gyulab/agentic-TCAD-calibration-public --skill agentic-tcad-calibration --agent claude-code
```

Install locations, checked with a project-level install:

| Agent | Path |
|---|---|
| Codex | `./.agents/skills/agentic-tcad-calibration/` |
| Claude Code | `./.claude/skills/agentic-tcad-calibration/` (symlink to the copy above) |

With `--global`, the skill goes to the agent's user-level skills directory instead.

The skill teaches an agent four things:
- how to set up and run the demo
- how to act as the planner through `--planner external`
- how to use the NAT workflows
- how to report results honestly (synthetic numbers, whether the run was blind, replay labelling)

This is a community skill. It is **not** an NVIDIA-verified or signed skill, and it is not part of the
`nvidia/skills` catalog.

## Tested here

- `nvidia-nat` 1.9.0 with `nvidia-nat-langchain` 1.9.0 on Python 3.11. The plugin is discovered through
  the entry point, and the offline `react_agent` workflow completes with no errors.
- The NAT run's action log replays exactly in the default environment (`tests/test_nat_plugin.py`).
- `npx skills` discovers the skill and installs it for Codex and Claude Code.

## Not tested here

- A live run against NIM or another hosted LLM. No API key was available when this integration was
  written.
- NAT's own `nat_test_llm` cannot drive `react_agent` in NAT 1.9, because it is not a LangChain Runnable.
  The plugin therefore ships `tcad_scripted_llm` for offline checks.
