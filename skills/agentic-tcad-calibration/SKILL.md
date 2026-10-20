---
name: agentic-tcad-calibration
description: Run or drive the residual-guided, bounded, J-gated TCAD calibration demo (synthetic oxide-semiconductor TFT, arXiv:2609.12184). Use when the user asks to run the agentic TCAD calibration demo, act as its planning agent, connect it to the NVIDIA NeMo Agent Toolkit, or replay and compare recorded planner runs. Not for real TCAD decks, measured data, or Sentaurus.
license: MIT
compatibility: Python 3.10+ for the demo; Python 3.11-3.13 for the NeMo Agent Toolkit extra. No GPU, TCAD license or API key is needed for the default path.
metadata:
  author: Gyujun Jeong
  repository: https://github.com/gyulab/agentic-TCAD-calibration-public
  paper: https://arxiv.org/abs/2609.12184
  verified: "community skill, not NVIDIA-verified or signed"
---

# Agentic TCAD calibration (synthetic demo)

The calibration loop lives in the `agentic_tcad_demo` package. An agent using this skill only *chooses*
actions. The controller computes every number and enforces every rule:
- OFAT sensitivity and a bounded Gauss–Newton step (paper Eq. 3)
- accept a trial only if it converges and J decreases
- model activation only when eligible, and only with an approval record

Never edit `controller.py`, the bounds, or the objective weights to make a run look better.

## 1. Set up (from the repository root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e . --no-deps   # Python 3.11, exact versions
# or: pip install -e ".[dev]"                                       # Python 3.10-3.12
pytest -q                                                            # expect all tests to pass
```

## 2. Run the default demo (rule planner, no LLM)

```bash
python -m agentic_tcad_demo run --config configs/synthetic_demo.yaml --out outputs/demo
```

Report J initial → final, the accepted and rejected counts, the activations, the stop reason, and the
held-out errors from `outputs/demo/summary.json`.

## 3. Act as the planning agent yourself (coding agents such as Codex or Claude Code)

Start the run in the background with the file-exchange planner. Name yourself honestly, giving the model
and how it is run:

```bash
python -m agentic_tcad_demo run --planner external --exchange-dir outputs/exchange \
    --planner-name "<model id> via <agent harness>" --out outputs/my_run --no-figures &
```

For each `outputs/exchange/obs_NNN.json`, read the observation and write exactly one JSON action to
`outputs/exchange/act_NNN.json`. Write it atomically: write to a temporary file, then rename it.

```json
{"action": "update_group", "target": "contact", "rationale": "one sentence of device physics"}
```

The rules for choosing an action are in [references/decision-rules.md](references/decision-rules.md).
An illegal or malformed action is logged as a planner error, and three errors stop the run.

## 4. Drive it from the NVIDIA NeMo Agent Toolkit

```bash
pip install -e ".[nat]"                     # nvidia-nat[langchain]==1.9.0, Python 3.11-3.13
nat run --config_file examples/nat/offline_mock.yml --input "Calibrate the synthetic TFT."   # offline check
export NVIDIA_API_KEY=...                   # live run with a NIM-hosted model
nat run --config_file examples/nat/nim_react.yml --input "$(cat examples/nat/task.md)"
```

The plugin registers the tools `tcad_observe`, `tcad_update_group`, `tcad_activate_component` and
`tcad_stop` through the `nat.plugins` entry point.

## 5. Make any run reproducible

Every planner (rule, LLM, external, NAT) leaves `<out>/action_log.jsonl`. Replay it without the LLM:

```bash
python -m agentic_tcad_demo run --planner replay --replay-log <out>/action_log.jsonl --out outputs/replay
```

The replay must reproduce the J trajectory and held-out tables exactly. It is labelled `replay(...)`, so
never present it as a live LLM run.

## Honesty rules for reports

- All numbers are synthetic. Never compare them directly with the paper's J 2.589 → 0.181 (14.3×).
- Say which planner made the decisions. Say whether the run was blind, meaning whether the planner had
  seen the synthetic truth in `configs/synthetic_demo.yaml` or a previous trajectory.
- One run on one seed is a demonstration, not an evaluation.
