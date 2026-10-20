# Recorded run: Claude Opus 5.5 as the planner

This folder records one calibration run of the default configuration (seed 2609). In this run, an LLM chose
every action instead of the rule planner. The LLM was Claude Opus 5.5 (`claude-opus-5-5`), working inside
a Claude Code session.

## How the run was made

The run used `--planner external`. At each decision the controller wrote the observation to
`obs_NNN.json`, the same JSON an API-based LLM planner receives. The model read the observation and
wrote one JSON action to `act_NNN.json`.

**No API call was made.** The decisions came from the interactive session, not from a scripted prompt, so
sampling settings do not apply.

**The controller enforced the same rules as for any planner:**
- legal actions only
- bounded Gauss–Newton steps
- accept only if the run converges and J decreases
- activation eligibility and an approval record

The planner chose actions only. All parameter values came from the OFAT and Gauss–Newton tools.

```bash
python -m agentic_tcad_demo run --planner external \
    --planner-name "claude-opus-5-5 via Claude Code session" --exchange-dir outputs/opus_exchange
```

## Reproduce it without the LLM

```bash
python -m agentic_tcad_demo run --planner replay \
    --replay-log examples/planner_runs/claude-opus-5-5/action_log.jsonl
```

The replay planner re-issues the recorded decisions in order. The planner is recorded as
`replay(external(...))`, so a replay is never presented as a live LLM run. The replay reproduces the
recorded J trajectory, final parameters, and held-out evaluation exactly
(`tests/test_planners.py::test_recorded_opus_run_replays_exactly`).

## Result compared with the rule planner (same seed, same config)

| | Rule planner | Claude Opus 5.5 planner |
|---|---|---|
| J initial → final | 20.77 → 0.407 | 20.77 → 0.391 |
| Accepted / rejected trials | 17 / 6 | 10 / 1 |
| Simulator calls (calibration) | 362 | 224 |
| Activations | `fixed_charge`, `tail_states` | `fixed_charge`, `tail_states` |
| Stop | parameter updates stalled | planner stop (remaining gains ≤ 0.006, judged model-form error) |
| Held-out drain bias, max \|ΔV<sub>th</sub>\| / \|Δlog I<sub>on</sub>\| | 16.9 mV / 0.024 dec | 14.6 mV / 0.027 dec |
| Held-out L = 2 µm, max \|ΔV<sub>th</sub>\| / \|Δlog I<sub>on</sub>\| | 17.0 mV / 0.021 dec | 12.5 mV / 0.040 dec |

The step-by-step decisions and their rationales are in [`decisions.md`](decisions.md).

The main differences in behaviour:
- **Earlier activation.** The LLM requested `fixed_charge` as soon as φ<sub>G</sub> was pinned at its bound.
  It argued that the dominant log-current error was mostly that rigid V<sub>th</sub> offset. The rule
  planner waited until parameter updates were exhausted.
- **Diagnosis-driven ordering.** It moved the contact barrier before mobility.
- **Avoiding null-space drift.** It used φ<sub>G</sub> alone near the end. Moving φ<sub>G</sub> and Q<sub>f</sub> together would only drift along their unidentifiable direction.
- **Explicit stop.** It stopped when the remaining predicted gains were below 2% of J.

## Caveats (read before citing these numbers)

- **Not a blind evaluation.** The same model wrote this repository. It had seen the synthetic truth
  parameters and the rule planner's trajectory before making these decisions. The run demonstrates that
  an LLM can drive the workflow through the same tools and guard rails. It does not show that an LLM
  outperforms the rule planner.
- **A single run on a single seed.** Held-out I<sub>on</sub> errors are slightly larger than the rule
  planner's, while V<sub>th</sub> errors are smaller.
- **Synthetic data only.** These numbers are unrelated to the paper's results, which used GPT-5.5 with
  measured data and Sentaurus.
