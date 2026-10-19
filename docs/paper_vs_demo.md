# Paper vs. public demo

The paper is G. Jeong *et al.*, arXiv:2609.12184v2. **Paper results** are published experimental/TCAD
results. **Demo results** are produced by this repository on synthetic data. The two must not be mixed.

## What the paper reports (not reproduced here)

- Calibration of a fabricated bottom-gate In–W–O transistor (2% W, W/L = 5/1 µm) with Synopsys Sentaurus.
- Five accepted agent-suggested updates reduce J from 2.589 to 0.181 (14.3×).
- With the calibrated model fixed:
  - drain-bias tests show maximum V<sub>th</sub>/I<sub>on</sub> errors of 36.1 mV / 0.022 dec
  - channel-length tests show 46.2 mV / 0.062 dec
- A 4% W device is recalibrated to give process insight.

These numbers depend on measured data, a Sentaurus device model, and normalization constants that are not
public. This repository cannot reproduce them and does not try to.

## What this repository supports

| Claim | Supported? | How |
|---|---|---|
| The residual → probe → bounded update → accept/reject loop runs end to end | Yes | `python -m agentic_tcad_demo run`, `action_log.jsonl` |
| Updates are accepted only if converged and J decreases | Yes | `tests/test_acceptance.py` |
| Failed (nonconverged) trials are recorded and excluded | Yes | `tests/test_acceptance.py` |
| A model term is added only when a systematic residual persists | Yes, under the stated thresholds. The candidate terms share the synthetic truth's functional forms, so the evidence is that they are *not* activated when the truth lacks them | approval evidence in the log, `test_activation_gate_is_selective`, `test_premature_activation_is_refused` |
| Held-out conditions are never used for calibration | Yes | `tests/test_workflow.py` (corrupting held-out data leaves calibration unchanged) |
| Transfer across V<sub>D</sub> and L on a model that depends on both | Yes, synthetic | `fig5_heldout_evaluation.png` |
| The paper's 14.3× reduction or its transfer errors | **No** | different data, solver, and J normalization |
| In–W–O device physics or the paper's parameter values | **No** | illustrative analytic model |
| Equivalence to Sentaurus | **No** | no device-equation solver |
| Behaviour of GPT-5.5 / Codex | **No** | the default planner is rule-based; the optional LLM planner is untested against any specific model |

## Specified by the paper vs. chosen here

| Item | Paper | Demo |
|---|---|---|
| J terms and weights | specified | identical |
| V<sub>th</sub>, SS, I<sub>on</sub> windows, 10 pA floor | specified | identical (SS lower bound read as 0.1 nA) |
| Normalization a<sub>i</sub> | "fixed before calibration", values not given | 0.1 dec, 0.1 I<sub>on</sub>, 0.1 V, 0.1 dec, 10 mV/dec |
| Bounds ℓ, u and trust radius Δ | not given | in `configs/synthetic_demo.yaml` |
| OFAT step, Eq. (3) solver | not given | 0.02 normalized; BVLS + SLSQP |
| Stopping rule | not given | see `docs/methodology.md` §4 |
| Splits | Table IV | same structure; held-out drain set made disjoint |
| Agent | GPT-5.5, Codex, NeMo Agent Toolkit, RAG, human reviewer | rule planner / optional LLM planner, approval record |
