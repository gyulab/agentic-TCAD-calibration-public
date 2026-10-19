# Reproducibility and security notes

## Environment
- Python ≥ 3.10. The reference runs used Python 3.11.
- Exact versions are pinned in `requirements-lock.txt`: numpy, scipy, matplotlib, PyYAML, pytest, and their dependencies.
- No GPU, no commercial simulator, no network access, and no API key are needed for the default path.
- If you have packages in a per-user site directory, run with `PYTHONNOUSERSITE=1`. This keeps them from shadowing the pinned ones.

## Determinism
- All synthetic data come from `numpy.random.SeedSequence(run.seed)`, with seed 2609 by default.
- Each condition gets its own child stream.
- The controller has no randomness.
- Two runs on the same platform produce byte-identical `summary.json` and `action_log.jsonl`. This is tested in `tests/test_workflow.py`.
- Across platforms or library versions, the last digits of floating-point results may differ.

## Runtime
On a laptop-class CPU the full run takes about 5 s, including figures. Calibration makes about 360
simulator calls, held-out evaluation makes a few more, and the test suite takes about 20–30 s.

## Outputs (`--out`, default `outputs/demo/`)

| File | Content |
|---|---|
| `action_log.jsonl` | One JSON event per line. Events: `run_start`, `baseline`, `diagnosis`, `ofat_ranking`, `proposal`, `trial`, `approval`, `activation`, `parameter_updates_exhausted`, `planner_error`, `stop`, `evaluation` |
| `summary.json` | J trajectory, weighted terms per step, accepted updates, rejected-trial count, activations, stop reason, `evaluation` tables (calibration, held-out drain, held-out length) for the initial and final models, final-parameter hash, versions |
| `final_params.yaml` | Final parameters and active components |
| `figures/*.png` | The five figures shown in the README |

## Security
- The default demo reads no credentials.
- The optional LLM planner reads its endpoint, model, and key from environment variables only.
- Never commit a filled-in `.env`. `.gitignore` excludes `.env`, `outputs/`, and local private directories.
- `tools/release_audit.py` scans tracked files and, with `--history`, every blob in git history. It looks for:
  - absolute home-directory paths (add site-specific paths through `--denylist`)
  - unexpected e-mail addresses
  - key-like strings
  - license-server strings
  - private IP ranges
  - simulator decks and outputs
  - large or binary files outside `docs/figures/`
- Maintainers can add a private, untracked denylist with `--denylist`.
