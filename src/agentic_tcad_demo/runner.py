"""End-to-end demo run: synthesize data, calibrate, evaluate held-out conditions, write outputs."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import yaml

from . import __version__
from .config import DemoConfig
from .controller import Controller, Planner, Reviewer, RulePlanner
from .device import synthesize_measurement
from .evaluate import evaluate_fixed, summarize
from .log import ActionLog, _clean
from .tools import Workspace


def all_conditions(cfg: DemoConfig):
    seen = []
    for conds in cfg.splits.values():
        for c in conds:
            if c not in seen:
                seen.append(c)
    return seen


def run_demo(cfg: DemoConfig, out_dir: Path | None, planner: Planner | None = None,
             reviewer: Reviewer | None = None) -> dict:
    log = ActionLog(None if out_dir is None else out_dir / "action_log.jsonl")
    conds = all_conditions(cfg)
    measured = synthesize_measurement(cfg.truth, cfg.truth_params, cfg.truth_active, conds, cfg.vg,
                                      cfg.seed, cfg.noise["lognormal_sigma"], cfg.noise["floor_noise_A"])
    ws = Workspace(cfg.simulator, measured, cfg.vg, cfg.specs, cfg.groups, cfg.objective,
                   cfg.splits["calibration"], log, cfg.ofat_step)
    planner = planner or RulePlanner()
    reviewer = reviewer or Reviewer(cfg.review_mode, cfg.approved_components)
    log.record("run_start", version=__version__, planner=planner.name, seed=cfg.seed,
               calibration=[c.label for c in cfg.splits["calibration"]],
               heldout={k: [c.label for c in v] for k, v in cfg.splits.items() if k != "calibration"},
               note="synthetic data from an analytic compact model; not Sentaurus, not measured data")
    result = Controller(ws, cfg.components, planner, reviewer, cfg.controller, log).run(cfg.initial)

    n_sim_calibration = ws.n_simulations
    heldout = {}
    for split, split_conds in cfg.splits.items():
        heldout[split] = {
            "initial": evaluate_fixed(ws, cfg.initial, frozenset(), split_conds),
            "final": evaluate_fixed(ws, result.params, result.active, split_conds),
        }
        for which in ("initial", "final"):
            heldout[split][which]["summary"] = summarize(heldout[split][which]["rows"])
    log.record("evaluation", splits={k: {w: v[w]["summary"] for w in ("initial", "final")} for k, v in heldout.items()})

    summary = _clean({
        "description": "Synthetic-data demonstration; numbers are NOT the paper's results.",
        "planner": planner.name,
        "seed": cfg.seed,
        "J_initial": result.j_history[0],
        "J_final": result.j_history[-1],
        "J_reduction_factor": result.j_history[0] / result.j_history[-1],
        "J_history": result.j_history,
        "terms_history": result.term_history,
        "accepted_updates": [{"group": a["group"], "activation": a.get("activation"), "changes": a["changes"],
                              "J_after": a["J_trial"]} for a in result.accepted],
        "rejected_trials": len(result.rejected),
        "activations": result.activations,
        "stop_reason": result.stop_reason,
        "n_simulations_calibration": n_sim_calibration,
        "final_params": result.params,
        "final_active_components": sorted(result.active),
        "truth_params_for_reference": cfg.truth_params,
        "evaluation": {k: {w: v[w]["summary"] for w in ("initial", "final")} for k, v in heldout.items()},
        "evaluation_rows": {k: {w: v[w]["rows"] for w in ("initial", "final")} for k, v in heldout.items()},
        "final_params_hash": heldout["calibration"]["final"]["params_hash"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    })
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (out_dir / "final_params.yaml").write_text(yaml.safe_dump(
            {"parameters": _clean(result.params), "active_components": sorted(result.active)}, sort_keys=True))
    return {"summary": summary, "result": result, "workspace": ws, "measured": measured,
            "heldout": heldout, "log": log}
