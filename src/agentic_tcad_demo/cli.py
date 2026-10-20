"""Command-line entry point: python -m agentic_tcad_demo run --config configs/synthetic_demo.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .controller import Reviewer, RulePlanner
from .runner import run_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_tcad_demo",
                                     description="Synthetic-data demonstration of an agentic TCAD calibration loop.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate synthetic data, calibrate, evaluate held-out conditions")
    run.add_argument("--config", default="configs/synthetic_demo.yaml")
    run.add_argument("--out", default="outputs/demo", help="output directory (log, summary, figures)")
    run.add_argument("--figures-dir", default=None, help="where to write PNGs (default: <out>/figures)")
    run.add_argument("--planner", choices=("rule", "llm", "external", "replay"), default="rule",
                     help="rule = deterministic offline controller (default); llm = OpenAI-compatible LLM; "
                          "external = decisions written to files by another agent or a person; "
                          "replay = re-issue the decisions recorded in an action log")
    run.add_argument("--exchange-dir", default="outputs/exchange", help="file exchange for --planner external")
    run.add_argument("--planner-name", default="unnamed", help="who answers for --planner external (recorded)")
    run.add_argument("--replay-log", help="action_log.jsonl to replay for --planner replay")
    run.add_argument("--interactive", action="store_true", help="ask on the terminal before each model activation")
    run.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.planner == "llm":
        from .llm import LLMConfigError, LLMPlanner
        try:
            planner = LLMPlanner.from_env()
        except LLMConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.planner == "external":
        from .external import ExternalPlanner
        planner = ExternalPlanner(Path(args.exchange_dir), args.planner_name)
    elif args.planner == "replay":
        from .external import ReplayPlanner
        if not args.replay_log:
            print("error: --planner replay needs --replay-log", file=sys.stderr)
            return 2
        planner = ReplayPlanner(Path(args.replay_log))
    else:
        planner = RulePlanner()
    reviewer = Reviewer("interactive" if args.interactive else cfg.review_mode, cfg.approved_components)

    out_dir = Path(args.out)
    out = run_demo(cfg, out_dir, planner=planner, reviewer=reviewer)
    s = out["summary"]
    if not args.no_figures:
        from .plots import write_figures
        write_figures(out, Path(args.figures_dir) if args.figures_dir else out_dir / "figures")

    print(f"planner            : {s['planner']}")
    print(f"J initial -> final : {s['J_initial']:.4g} -> {s['J_final']:.4g}  (x{s['J_reduction_factor']:.3g}, synthetic data)")
    print(f"accepted / rejected: {len(s['accepted_updates'])} / {s['rejected_trials']}")
    print(f"activations        : {', '.join(s['activations']) or 'none'}")
    print(f"stop reason        : {s['stop_reason']}")
    print(f"simulator calls    : {s['n_simulations_calibration']} (calibration)")
    for split, v in s["evaluation"].items():
        f = v["final"]
        print(f"{split:<19}: max|dVth| {f['max_abs_dVth_mV']:.3g} mV, max|dlogIon| {f['max_abs_dlogIon_dec']:.3g} dec "
              f"(initial model: {v['initial']['max_abs_dVth_mV']:.3g} mV, {v['initial']['max_abs_dlogIon_dec']:.3g} dec)")
    print(f"outputs            : {out_dir}/action_log.jsonl, summary.json, final_params.yaml")
    return 0
