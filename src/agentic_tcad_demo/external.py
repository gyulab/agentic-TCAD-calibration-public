"""Planners that take decisions from outside the process, and their deterministic replay.

ExternalPlanner writes each observation to `<dir>/obs_NNN.json` and waits for `<dir>/act_NNN.json`.
Any agent or person can answer: a human reviewer, a coding agent in another session, or an LLM
without an API endpoint. The answer goes through the same parser and the same controller rules as the
LLM planner. The planner name passed in is recorded in the action log, so the source of every
decision stays visible.

ReplayPlanner re-issues the decisions recorded in an action log, so a run with an external or LLM
planner can be reproduced exactly without that planner. Its name is "replay(...)", never the original.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .controller import Action
from .llm import SYSTEM_PROMPT, parse_action


class ExternalPlanner:
    def __init__(self, exchange_dir: Path, name: str, poll_s: float = 1.0, timeout_s: float = 3600.0):
        self.dir, self.name = Path(exchange_dir), f"external({name})"
        self.poll_s, self.timeout_s, self.n = poll_s, timeout_s, 0
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "INSTRUCTIONS.md").write_text(
            SYSTEM_PROMPT + "\n\nFor observation obs_NNN.json, write the JSON answer to act_NNN.json.\n")

    def choose(self, observation: dict) -> Action:
        obs = self.dir / f"obs_{self.n:03d}.json"
        act = self.dir / f"act_{self.n:03d}.json"
        tmp = obs.with_suffix(".tmp")
        tmp.write_text(json.dumps(observation, indent=1, sort_keys=True, default=float))
        tmp.rename(obs)  # atomic: the reader never sees a half-written observation
        self.n += 1
        deadline = time.monotonic() + self.timeout_s
        while not act.exists():
            if time.monotonic() > deadline:
                return Action("invalid", None, f"no answer in {act.name} within {self.timeout_s:.0f} s")
            time.sleep(self.poll_s)
        time.sleep(0.2)  # let the writer finish
        try:
            return parse_action(act.read_text())
        except Exception as exc:
            return Action("invalid", None, f"external answer could not be used: {type(exc).__name__}: {exc}")


class ReplayPlanner:
    def __init__(self, action_log: Path):
        events = [json.loads(line) for line in Path(action_log).read_text().splitlines() if line.strip()]
        start = next(e for e in events if e["event"] == "run_start")
        self.source = start["planner"]
        self.name = f"replay({self.source})"
        self.queue = [e for e in events if e["event"] == "proposal"]
        self.i = 0

    def choose(self, observation: dict) -> Action:
        if self.i >= len(self.queue):
            return Action("invalid", None, "replay log exhausted before the run ended")
        rec = self.queue[self.i]
        self.i += 1
        if rec["cycle"] != observation["cycle"]:
            return Action("invalid", None, f"replay diverged: log cycle {rec['cycle']} vs run cycle {observation['cycle']}")
        return Action(rec["action"], rec["target"], rec["rationale"])
