"""Step-wise calibration session for agent frameworks that call tools one at a time.

Frameworks such as the NVIDIA NeMo Agent Toolkit drive a workflow by calling tools. They cannot pass a
callback into `Controller.run`. A `CalibrationSession` runs the unchanged controller in a background
thread with a queue-backed planner. `observe()` returns the current observation, and `act()` submits one
action and returns its outcome together with the next observation. Bounds, the accept rule, activation
eligibility, approval and logging all stay inside the controller.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from .config import load_config
from .controller import Action
from .log import ActionLog
from .runner import run_demo

_DONE = object()


class QueuePlanner:
    def __init__(self, name: str):
        self.name = name
        self.observations: queue.Queue = queue.Queue()
        self.actions: queue.Queue = queue.Queue()

    def choose(self, observation: dict) -> Action:
        self.observations.put(observation)
        return self.actions.get()


def compact_observation(obs: dict) -> dict:
    """The observation an agent needs, rounded so it fits comfortably in an LLM context."""
    r = lambda v: float(f"{v:.4g}")  # noqa: E731
    return {
        "cycle": obs["cycle"],
        "J": r(obs["J"]),
        "weighted_terms": {k: r(v) for k, v in obs["terms"].items()},
        "dominant_term": obs["dominant_term"],
        "diagnosis": obs["diagnosis"],
        "parameters": {k: r(v) for k, v in obs["parameters"].items()},
        "active_components": obs["active_components"],
        "groups": [{"name": g["name"], "parameters": g["parameters"], "predicted_gain": r(g["predicted_gain"]),
                    "tried_this_cycle": g["tried"]} for g in obs["groups"]],
        "eligible_components": [{"name": c["name"], "addresses": c["addresses"], "reason": c["reason"]}
                                for c in obs["eligible_components"]],
        "parameter_updates_stalled": obs["parameter_updates_stalled"],
        "legal_actions": ["update_group(group, rationale)", "activate_component(component, rationale)",
                          "stop(rationale)"],
    }


class CalibrationSession:
    def __init__(self, config_file: str | Path, out_dir: str | Path | None, planner_name: str):
        self.cfg = load_config(config_file)
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.planner = QueuePlanner(planner_name)
        self.log = ActionLog(None if self.out_dir is None else self.out_dir / "action_log.jsonl")
        self.result: dict | None = None
        self.error: BaseException | None = None
        self._n_events = 0
        self._current: dict | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._current = self._next()

    def _run(self) -> None:
        try:
            self.result = run_demo(self.cfg, self.out_dir, planner=self.planner, log=self.log)
        except BaseException as exc:  # surfaced to the caller by _next()
            self.error = exc
        finally:
            self.planner.observations.put(_DONE)

    def _next(self) -> dict | None:
        item = self.planner.observations.get()
        if item is _DONE:
            self._thread.join()
            if self.error is not None:
                raise RuntimeError(f"calibration failed: {self.error}") from self.error
            return None
        return item

    @property
    def finished(self) -> bool:
        return self._current is None

    def observe(self) -> dict:
        if self.finished:
            return {"finished": True, "summary": self.summary()}
        return {"finished": False, "observation": compact_observation(self._current)}

    def act(self, kind: str, target: str | None, rationale: str) -> dict:
        if self.finished:
            return {"finished": True, "note": "calibration already finished", "summary": self.summary()}
        start = len(self.log.events)
        self.planner.actions.put(Action(kind, target, rationale))
        self._current = self._next()
        outcome = self._outcome_since(start)
        if self.finished:
            return {"finished": True, "outcome": outcome, "summary": self.summary()}
        return {"finished": False, "outcome": outcome, "observation": compact_observation(self._current)}

    # -- helpers ----------------------------------------------------------------------------
    def _outcome_since(self, start: int) -> list[dict]:
        out = []
        for e in self.log.events[start:]:
            if e["event"] == "trial":
                out.append({"group": e["group"], "decision": e["decision"], "reason": e["reason"],
                            "J_before": e["J_before"], "J_trial": e["J_trial"],
                            "changes": {k: v["to"] for k, v in e["changes"].items()}})
            elif e["event"] in ("approval", "activation", "planner_error", "parameter_updates_exhausted"):
                out.append({k: v for k, v in e.items() if k not in ("seq", "evidence")})
        return out

    def summary(self) -> dict:
        if self.result is None:
            return {}
        s = self.result["summary"]
        return {"planner": s["planner"], "J_initial": s["J_initial"], "J_final": s["J_final"],
                "accepted_updates": len(s["accepted_updates"]), "rejected_trials": s["rejected_trials"],
                "activations": s["activations"], "stop_reason": s["stop_reason"],
                "heldout_max_abs_dVth_mV": {k: v["final"].get("max_abs_dVth_mV") for k, v in s["evaluation"].items()},
                "outputs": None if self.out_dir is None else str(self.out_dir)}

    def close(self, reason: str = "agent session ended without an explicit stop") -> None:
        while not self.finished:
            self.act("stop", None, reason)
