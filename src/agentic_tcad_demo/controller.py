"""Calibration controller: observe -> diagnose -> choose action -> trial -> accept/reject -> log.

The controller owns every rule that the paper places outside the agent:
  * only parameters in the chosen group move, inside their bounds and a trust radius (Eq. 3),
  * a trial is accepted only if every calibration simulation converges and J decreases,
  * nonconverged trials are recorded as numerical failures and excluded,
  * a model component may be activated only if its residual has persisted through bounded parameter
    adjustment (or parameter updates are exhausted), only if its trigger shows that the existing
    parameters cannot remove that residual, and only after an approval record.
A *planner* only chooses the next action from the enumerated, currently legal ones.
`RulePlanner` is a deterministic, inspectable policy; `llm.LLMPlanner` is the optional LLM one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np

from .log import ActionLog
from .objective import TERMS
from .parameters import params_hash
from .tools import Evaluation, GroupRanking, Workspace

# Rule table used for the logged diagnosis: which existing groups or model components are
# physically expected to address each dominant residual term.
DIAGNOSIS_RULES = {
    "vth": ("threshold-voltage offset", ["electrostatics", "fixed_charge"]),
    "ion": ("on-current magnitude mismatch", ["contact", "transport"]),
    "lin": ("above-threshold current mismatch", ["transport", "contact"]),
    "log": ("log-current shape/offset mismatch", ["contact", "electrostatics", "transport"]),
    "ss": ("subthreshold-swing mismatch", ["tail_states"]),
}


@dataclass(frozen=True)
class Component:
    name: str
    addresses: str  # objective term this component is meant to fix (th, SS, ...)
    trigger: str  # "insensitive": no active group can move that residual; "railed": controlling parameter at a bound
    initial: Mapping[str, float]
    controls: tuple[str, ...] = ()
    description: str = ""


@dataclass
class ControllerSettings:
    trust_radius: float = 0.3
    max_step_halvings: int = 2
    max_accepted_updates: int = 10
    min_relative_improvement: float = 0.01
    stall_patience: int = 2
    activation_min_share: float = 0.15
    persistence_cycles: int = 2
    insensitivity_ratio: float = 0.5
    rail_tolerance: float = 1e-3
    max_planner_errors: int = 3


@dataclass
class Action:
    kind: str  # update_group | activate_component | stop
    target: str | None = None
    rationale: str = ""


class Planner(Protocol):
    name: str

    def choose(self, observation: dict) -> Action: ...


class RulePlanner:
    """Deterministic policy: largest OFAT-predicted gain first; activation only when eligible."""

    name = "rule"

    def choose(self, observation: dict) -> Action:
        eligible = observation["eligible_components"]
        for comp in eligible:
            if comp["addresses"] == observation["dominant_term"]:
                return Action("activate_component", comp["name"],
                              f"dominant '{comp['addresses']}' residual persists after bounded updates: {comp['reason']}")
        candidates = [g for g in observation["groups"] if not g["tried"] and g["predicted_gain"] > 1e-12]
        if candidates:
            best = max(candidates, key=lambda g: (g["predicted_gain"], g["in_diagnosis"]))
            return Action("update_group", best["name"],
                          f"largest OFAT-predicted reduction of the Gauss-Newton model ({best['predicted_gain']:.3g})")
        if eligible:
            comp = eligible[0]
            return Action("activate_component", comp["name"],
                          f"no untried parameter group predicts a gain; persistent '{comp['addresses']}' residual: {comp['reason']}")
        if observation.get("parameter_updates_stalled"):
            return Action("stop", None, "parameter updates stalled (relative gain below threshold) and no "
                                        "model component is eligible")
        return Action("stop", None, "no untried parameter group predicts a gain and no model component is eligible")


class Reviewer:
    """Human-review gate for model activation: a policy file by default, or an interactive prompt."""

    def __init__(self, mode: str = "policy", approved: tuple[str, ...] = (), ask: Callable[[str], str] = input):
        self.mode, self.approved, self.ask = mode, set(approved), ask

    def review(self, component: Component, evidence: dict) -> tuple[bool, str]:
        if self.mode == "interactive":
            answer = self.ask(f"Approve activating '{component.name}' ({component.description})? "
                              f"Evidence: {evidence}. [y/N] ").strip().lower()
            return answer in {"y", "yes"}, "interactive reviewer"
        return component.name in self.approved, "policy file (review.approve_components)"


@dataclass
class CalibrationResult:
    params: dict[str, float]
    active: frozenset[str]
    j_history: list[float]
    term_history: list[dict[str, float]]
    accepted: list[dict]
    rejected: list[dict]
    stop_reason: str
    initial: Evaluation
    final: Evaluation
    n_simulations: int
    activations: list[str] = field(default_factory=list)


class Controller:
    def __init__(self, ws: Workspace, components: Mapping[str, Component], planner: Planner,
                 reviewer: Reviewer, settings: ControllerSettings, log: ActionLog):
        self.ws, self.components, self.planner = ws, dict(components), planner
        self.reviewer, self.s, self.log = reviewer, settings, log

    # ---------------------------------------------------------------------------------
    def _available_groups(self, active: frozenset[str]) -> list[str]:
        return [g for g, spec in self.ws.groups.items() if spec.component is None or spec.component in active]

    def _diagnose(self, ev: Evaluation) -> tuple[str, str, list[str]]:
        terms = ev.weighted_terms
        dominant = max(TERMS, key=lambda t: terms[t])
        label, groups = DIAGNOSIS_RULES[dominant]
        return dominant, label, groups

    def _eligible(self, ev: Evaluation, params, active, rankings: dict[str, GroupRanking], exhausted: bool) -> list[dict]:
        """Components whose activation the paper's rule allows right now.

        A component is eligible only if (1) the residual term it addresses carries at least
        `activation_min_share` of J, (2) that residual has persisted for `persistence_cycles` cycles of
        bounded parameter adjustment (or no parameter group predicts any gain), and (3) its trigger
        holds: no active group can move that residual ("insensitive"), or the parameter that normally
        controls it sits at an approved bound ("railed").
        """
        terms, out = ev.weighted_terms, []
        for comp in self.components.values():
            if comp.name in active or comp.name in self._rejected_components:
                continue
            share = terms[comp.addresses] / max(ev.j, 1e-30)
            if share < self.s.activation_min_share:
                continue
            history = self._share_history[comp.addresses][-self.s.persistence_cycles:]
            persisted = len(history) >= self.s.persistence_cycles and min(history) >= self.s.activation_min_share
            if not (persisted or exhausted):
                continue
            if comp.trigger == "insensitive":
                # Reach of a group = how much a full-range move (|dx| = 1) of its best parameter
                # could reduce this residual to first order: |S_rows^T r_rows| / |r_rows|.
                # Projecting onto the residual direction cancels grid-interpolation noise of
                # opposite sign at different biases.
                # Only fully converged OFAT rankings count: a failed probe (zero column) must not be
                # mistaken for evidence that a parameter cannot move the residual.
                complete = [rk for rk in rankings.values() if all(p["converged"] for p in rk.probes)]
                if not complete or len(complete) < len(rankings):
                    continue
                rows = self.ws.metric_rows(comp.addresses)
                r_rows = complete[0].r[rows]
                r_norm = float(np.linalg.norm(r_rows))
                if r_norm == 0.0:
                    continue
                reach = max(float(np.abs(rk.jacobian[rows].T @ r_rows).max()) / r_norm for rk in complete)
                ratio = reach / max(r_norm, 1e-30)
                if ratio < self.s.insensitivity_ratio:
                    out.append({"name": comp.name, "addresses": comp.addresses, "share_of_J": share,
                                "reason": f"max |dr_{comp.addresses}/dx| over active groups is "
                                          f"{ratio:.2f}x the residual (threshold {self.s.insensitivity_ratio})"})
            elif comp.trigger == "railed":
                railed = [n for n in comp.controls if self._pushed_into_bound(n, params, rankings)]
                if railed:
                    out.append({"name": comp.name, "addresses": comp.addresses, "share_of_J": share,
                                "reason": f"controlling parameter(s) {railed} at an approved bound, and the "
                                          f"residual gradient pushes further into that bound"})
        return out

    def _pushed_into_bound(self, name: str, params, rankings: dict[str, GroupRanking]) -> bool:
        """True if `name` sits at a bound and the Gauss-Newton descent direction points out of the box."""
        x = self.ws.specs[name].to_x(params[name])
        at_lower, at_upper = x <= self.s.rail_tolerance, x >= 1 - self.s.rail_tolerance
        if not (at_lower or at_upper):
            return False
        for rk in rankings.values():
            if name in rk.names:
                j = rk.names.index(name)
                grad = float(rk.jacobian[:, j] @ (rk.weights * rk.r))  # d(1/2 ||W^1/2 r||^2)/dx
                return (at_lower and grad > 0) or (at_upper and grad < 0)
        return False

    def _try_group(self, group: str, ranking: GroupRanking, params, active, ev: Evaluation, cycle: int):
        """Bounded trial with up to `max_step_halvings` trust-radius reductions. Returns new state or None."""
        delta = self.s.trust_radius
        for attempt in range(self.s.max_step_halvings + 1):
            trial = self.ws.propose(ranking, params, delta)
            changes = {n: {"from": params[n], "to": trial[n]} for n in ranking.names}
            bounds_ok = all(self.ws.specs[n].within_bounds(trial[n]) for n in ranking.names)
            new = self.ws.evaluate(trial, active) if bounds_ok else None
            if not bounds_ok:
                decision, reason = "rejected", "proposal outside approved bounds"
            elif new is None:
                decision, reason = "rejected", "numerical failure: simulation did not converge (excluded)"
            elif new.j < ev.j:
                decision, reason = "accepted", "converged and J decreased"
            else:
                decision, reason = "rejected", "converged but J did not decrease"
            record = {"cycle": cycle, "group": group, "attempt": attempt, "trust_radius": delta,
                      "changes": changes, "J_before": ev.j, "J_trial": None if new is None else new.j,
                      "decision": decision, "reason": reason}
            self.log.record("trial", **record)
            if decision == "accepted":
                return trial, new, record
            self._rejected.append(record)
            if new is None:
                break  # numerical failure or out-of-bounds proposal: do not keep probing this direction
            step_norm = float(np.linalg.norm([self.ws.specs[n].to_x(trial[n]) - self.ws.specs[n].to_x(params[n])
                                              for n in ranking.names]))
            if step_norm < 1e-9:
                break
            delta = 0.5 * min(delta, step_norm)  # trust-region shrink: never re-propose the same step
        return None

    # ---------------------------------------------------------------------------------
    def run(self, params: Mapping[str, float], active: frozenset[str] = frozenset()) -> CalibrationResult:
        params = dict(params)
        self._rejected: list[dict] = []
        self._rejected_components: set[str] = set()
        self._share_history: dict[str, list[float]] = {}
        accepted: list[dict] = []
        activations: list[str] = []
        ev = self.ws.evaluate(params, active)
        if ev is None:
            raise RuntimeError("initial model does not converge; fix the starting point first")
        initial = ev
        j_hist, term_hist = [ev.j], [ev.weighted_terms]
        self.log.record("baseline", J=ev.j, terms=ev.weighted_terms, params=params, active=sorted(active),
                        metrics={c.label: {"sim": s.sim.as_dict(), "meas": s.meas.as_dict()} for c, s in ev.scores.items()})
        stall, planner_errors, cycle, stop_reason = 0, 0, 0, None
        stalled = False  # parameter updates no longer give material progress

        while stop_reason is None:
            cycle += 1
            dominant, label, prior = self._diagnose(ev)
            for t, v in ev.weighted_terms.items():
                self._share_history.setdefault(t, []).append(v / max(ev.j, 1e-30))
            self.log.record("diagnosis", cycle=cycle, dominant_term=dominant, interpretation=label,
                            rule_candidates=prior, terms=ev.weighted_terms)
            rankings = {}
            for g in self._available_groups(active):
                rk = self.ws.rank_group(g, params, active, self.s.trust_radius)
                if rk is not None:
                    rankings[g] = rk
            self.log.record("ofat_ranking", cycle=cycle, ranking=[
                {"group": g, "parameters": list(rk.names), "predicted_gain": rk.predicted_gain, "probes": rk.probes}
                for g, rk in sorted(rankings.items(), key=lambda kv: -kv[1].predicted_gain)])
            tried: set[str] = set()
            progressed = False
            while not progressed and stop_reason is None:
                exhausted = stalled or all(g in tried or rk.predicted_gain <= 1e-12 for g, rk in rankings.items())
                observation = {
                    "cycle": cycle, "J": ev.j, "terms": ev.weighted_terms, "dominant_term": dominant,
                    "diagnosis": label, "active_components": sorted(active),
                    "parameters": {k: params[k] for k in sorted(params)},
                    "groups": [{"name": g, "parameters": list(rk.names), "predicted_gain": rk.predicted_gain,
                                "tried": g in tried or stalled, "in_diagnosis": g in prior} for g, rk in rankings.items()],
                    "parameter_updates_exhausted": exhausted,
                    "parameter_updates_stalled": stalled,
                    "eligible_components": self._eligible(ev, params, active, rankings, exhausted),
                    "legal_actions": list(("update_group", "activate_component", "stop")),
                }
                action = self.planner.choose(observation)
                self.log.record("proposal", cycle=cycle, planner=self.planner.name, action=action.kind,
                                target=action.target, rationale=action.rationale)
                legal_groups = set() if stalled else {g for g in rankings if g not in tried}
                legal_comps = {c["name"] for c in observation["eligible_components"]}
                if action.kind == "stop":
                    stop_reason = f"planner stop: {action.rationale}"
                elif action.kind == "update_group" and action.target in legal_groups:
                    tried.add(action.target)
                    out = self._try_group(action.target, rankings[action.target], params, active, ev, cycle)
                    if out is not None:
                        params, new, record = out
                        rel = (ev.j - new.j) / ev.j
                        ev = new
                        accepted.append(record)
                        j_hist.append(ev.j)
                        term_hist.append(ev.weighted_terms)
                        progressed = True
                        stall = stall + 1 if rel < self.s.min_relative_improvement else 0
                elif action.kind == "activate_component" and action.target in legal_comps:
                    comp = self.components[action.target]
                    evidence = next(c for c in observation["eligible_components"] if c["name"] == comp.name)
                    approved, reviewer = self.reviewer.review(comp, evidence)
                    self.log.record("approval", cycle=cycle, component=comp.name, approved=approved,
                                    reviewer=reviewer, evidence=evidence)
                    if not approved:
                        self._rejected_components.add(comp.name)
                        continue
                    trial_active = active | {comp.name}
                    trial_params = {**params, **comp.initial}
                    group = next(g for g, spec in self.ws.groups.items() if spec.component == comp.name)
                    rk = self.ws.rank_group(group, trial_params, trial_active, self.s.trust_radius)
                    out = None if rk is None else self._try_group(group, rk, trial_params, trial_active, ev, cycle)
                    if out is None:
                        self._rejected_components.add(comp.name)
                        self.log.record("activation", cycle=cycle, component=comp.name, kept=False,
                                        reason="no converged trial with lower J after activation")
                        continue
                    params, new, record = out
                    active = trial_active
                    activations.append(comp.name)
                    self.log.record("activation", cycle=cycle, component=comp.name, kept=True,
                                    initial_values=dict(comp.initial), J_before=ev.j, J_after=new.j)
                    rel = (ev.j - new.j) / ev.j
                    ev = new
                    accepted.append({**record, "activation": comp.name})
                    stalled, stall = False, -1  # new model class: parameter updates are legal again
                    j_hist.append(ev.j)
                    term_hist.append(ev.weighted_terms)
                    progressed = True
                    stall = stall + 1 if rel < self.s.min_relative_improvement else 0
                else:
                    planner_errors += 1
                    self.log.record("planner_error", cycle=cycle, action=action.kind, target=action.target,
                                    reason="action not legal in the current state")
                    if planner_errors >= self.s.max_planner_errors:
                        stop_reason = "planner produced too many illegal actions"
                if progressed:
                    if len(accepted) >= self.s.max_accepted_updates:
                        stop_reason = f"reached max_accepted_updates={self.s.max_accepted_updates}"
                    elif stall >= self.s.stall_patience and not stalled:
                        # Bounded parameter adjustment has stopped paying off: from now on only a
                        # model activation (if eligible) or a stop is legal (paper Sec. II-C).
                        stalled = True
                        self.log.record("parameter_updates_exhausted", cycle=cycle, reason=(
                            f"relative J improvement < {self.s.min_relative_improvement:g} for "
                            f"{self.s.stall_patience} accepted updates in a row"))

        self.log.record("stop", reason=stop_reason, J=ev.j, accepted_updates=len(accepted),
                        rejected_trials=len(self._rejected), n_simulations=self.ws.n_simulations,
                        params=params, active=sorted(active), params_hash=params_hash(params, active))
        return CalibrationResult(params=params, active=active, j_history=j_hist, term_history=term_hist,
                                 accepted=accepted, rejected=list(self._rejected), stop_reason=stop_reason,
                                 initial=initial, final=ev, n_simulations=self.ws.n_simulations,
                                 activations=activations)
