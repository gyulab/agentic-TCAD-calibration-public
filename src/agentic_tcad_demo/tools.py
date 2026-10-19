"""Tools the calibration agent may call.

Both the deterministic controller and the optional LLM planner act only through this layer:
they can ask for a simulation, metric extraction, an OFAT ranking, a bounded Gauss-Newton trial,
or a model-component activation. Bounds, convergence checks and the accept-if-J-decreases rule
live here and in controller.py, never in the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .device import Condition, SimResult, Simulator
from .log import ActionLog
from .objective import CurveScore, ObjectiveSpec, mean_weighted_terms, objective, residual_vector, score_curve
from .parameters import ParameterGroup, ParameterSpec
from .sensitivity import ofat_jacobian
from .update import bounded_gn_step

# Machine-readable description of the planner-visible actions (also sent to the LLM).
ACTION_SCHEMA = {
    "update_group": "Run OFAT + bounded Gauss-Newton (Eq. 3) on one available parameter group and "
                    "accept the trial only if the simulation converges and J decreases.",
    "activate_component": "Request human/policy approval to enable one eligible model component, then "
                          "update its parameter group; kept only if J decreases.",
    "stop": "End calibration.",
}


@dataclass
class Evaluation:
    j: float
    scores: dict[Condition, CurveScore]
    curves: dict[Condition, np.ndarray]

    @property
    def weighted_terms(self) -> dict[str, float]:
        return mean_weighted_terms(list(self.scores.values()))


@dataclass
class GroupRanking:
    group: str
    names: tuple[str, ...]
    jacobian: np.ndarray
    r: np.ndarray
    weights: np.ndarray
    x: np.ndarray
    predicted_gain: float
    probes: list[dict] = field(default_factory=list)


class Workspace:
    def __init__(
        self,
        simulator: Simulator,
        measured: Mapping[Condition, np.ndarray],
        vg: np.ndarray,
        specs: Mapping[str, ParameterSpec],
        groups: Mapping[str, ParameterGroup],
        objective_spec: ObjectiveSpec,
        calibration: list[Condition],
        log: ActionLog,
        ofat_step: float,
    ):
        self.simulator = simulator
        self.measured = dict(measured)
        self.vg = vg
        self.specs = dict(specs)
        self.groups = dict(groups)
        self.objective_spec = objective_spec
        self.calibration = list(calibration)
        self.log = log
        self.ofat_step = ofat_step
        self.n_simulations = 0
        self.n_failed_simulations = 0

    # -- tool: simulate ------------------------------------------------------------------
    def simulate(self, params: Mapping[str, float], active: frozenset[str], cond: Condition) -> SimResult:
        self.n_simulations += 1
        res = self.simulator.simulate(params, active, cond, self.vg)
        if not res.converged and hasattr(self.simulator, "recovered"):
            # Predefined numerical-recovery step (paper Sec. II-C), tried once.
            self.n_simulations += 1
            res = self.simulator.recovered().simulate(params, active, cond, self.vg)
        if not res.converged:
            self.n_failed_simulations += 1
        return res

    def simulate_set(self, params, active, conds) -> dict[Condition, np.ndarray] | None:
        out = {}
        for cond in conds:
            res = self.simulate(params, active, cond)
            if not res.converged:
                return None
            out[cond] = res.id
        return out

    # -- tool: extract metrics / objective ------------------------------------------------
    def evaluate(self, params, active, conds=None) -> Evaluation | None:
        conds = self.calibration if conds is None else conds
        curves = self.simulate_set(params, active, conds)
        if curves is None:
            return None
        scores = {c: score_curve(self.vg, curves[c], self.measured[c], self.objective_spec) for c in conds}
        return Evaluation(j=objective(list(scores.values())), scores=scores, curves=curves)

    def residual(self, params, active) -> tuple[np.ndarray, np.ndarray] | None:
        curves = self.simulate_set(params, active, self.calibration)
        if curves is None:
            return None
        parts = [residual_vector(self.vg, curves[c], self.measured[c], self.objective_spec) for c in self.calibration]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def metric_rows(self, term: str) -> np.ndarray:
        """Indices of one metric residual (th, on, SS) inside the stacked residual vector."""
        idx, offset = [], 0
        pos = {"vth": 0, "ion": 1, "ss": 2}[term]
        for c in self.calibration:
            r, _ = residual_vector(self.vg, self.measured[c], self.measured[c], self.objective_spec)
            offset += r.size
            idx.append(offset - 3 + pos)
        return np.array(idx)

    # -- tool: OFAT ranking ----------------------------------------------------------------
    def rank_group(self, group: str, params, active, delta: float) -> GroupRanking | None:
        names = self.groups[group].parameters
        base = self.residual(params, active)
        if base is None:
            return None
        r0, w = base
        jac, probes = ofat_jacobian(lambda p: self.residual(p, active), params, names, self.specs, r0, self.ofat_step)
        x = np.array([self.specs[n].to_x(params[n]) for n in names])
        step = bounded_gn_step(r0, jac, w, x, delta)
        return GroupRanking(group, names, jac, r0, w, x, step.model_reduction, probes)

    # -- tool: bounded trial ---------------------------------------------------------------
    def propose(self, ranking: GroupRanking, params, delta: float) -> dict[str, float]:
        step = bounded_gn_step(ranking.r, ranking.jacobian, ranking.weights, ranking.x, delta)
        trial = dict(params)
        for name, xi, si in zip(ranking.names, ranking.x, step.step):
            trial[name] = self.specs[name].from_x(xi + si)
        return trial
