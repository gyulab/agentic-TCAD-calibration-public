"""Multi-metric calibration objective J (paper Eq. 1) and the normalized residual vector (Eq. 2).

Eq. (1):  J = 1/|B| * sum_b [ w_log E_log,b + w_lin E_lin,b + w_th |dVth_b| + w_on |dlog10 Ion_b| + w_SS |dSS_b| ]

The weights are the paper's. The paper states that each term is normalized by a fixed factor
a_i chosen before calibration, but it does not list the a_i values; the values used here are
demo choices declared in the config (`objective.normalization`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import LOG_FLOOR_A, Metrics, extract_metrics

TERMS = ("log", "lin", "vth", "ion", "ss")
PAPER_WEIGHTS = {"log": 0.25, "lin": 0.10, "vth": 1.00, "ion": 0.20, "ss": 0.25}
UNDEFINED_METRIC_PENALTY = 10.0  # in normalized units, used when a metric cannot be extracted


@dataclass(frozen=True)
class ObjectiveSpec:
    weights: dict[str, float] = field(default_factory=lambda: dict(PAPER_WEIGHTS))
    # a_i: log [dec], lin [fraction of measured I_on], th [V], on [dec], SS [V/dec]
    normalization: dict[str, float] = field(
        default_factory=lambda: {"log": 0.1, "lin": 0.1, "vth": 0.1, "ion": 0.1, "ss": 0.01}
    )
    measurement_floor_a: float = 10e-12
    lin_window_decades: float = 1.0


@dataclass
class CurveScore:
    """Per-bias contribution to J plus the signed metric differences behind it."""

    terms: dict[str, float]  # normalized, unweighted |error| per term
    weighted: dict[str, float]  # w_i * terms[i]
    signed: dict[str, float]  # signed normalized metric residuals (sim - meas) / a_i
    sim: Metrics
    meas: Metrics

    @property
    def j(self) -> float:
        return float(sum(self.weighted.values()))


def _masks(meas: np.ndarray, spec: ObjectiveSpec) -> tuple[np.ndarray, np.ndarray]:
    log_mask = meas >= spec.measurement_floor_a
    lin_mask = meas >= np.max(meas) / 10.0**spec.lin_window_decades
    return log_mask, lin_mask


def _metric_residual(sim_val: float, meas_val: float, scale: float) -> float:
    if not (np.isfinite(sim_val) and np.isfinite(meas_val)):
        return UNDEFINED_METRIC_PENALTY
    return (sim_val - meas_val) / scale


def score_curve(vg: np.ndarray, sim: np.ndarray, meas: np.ndarray, spec: ObjectiveSpec) -> CurveScore:
    a = spec.normalization
    m_sim = extract_metrics(vg, sim)
    m_meas = extract_metrics(vg, meas)
    log_mask, lin_mask = _masks(meas, spec)
    log_err = np.log10(np.maximum(sim[log_mask], LOG_FLOOR_A)) - np.log10(meas[log_mask])
    lin_err = (sim[lin_mask] - meas[lin_mask]) / m_meas.ion
    signed = {
        "vth": _metric_residual(m_sim.vth, m_meas.vth, a["vth"]),
        "ion": _metric_residual(np.log10(m_sim.ion), np.log10(m_meas.ion), a["ion"]),
        "ss": _metric_residual(m_sim.ss, m_meas.ss, a["ss"]),
    }
    terms = {
        "log": float(np.sqrt(np.mean(log_err**2))) / a["log"],
        "lin": float(np.sqrt(np.mean(lin_err**2))) / a["lin"],
        "vth": abs(signed["vth"]),
        "ion": abs(signed["ion"]),
        "ss": abs(signed["ss"]),
    }
    weighted = {k: spec.weights[k] * terms[k] for k in TERMS}
    return CurveScore(terms=terms, weighted=weighted, signed=signed, sim=m_sim, meas=m_meas)


def objective(scores: list[CurveScore]) -> float:
    """Eq. (1): mean over calibration biases of the weighted per-bias sum."""
    return float(np.mean([s.j for s in scores]))


def mean_weighted_terms(scores: list[CurveScore]) -> dict[str, float]:
    return {k: float(np.mean([s.weighted[k] for s in scores])) for k in TERMS}


def residual_vector(vg: np.ndarray, sim: np.ndarray, meas: np.ndarray, spec: ObjectiveSpec) -> tuple[np.ndarray, np.ndarray]:
    """Normalized signed residual vector r (Eq. 2) and diagonal weights W for one bias.

    r stacks pointwise log-current residuals, pointwise linear-current residuals, and the three
    metric residuals. The pointwise blocks are weighted w/N so that their squared sum mirrors the
    RMSE terms of J. Eq. (3) minimizes a weighted squared norm, so it is a smooth surrogate of the
    absolute-value terms in Eq. (1); acceptance is still decided on J itself.
    """
    a, w = spec.normalization, spec.weights
    log_mask, lin_mask = _masks(meas, spec)
    ion_meas = float(np.max(meas))
    r_log = (np.log10(np.maximum(sim[log_mask], LOG_FLOOR_A)) - np.log10(meas[log_mask])) / a["log"]
    r_lin = (sim[lin_mask] - meas[lin_mask]) / ion_meas / a["lin"]
    s = score_curve(vg, sim, meas, spec).signed
    r_met = np.array([s["vth"], s["ion"], s["ss"]])
    r = np.concatenate([r_log, r_lin, r_met])
    weights = np.concatenate([
        np.full(r_log.size, w["log"] / max(r_log.size, 1)),
        np.full(r_lin.size, w["lin"] / max(r_lin.size, 1)),
        np.array([w["vth"], w["ion"], w["ss"]]),
    ])
    return r, weights
