"""Bounded Gauss-Newton step of paper Eq. (3).

    s* = argmin_s 1/2 || W^1/2 (r + S s) ||^2   s.t.  0 <= x + s <= 1  and  ||s||_2 <= Delta

Coordinates are normalized, so the paper's box [l_P, u_P] becomes [0, 1]. The problem is a small
convex QP. We first solve the box-constrained least-squares problem exactly (scipy lsq_linear);
if that step is inside the trust ball it is the optimum. Otherwise SLSQP solves the problem with
the ball constraint added. The Jacobian comes from finite-difference simulator probes, which is
what makes the update derivative-free with respect to the simulator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize


@dataclass
class StepResult:
    step: np.ndarray
    model_reduction: float  # m(0) - m(s) of the Gauss-Newton model, >= 0
    ball_active: bool


def bounded_gn_step(r: np.ndarray, jac: np.ndarray, weights: np.ndarray, x: np.ndarray, delta: float) -> StepResult:
    sw = np.sqrt(weights)
    a = jac * sw[:, None]
    b = -r * sw
    lo, hi = -x, 1.0 - x
    m0 = 0.5 * float(b @ b)

    def model(s: np.ndarray) -> float:
        res = a @ s - b
        return 0.5 * float(res @ res)

    if not np.any(a):
        return StepResult(np.zeros_like(x), 0.0, False)

    box = lsq_linear(a, b, bounds=(lo, hi), method="bvls")
    s = box.x
    ball_active = False
    if np.linalg.norm(s) > delta:
        ball_active = True
        s0 = s * (delta / np.linalg.norm(s))
        s0 = np.clip(s0, lo, hi)
        with warnings.catch_warnings():  # older SciPy warns when SLSQP clips to bounds; we clip anyway
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(
                model, s0, jac=lambda v: a.T @ (a @ v - b), method="SLSQP",
                bounds=list(zip(lo, hi)),
                constraints=[{"type": "ineq", "fun": lambda v: delta**2 - v @ v, "jac": lambda v: -2.0 * v}],
                options={"ftol": 1e-12, "maxiter": 200},
            )
        s = np.clip(res.x, lo, hi) if res.success and model(np.clip(res.x, lo, hi)) <= model(s0) else s0
        if np.linalg.norm(s) > delta:  # tiny SLSQP constraint violation
            s *= delta / np.linalg.norm(s)
    return StepResult(step=s, model_reduction=max(m0 - model(s), 0.0), ball_active=ball_active)
