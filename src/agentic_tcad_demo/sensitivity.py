"""OFAT finite-difference sensitivity S_P = dr/dx_P (paper Eq. 2) in normalized coordinates."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .parameters import ParameterSpec

ResidualFn = Callable[[Mapping[str, float]], "tuple[np.ndarray, np.ndarray] | None"]


def ofat_jacobian(
    residual_fn: ResidualFn,
    params: Mapping[str, float],
    names: tuple[str, ...],
    specs: Mapping[str, ParameterSpec],
    r0: np.ndarray,
    step: float,
) -> tuple[np.ndarray, list[dict]]:
    """One-factor-at-a-time forward differences of the residual vector.

    Each parameter is moved by `step` in normalized units (backwards if that would leave [0, 1]).
    `residual_fn` returns None when the probe simulation does not converge; that column is set to
    zero and reported, so a failed probe can never create a spurious descent direction.
    """
    jac = np.zeros((r0.size, len(names)))
    probes = []
    for j, name in enumerate(names):
        spec = specs[name]
        x0 = spec.to_x(params[name])
        h = step if x0 + step <= 1.0 else -step
        probe = dict(params)
        probe[name] = spec.from_x(x0 + h)
        out = residual_fn(probe)
        if out is None:
            probes.append({"parameter": name, "dx": h, "converged": False})
            continue
        r1, _ = out
        jac[:, j] = (r1 - r0) / h
        probes.append({"parameter": name, "dx": h, "converged": True,
                       "probe_value": probe[name], "column_norm": float(np.linalg.norm(jac[:, j]))})
    return jac, probes
