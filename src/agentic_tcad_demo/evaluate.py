"""Fixed-model evaluation on held-out conditions (no retuning)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .device import Condition
from .objective import score_curve
from .parameters import params_hash
from .tools import Workspace


def evaluate_fixed(ws: Workspace, params: Mapping[str, float], active: frozenset[str],
                   conditions: list[Condition]) -> dict:
    """Simulate and score a frozen parameter set. Only the simulate/score tools are used."""
    frozen = dict(params)
    rows = []
    for cond in conditions:
        res = ws.simulate(frozen, active, cond)
        if not res.converged:
            rows.append({"condition": cond.label, "vd": cond.vd, "length_um": cond.length_um, "converged": False})
            continue
        s = score_curve(ws.vg, res.id, ws.measured[cond], ws.objective_spec)
        rows.append({
            "condition": cond.label, "vd": cond.vd, "length_um": cond.length_um, "converged": True,
            "dVth_mV": 1e3 * (s.sim.vth - s.meas.vth),
            "dlogIon_dec": float(np.log10(s.sim.ion) - np.log10(s.meas.ion)),
            "dSS_mV_per_dec": 1e3 * (s.sim.ss - s.meas.ss),
            "J_condition": s.j,
        })
    # No update tool is reachable from here; tests check that exactly one simulation per condition runs.
    return {"params_hash": params_hash(frozen, active), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["converged"]]
    if not ok:
        return {"n": 0}
    vth = np.abs([r["dVth_mV"] for r in ok])
    ion = np.abs([r["dlogIon_dec"] for r in ok])
    return {"n": len(ok), "mean_abs_dVth_mV": float(vth.mean()), "max_abs_dVth_mV": float(vth.max()),
            "mean_abs_dlogIon_dec": float(ion.mean()), "max_abs_dlogIon_dec": float(ion.max())}
