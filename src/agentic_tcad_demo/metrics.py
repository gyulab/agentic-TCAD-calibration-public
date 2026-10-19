"""Device-metric extraction with the windows defined in the paper (arXiv:2609.12184, Sec. II-B).

  V_th : V_G where |I_D| first reaches 100 nA (linear interpolation in log10|I_D|)
  SS   : average inverse slope between 0.1 nA and 100 nA = dV_G / dlog10|I_D| over that window
  I_on : maximum |I_D| of the transfer curve
The "0.1" lower SS bound is read as 0.1 nA (the paper's text does not repeat the unit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VTH_CURRENT_A = 100e-9
SS_LOW_A = 0.1e-9
SS_HIGH_A = 100e-9
LOG_FLOOR_A = 1e-15  # numerical floor for log10 of simulated currents only


@dataclass(frozen=True)
class Metrics:
    vth: float  # V (nan if 100 nA is never reached)
    ss: float  # V/dec (nan if the window is not crossed)
    ion: float  # A

    def as_dict(self) -> dict[str, float]:
        return {"vth_V": self.vth, "ss_V_per_dec": self.ss, "ion_A": self.ion}


def crossing_voltage(vg: np.ndarray, current: np.ndarray, target_a: float) -> float:
    """First upward crossing of `target_a`, interpolated linearly in log10(current)."""
    log_i = np.log10(np.maximum(np.abs(current), LOG_FLOOR_A))
    log_t = np.log10(target_a)
    above = np.nonzero(log_i >= log_t)[0]
    if above.size == 0:
        return float("nan")
    k = int(above[0])
    if k == 0:
        return float("nan")  # already above target at the first point: crossing not observed
    frac = (log_t - log_i[k - 1]) / (log_i[k] - log_i[k - 1])
    return float(vg[k - 1] + frac * (vg[k] - vg[k - 1]))


def extract_metrics(vg: np.ndarray, current: np.ndarray) -> Metrics:
    vth = crossing_voltage(vg, current, VTH_CURRENT_A)
    v_lo = crossing_voltage(vg, current, SS_LOW_A)
    v_hi = crossing_voltage(vg, current, SS_HIGH_A)
    decades = np.log10(SS_HIGH_A / SS_LOW_A)
    ss = (v_hi - v_lo) / decades
    return Metrics(vth=vth, ss=float(ss), ion=float(np.max(np.abs(current))))
