"""Educational compact model of a bottom-gate oxide-semiconductor TFT.

This is a closed-form, charge-based transfer-curve model, NOT a TCAD solver.
It replaces the Sentaurus device simulation of the paper so that the calibration
workflow can run anywhere. See docs/methodology.md for the equations and limits.

Physics included (all phenomenological):
  * threshold voltage set by the gate work function and optional fixed charge,
  * subthreshold ideality n from a fixed interface term plus optional tail states,
  * EKV-style smooth interpolation from subthreshold to above threshold,
    with drain-bias and channel-length dependence,
  * a Schottky-like source/drain contact resistance solved self-consistently,
  * optional gate-dependent (percolation-like) mobility and an optional drain-induced
    threshold shift (proportional to V_D / L). Both are used only by the synthetic "truth"
    device, so the calibration model is structurally imperfect and the held-out drain-bias
    and channel-length tests can expose model-form error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

Q = 1.602176634e-19  # C
EPS0 = 8.8541878128e-14  # F/cm
KB_T_Q_300K = 0.025852  # V


@dataclass(frozen=True)
class Condition:
    """One measured/simulated transfer curve: drain bias and channel length."""

    vd: float  # V
    length_um: float  # um

    @property
    def label(self) -> str:
        return f"L={self.length_um:g}um,VD={self.vd:g}V"


@dataclass
class SimResult:
    vg: np.ndarray
    id: np.ndarray
    converged: bool
    message: str = ""


class Simulator(Protocol):
    """Interface every simulator (synthetic or private plug-in) must satisfy."""

    def simulate(
        self, params: Mapping[str, float], active: frozenset[str], cond: Condition, vg: np.ndarray
    ) -> SimResult: ...


@dataclass
class DeviceConstants:
    """Fixed device description. Geometry follows the paper's Table III; the rest is illustrative."""

    width_um: float = 5.0
    t_ch_nm: float = 3.5
    t_ox_nm: float = 7.0
    eps_ox: float = 20.0  # illustrative high-k permittivity
    chi_ch_ev: float = 4.30  # illustrative channel electron affinity
    n_interface: float = 0.10  # C_it0 / C_ox, fixed interface contribution to ideality
    rho_c0_ohm_cm: float = 0.20  # contact prefactor, width-normalized
    contact_ideality: float = 4.0  # m in exp(phi_B / (m kT/q))
    max_barrier_ev: float = 0.80  # validity limit of the contact expression
    temperature_k: float = 300.0

    @property
    def c_ox(self) -> float:  # F/cm^2
        return EPS0 * self.eps_ox / (self.t_ox_nm * 1e-7)

    @property
    def v_t(self) -> float:
        return KB_T_Q_300K * self.temperature_k / 300.0


@dataclass
class CompactTFT:
    """Compact TFT simulator.

    Parameters are passed by name:
      phi_SD  [eV]      effective source/drain work function (contact barrier)
      phi_G   [eV]      effective gate work function
      mu0     [cm2/Vs]  band mobility prefactor
      Q_f     [cm^-2]   gate-interface fixed charge          (component "fixed_charge")
      N_tail  [cm^-3/eV] effective tail-state density        (component "tail_states")

    `mobility_gamma` > 0 enables a gate-dependent mobility mu0*(1+V_ov/1V)^gamma.
    `dibl_v_per_v_um` > 0 lowers V_T by dibl * V_D * (1 um / L).
    The calibration model keeps both at 0; the synthetic truth device does not.
    """

    const: DeviceConstants = field(default_factory=DeviceConstants)
    mobility_gamma: float = 0.0
    dibl_v_per_v_um: float = 0.0
    bisection_iters: int = 60
    rel_tol: float = 1e-6

    def threshold_voltage(self, params: Mapping[str, float], active: frozenset[str]) -> float:
        c = self.const
        vt0 = params["phi_G"] - c.chi_ch_ev
        if "fixed_charge" in active:
            vt0 -= Q * params["Q_f"] / c.c_ox
        return vt0

    def ideality(self, params: Mapping[str, float], active: frozenset[str]) -> float:
        c = self.const
        n = 1.0 + c.n_interface
        if "tail_states" in active:
            c_tail = Q * params["N_tail"] * c.t_ch_nm * 1e-7  # F/cm^2 (N in cm^-3 V^-1)
            n += c_tail / c.c_ox
        return n

    def contact_resistance(self, params: Mapping[str, float]) -> float:
        """Total (source + drain) contact resistance in ohm."""
        c = self.const
        smooth = 0.02
        barrier = smooth * np.logaddexp(0.0, (params["phi_SD"] - c.chi_ch_ev) / smooth)
        rc_w = c.rho_c0_ohm_cm * np.exp(barrier / (c.contact_ideality * c.v_t))
        return float(2.0 * rc_w / (c.width_um * 1e-4))

    def _intrinsic(self, vgs, vds, vt, n, mu0, length_um):
        c = self.const
        two_n_vt = 2.0 * n * c.v_t
        f_s = np.logaddexp(0.0, (vgs - vt) / two_n_vt) ** 2
        f_d = np.logaddexp(0.0, (vgs - vt - n * vds) / two_n_vt) ** 2
        mu = mu0
        if self.mobility_gamma > 0.0:
            v_ov = two_n_vt * np.logaddexp(0.0, (vgs - vt) / two_n_vt)
            mu = mu0 * (1.0 + v_ov / 1.0) ** self.mobility_gamma
        beta = mu * c.c_ox * (c.width_um / length_um)
        return 2.0 * n * beta * c.v_t**2 * (f_s - f_d)

    def recovered(self) -> "CompactTFT":
        """Numerical-recovery variant (more solver iterations), tried once after a failure."""
        return CompactTFT(self.const, self.mobility_gamma, self.dibl_v_per_v_um, 2 * self.bisection_iters, self.rel_tol)

    def simulate(
        self, params: Mapping[str, float], active: frozenset[str], cond: Condition, vg: np.ndarray
    ) -> SimResult:
        vg = np.asarray(vg, dtype=float)
        c = self.const
        if not np.isfinite(list(params.values())).all() or params["mu0"] <= 0:
            return SimResult(vg, np.full_like(vg, np.nan), False, "invalid parameter values")
        if params["phi_SD"] - c.chi_ch_ev > c.max_barrier_ev:
            return SimResult(vg, np.full_like(vg, np.nan), False, "contact barrier outside model validity")

        vt = self.threshold_voltage(params, active) - self.dibl_v_per_v_um * cond.vd / cond.length_um
        n = self.ideality(params, active)
        r_c = self.contact_resistance(params)
        mu0 = params["mu0"]

        # Self-consistent series resistance: I = I_int(VG - I*Rc/2, VD - I*Rc).
        # f(I) = I - I_int(...) is increasing in I, and I_int at I=0 bounds the root.
        hi = self._intrinsic(vg, cond.vd, vt, n, mu0, cond.length_um)
        lo = np.zeros_like(hi)
        for _ in range(self.bisection_iters):
            mid = 0.5 * (lo + hi)
            f = mid - self._intrinsic(vg - 0.5 * mid * r_c, cond.vd - mid * r_c, vt, n, mu0, cond.length_um)
            pos = f > 0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
        current = 0.5 * (lo + hi)
        resid = np.abs(current - self._intrinsic(vg - 0.5 * current * r_c, cond.vd - current * r_c, vt, n, mu0, cond.length_um))
        ok = bool(np.all(np.isfinite(current)) and np.all(resid <= self.rel_tol * np.maximum(current, 1e-30) + 1e-30))
        return SimResult(vg, current, ok, "" if ok else "self-consistent contact solve did not converge")


def synthesize_measurement(
    truth: CompactTFT,
    params: Mapping[str, float],
    active: frozenset[str],
    conditions: list[Condition],
    vg: np.ndarray,
    seed: int,
    lognormal_sigma: float,
    floor_noise_a: float,
) -> dict[Condition, np.ndarray]:
    """Synthetic "measured" curves: truth device + multiplicative noise + an additive noise floor.

    Each condition gets its own child stream of one SeedSequence, so results do not depend on
    the order in which conditions are generated.
    """
    children = np.random.SeedSequence(seed).spawn(len(conditions))
    out: dict[Condition, np.ndarray] = {}
    for cond, child in zip(conditions, children):
        rng = np.random.default_rng(child)
        res = truth.simulate(params, active, cond, vg)
        if not res.converged:
            raise RuntimeError(f"truth device failed for {cond.label}: {res.message}")
        noisy = res.id * np.exp(lognormal_sigma * rng.standard_normal(vg.size))
        noisy = np.abs(noisy + floor_noise_a * rng.standard_normal(vg.size))
        out[cond] = noisy
    return out
