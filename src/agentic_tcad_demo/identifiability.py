"""Identifiability check: an optional step between the held-out evaluation and the interpretation of the
calibrated parameters (`run --identifiability`, docs/methodology.md section 7).

The calibrated values are calibration-effective descriptors (docs/methodology.md section 6): the data may
fix only some combinations of them. This step computes which combinations the calibration data leave
undetermined, and ranks the candidate conditions given in the config (bias points, channel lengths,
gate-dielectric thickness splits) by the information each would add there. It is meant for simulators
whose equations cannot be inspected; in the demo's compact model the answer is known in closed form. It
uses simulator calls only, about 3.5 times as many as the default calibration, most of them for the
profiles of step 2 (their cost is set by `profile_points` and the number of flagged parameters):

1. Find. Fisher information F = S^T S / sigma^2 of the calibration log-current data, linearized at the
   calibrated point, where S = d log10 I_D / dx comes from central differences in the normalized
   coordinates x. An eigen-direction of F that carries less information than the approved bounds is
   flagged. A numerically zero one is an exact degeneracy and is reported as the combination that the
   data determine.
2. Verify. Profile likelihood of every parameter in a flagged direction, around the best log-current fit:
   fix the parameter, refit the others (bounded least squares), and keep the values with
   Delta chi^2 < 3.84 (95 %, one parameter).
3. Design. Each candidate condition is simulated at the calibrated point; it needs no data. A candidate
   resolves a flagged direction if it raises the information along it (the other parameters free) by at
   least as much as the approved bounds provide. Among those, the one with the largest D-optimal gain
   Delta log det F is recommended.
4. Resolve. Not automatic: add the recommended condition to the calibration split and run again.

The likelihood is Gaussian in log10 I_D on the points above the objective's measurement floor. If the
model fits worse than the noise (model-form error), sigma is raised to the residual RMS (sigma_eff), so
the model error is treated like independent noise. Model-form residuals are correlated along a curve, so
the intervals and information values are then optimistic; an exact degeneracy does not depend on this.
The approved bounds enter steps 1 and 3 as a Gaussian prior whose variance, 1/12 in x, is that of a
uniform distribution over the bounds. Along an exact degeneracy they are the only information. The
calibration result and J are not changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares

from .device import Condition
from .log import ActionLog
from .metrics import LOG_FLOOR_A
from .parameters import ParameterSpec
from .tools import Workspace

DCHI2_95 = 3.84  # 95 % quantile of chi^2 with one degree of freedom
PRIOR_PRECISION = 12.0  # information of the approved bounds per normalized coordinate: 1 / variance of U(0, 1)
EXACT_TOL = 1e-10  # eigenvalue / largest eigenvalue below this: numerically exact degeneracy
FD_STEP = 1e-4  # central-difference step in normalized coordinates
N_BISECT = 6  # bisection steps for each edge of a profile-likelihood interval


class IdentifiabilityError(RuntimeError):
    """The check could not be completed: a probe of a calibration curve did not converge."""


@dataclass(frozen=True)
class IdentifiabilitySettings:
    enabled: bool = False
    noise_dec: float = 0.013  # sigma of log10 I_D
    profile_points: int = 13  # grid across each approved range (a local grid at the optimum is added)
    candidates: tuple[Condition, ...] = ()


def check_identifiability(ws: Workspace, params: Mapping[str, float], active: frozenset[str],
                          settings: IdentifiabilitySettings, log: ActionLog | None = None) -> dict:
    """Steps 1-3, from the calibration data only.

    Only Workspace.simulate() is called. The refits (for sigma_eff and the profiles) stay inside the check,
    so the calibration result is not changed.
    """
    n_sim0 = ws.n_simulations
    params = dict(params)
    names = _analyzed_parameters(ws, active)
    data = _CalibrationData(ws)

    # sigma_eff from the best log-current fit (model-form error is treated as noise)
    best, chi2_noise = _fit(ws, data, active, params, names, settings.noise_dec, max_nfev=200)
    dof = max(data.y.size - len(names), 1)
    sigma = settings.noise_dec * max(1.0, float(np.sqrt(chi2_noise / dof)))
    chi2_best = chi2_noise * (settings.noise_dec / sigma) ** 2  # in units of sigma_eff

    # 1. Find
    s = _sensitivity(ws, params, active, data.conds, data.masks, names)
    if s is None:
        raise IdentifiabilityError("a sensitivity probe of a calibration curve did not converge")
    f_data = s.T @ s / sigma**2
    f_reg = f_data + PRIOR_PRECISION * np.eye(len(names))
    eigval, eigvec = np.linalg.eigh(f_data)
    exact = eigval <= EXACT_TOL * eigval[-1]
    if exact.sum() > 1:  # any basis of a joint null space is valid; use the sparsest, one combination each
        eigvec[:, exact] = _sparse_basis(eigvec[:, exact])
    flagged, vectors = [], []
    for lam, v in zip(eigval, eigvec.T):
        if lam >= PRIOR_PRECISION:
            continue
        v = v * np.sign(v[np.argmax(np.abs(v))])
        part = [n for n, vj in zip(names, v) if abs(vj) >= 0.1]
        item = {"eigenvalue_rel": float(max(lam, 0.0) / eigval[-1]), "exact": bool(lam <= EXACT_TOL * eigval[-1]),
                "parameters": part, "direction": {n: _step(ws.specs[n], vj) for n, vj in zip(names, v) if n in part}}
        if item["exact"] and len(part) == 2:
            item["determined_combination"] = _combination(ws.specs, params, item["direction"], part)
        flagged.append(item)
        vectors.append(v)
    find = {"eigenvalues_rel": (np.maximum(eigval, 0.0) / eigval[-1]).tolist(), "flagged": flagged}
    if log is not None:
        log.record("identifiability_find", parameters=names, sigma_dec=settings.noise_dec, sigma_eff_dec=sigma,
                   **find)

    # 2. Verify
    cov = np.linalg.inv(f_reg)
    verify = {}
    for j, n in enumerate(names):
        if any(n in f["parameters"] for f in flagged):
            verify[n] = _profile_likelihood(ws, data, active, best, names, n, sigma, chi2_best,
                                            path=cov[:, j] / cov[j, j], sigma_x=float(np.sqrt(cov[j, j])),
                                            n_grid=settings.profile_points)
    if log is not None:
        log.record("identifiability_verify", intervals={n: {k: v for k, v in r.items() if k != "profile"}
                                                         for n, r in verify.items()})

    # 3. Design
    design = _design(ws, params, active, names, f_reg, sigma, vectors, settings.candidates)
    if log is not None:
        log.record("identifiability_design", **design)

    verdict = {}
    for n in names:
        if any(f["exact"] and n in f["parameters"] for f in flagged):
            verdict[n] = "combination only (exact degeneracy)"
        elif n in verify:
            r = verify[n]
            verdict[n] = ("not determined within the approved bounds" if r["reaches_lower_bound"]
                          or r["reaches_upper_bound"] else "weakly determined (95% interval inside the bounds)")
        else:
            verdict[n] = "not flagged"
    return {"parameters": names, "sigma_dec": settings.noise_dec, "sigma_eff_dec": sigma, "n_points": int(data.y.size),
            "best_fit": {n: best[n] for n in names}, "find": find, "verify": verify, "design": design,
            "verdict": verdict, "n_simulations": ws.n_simulations - n_sim0}


def _analyzed_parameters(ws: Workspace, active: frozenset[str]) -> list[str]:
    """Parameters of every group available with these active components, in the config order."""
    used = {n for g in ws.groups.values() if g.component is None or g.component in active for n in g.parameters}
    return [n for n in ws.specs if n in used]


def _profile_likelihood(ws: Workspace, data: "_CalibrationData", active: frozenset[str], best: Mapping[str, float],
                        names: list[str], name: str, sigma: float, chi2_best: float, path: np.ndarray,
                        sigma_x: float, n_grid: int) -> dict:
    """Delta chi^2 of `name` fixed at a value, the other parameters refitted, and its 95 % interval.

    From the best fit, the profile is followed outward on each side over a grid (the approved range in
    `n_grid` points plus 1-4 linear 1-sigma around the optimum) until Delta chi^2 reaches 3.84; that edge is
    then bisected. A side that never reaches 3.84 reaches the approved bound. Each refit starts from the
    previous point, moved along the linearized profile `path` (dx_others / dx_name). Delta chi^2 is taken
    from the lowest chi^2 seen so far, in case a refit improves on the best fit.
    """
    spec, j = ws.specs[name], names.index(name)
    free = [n for n in names if n != name]
    x_best = np.array([ws.specs[n].to_x(best[n]) for n in names])
    grid = np.concatenate([np.linspace(0.0, 1.0, n_grid), x_best[j] + sigma_x * np.arange(-4, 5)])
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    chi2 = {float(x_best[j]): chi2_best}

    def refit(xg, x_from):
        start = np.clip(x_from + path * (xg - x_from[j]), 0.0, 1.0)
        p, c = _fit(ws, data, active, {n: ws.specs[n].from_x(v) for n, v in zip(names, start)}, free, sigma,
                    fixed={name: spec.from_x(xg)}, max_nfev=100)
        chi2[float(xg)] = c
        return c - min(chi2.values()), np.array([ws.specs[n].to_x(p[n]) for n in names])

    edge, reaches = {}, {}
    for side, points in ((+1, grid[grid > x_best[j]]), (-1, grid[grid < x_best[j]][::-1])):
        x_in, v_in = x_best[j], x_best
        edge[side], reaches[side] = (1.0 if side > 0 else 0.0), True
        for xg in points:
            d, v = refit(xg, v_in)
            if d < DCHI2_95:
                x_in, v_in = xg, v
                continue
            x_out = xg
            for _ in range(N_BISECT):
                mid = 0.5 * (x_in + x_out)
                d, v = refit(mid, v_in)
                if d < DCHI2_95:
                    x_in, v_in = mid, v
                else:
                    x_out = mid
            edge[side], reaches[side] = 0.5 * (x_in + x_out), False
            break
    ref = min(chi2.values())
    return {"interval_95": [spec.from_x(edge[-1]), spec.from_x(edge[+1])], "unit": spec.unit,
            "reaches_lower_bound": reaches[-1], "reaches_upper_bound": reaches[+1],
            "profile": [{"value": spec.from_x(x), "dchi2": chi2[x] - ref} for x in sorted(chi2)]}


class _CalibrationData:
    """log10 of the measured calibration currents above the objective's measurement floor."""

    def __init__(self, ws: Workspace):
        floor = ws.objective_spec.measurement_floor_a
        self.conds = list(ws.calibration)
        self.masks = {c: ws.measured[c] >= floor for c in self.conds}
        self.y = np.concatenate([np.log10(ws.measured[c][self.masks[c]]) for c in self.conds])


def _log_current(ws, params, active, conds, masks) -> np.ndarray | None:
    out = []
    for c in conds:
        res = ws.simulate(params, active, c)
        current = res.id[masks[c]]
        if not res.converged or not np.all(np.isfinite(current)):
            return None
        out.append(np.log10(np.maximum(current, LOG_FLOOR_A)))
    return np.concatenate(out)


def _sensitivity(ws, params, active, conds, masks, names) -> np.ndarray | None:
    """d log10 I_D / dx per parameter: central differences, one-sided at a bound. None if a probe fails."""
    cols = []
    for n in names:
        spec = ws.specs[n]
        x0 = spec.to_x(params[n])
        lo, hi = max(x0 - FD_STEP, 0.0), min(x0 + FD_STEP, 1.0)
        y_lo = _log_current(ws, {**params, n: spec.from_x(lo)}, active, conds, masks)
        y_hi = _log_current(ws, {**params, n: spec.from_x(hi)}, active, conds, masks)
        if y_lo is None or y_hi is None:
            return None
        cols.append((y_hi - y_lo) / (hi - lo))
    return np.column_stack(cols)


def _fit(ws, data, active, start, free, sigma, fixed=None, max_nfev=300):
    """Minimize the log-current chi^2 over `free` inside the approved bounds. Returns (params, chi^2)."""
    base = {**start, **(fixed or {})}
    if not free:
        y = _log_current(ws, base, active, data.conds, data.masks)
        return base, float("inf") if y is None else float(np.sum(((y - data.y) / sigma) ** 2))

    def params_of(x):
        return {**base, **{n: ws.specs[n].from_x(xi) for n, xi in zip(free, x)}}

    def resid(x):
        y = _log_current(ws, params_of(x), active, data.conds, data.masks)
        return np.full(data.y.size, 1e3) if y is None else (y - data.y) / sigma

    x0 = np.clip([ws.specs[n].to_x(base[n]) for n in free], 1e-9, 1.0 - 1e-9)
    sol = least_squares(resid, x0, bounds=(0.0, 1.0), method="trf", diff_step=1e-6,
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=max_nfev)
    return params_of(sol.x), float(sol.fun @ sol.fun)


def _design(ws, params, active, names, f_reg, sigma, vectors, candidates) -> dict:
    """Information each candidate adds along every flagged direction, and its Delta log det F.

    The information along v is the marginal one, 1 / (v^T C v) with C the covariance, so the other
    parameters stay free.
    """
    floor = ws.objective_spec.measurement_floor_a
    cov = np.linalg.inv(f_reg)
    before = [1.0 / float(v @ cov @ v) for v in vectors]
    logdet0 = np.linalg.slogdet(f_reg)[1]
    splits_ok = getattr(ws.simulator, "supports_thickness_splits", False)
    rows = []
    for cond in candidates:
        if cond in ws.calibration:
            continue
        row = {"condition": cond.label, "usable": False}
        rows.append(row)
        if cond.t_ox_nm is not None and not splits_ok:
            row["reason"] = "the simulator does not declare supports_thickness_splits"
            continue
        res = ws.simulate(params, active, cond)
        mask = res.id >= floor if res.converged else np.zeros(ws.vg.size, bool)
        if not mask.any():
            row["reason"] = "no converged point above the measurement floor"
            continue
        s = _sensitivity(ws, params, active, [cond], {cond: mask}, names)
        if s is None:
            row["reason"] = "a sensitivity probe did not converge"
            continue
        f_new = f_reg + s.T @ s / sigma**2
        cov_new = np.linalg.inv(f_new)
        gain = [max(1.0 / float(v @ cov_new @ v) - b, 0.0) for v, b in zip(vectors, before)]
        row.update(usable=True, n_points=int(mask.sum()), information_gain_along_flagged=gain,
                   resolves=any(g >= PRIOR_PRECISION for g in gain),
                   dlogdetF=float(np.linalg.slogdet(f_new)[1] - logdet0), add_to_calibration=_split_block(cond))
    rows.sort(key=lambda r: (not r["usable"], -r.get("dlogdetF", 0.0)))
    best = next((r for r in rows if r["usable"] and r["resolves"]), None)
    return {"candidates": rows, "recommended": None if best is None else best["condition"],
            "add_to_calibration": None if best is None else best["add_to_calibration"]}


def summary_lines(report: dict) -> list[str]:
    """A few console lines: flagged directions, profile intervals, recommended candidate condition."""
    if "error" in report:
        return [f"not completed: {report['error']}"]
    flagged = report["find"]["flagged"]
    if not flagged:
        return ["nothing flagged: the data carry more information than the approved bounds in every direction "
                "(identifiable, not necessarily accurate)"]
    lines = []
    for f in flagged:
        kind = "exact degeneracy" if f["exact"] else f"weak direction (eigenvalue {f['eigenvalue_rel']:.2g} of max)"
        text = f"{kind}, combination only: {', '.join(f['parameters'])}"
        if "determined_combination" in f:
            c = f["determined_combination"]
            text += f"; the data fix {c['expression']} = {c['value']:.4g}"
        lines.append(text)
    exact = {n for f in flagged if f["exact"] for n in f["parameters"]}
    for n, r in report["verify"].items():
        lo, hi = r["interval_95"]
        if n in exact:
            note = " (set by the approved bounds along the degenerate line)"
        else:
            note = " (reaches an approved bound)" if r["reaches_lower_bound"] or r["reaches_upper_bound"] else ""
        lines.append(f"{n} 95% profile interval {lo:.4g} .. {hi:.4g} {r['unit']}{note}")
    d = report["design"]
    usable = [r for r in d["candidates"] if r["usable"]]
    if d["candidates"]:
        skipped = len(d["candidates"]) - len(usable)
        for i, f in enumerate(flagged):
            n_res = sum(r["information_gain_along_flagged"][i] >= PRIOR_PRECISION for r in usable)
            lines.append(f"{', '.join(f['parameters'])}: " + (f"{n_res} of {len(usable)} candidates add more "
                         "information than the approved bounds" if n_res else "no candidate adds information; "
                         "report the combination only") + (f" ({skipped} skipped)" if skipped else ""))
        lines.append(f"recommended candidate: {d['recommended'] or 'none'}")
    return lines


def _sparse_basis(v: np.ndarray) -> np.ndarray:
    """Basis of span(v) in which each vector is zero at the pivot parameters of the others.

    With pivoted QR, the pivots fall on different independent degeneracies, so separate pairs (for example
    phi_G-Q_f and an interface term with N_tail) come out as separate vectors.
    """
    piv = qr(v.T, pivoting=True)[2][: v.shape[1]]
    b = np.linalg.solve(v[piv].T, v.T).T  # b[piv] is the identity
    return b / np.linalg.norm(b, axis=0)


def _step(spec: ParameterSpec, dx: float) -> float:
    """Change of the parameter (log10 for log-scale parameters) for a change dx of its coordinate."""
    if spec.scale == "log":
        return float(dx * (np.log10(spec.upper) - np.log10(spec.lower)))
    return float(dx * (spec.upper - spec.lower))


def _combination(specs, params, direction: Mapping[str, float], part: list[str]) -> dict:
    """For an exact two-parameter degeneracy p-q: the combination p - k q that the data determine."""
    p, q = part

    def coord(n):
        return float(np.log10(params[n])) if specs[n].scale == "log" else float(params[n])

    def label(n):
        return f"log10({n})" if specs[n].scale == "log" else n

    k = direction[p] / direction[q]
    sign = "-" if k >= 0 else "+"
    return {"expression": f"{label(p)} {sign} {abs(k):.4g} * {label(q)}", "slope": k,
            "value": coord(p) - k * coord(q)}


def _split_block(cond: Condition) -> dict:
    """The condition as a block for `splits.calibration` in the YAML config."""
    block = {"length_um": cond.length_um, "vd": [cond.vd]}
    if cond.t_ox_nm is not None:
        block["t_ox_nm"] = cond.t_ox_nm
    return block
