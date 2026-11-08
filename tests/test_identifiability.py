"""The optional identifiability check after the held-out evaluation (docs/methodology.md section 7)."""

import dataclasses

import numpy as np
import pytest

import agentic_tcad_demo.identifiability as ident_mod
import agentic_tcad_demo.tools as tools_mod
import agentic_tcad_demo.update as update_mod
from agentic_tcad_demo.config import load_config
from agentic_tcad_demo.device import Q, CompactTFT, Condition
from agentic_tcad_demo.identifiability import check_identifiability
from agentic_tcad_demo.log import _clean
from agentic_tcad_demo.runner import all_conditions, run_demo

from .conftest import CONFIG, ROOT

P = {"phi_SD": 4.40, "phi_G": 4.35, "mu0": 15.0, "Q_f": 5e12, "N_tail": 1e18}
VG = np.linspace(-1.5, 2.5, 201)


class Legacy:
    """A plug-in simulator written before thickness splits existed: it ignores Condition.t_ox_nm."""

    def __init__(self, inner):
        self.inner = inner

    def simulate(self, params, active, cond, vg):
        return self.inner.simulate(params, active, Condition(cond.vd, cond.length_um), vg)


def enabled(cfg):
    return dataclasses.replace(cfg.identifiability, enabled=True)


def checked_run(extra_calibration=()):
    cfg = load_config(CONFIG)
    cfg.identifiability = enabled(cfg)
    cfg.splits["calibration"] = cfg.splits["calibration"] + list(extra_calibration)
    return run_demo(cfg, None)


@pytest.fixture(scope="module")
def checked():
    return checked_run()


@pytest.fixture(scope="module")
def resolved(checked):
    """Step 4: the recommended split added to the calibration, then a fresh run."""
    block = checked["summary"]["identifiability"]["design"]["add_to_calibration"]
    return checked_run([Condition(block["vd"][0], block["length_um"], block["t_ox_nm"])])


def recheck(run, cfg):
    res = run["result"]
    return _clean(check_identifiability(run["workspace"], res.params, res.active, enabled(cfg)))


def test_check_is_opt_in_and_does_not_change_the_calibration(demo_run, checked):
    assert "identifiability" not in demo_run["summary"]
    for key in ("J_history", "final_params", "evaluation", "n_simulations_calibration"):
        assert checked["summary"][key] == demo_run["summary"][key]
    events = [e["event"] for e in checked["log"].events]
    assert events[-4:] == ["evaluation", "identifiability_find", "identifiability_verify", "identifiability_design"]


def test_check_uses_neither_heldout_data_nor_update_tools(checked, cfg, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("an update was attempted during the identifiability check")

    monkeypatch.setattr(update_mod, "bounded_gn_step", forbidden)
    monkeypatch.setattr(tools_mod, "bounded_gn_step", forbidden)
    ws = checked["workspace"]
    monkeypatch.setattr(ws, "measured", {c: m for c, m in ws.measured.items() if c in ws.calibration})
    assert recheck(checked, cfg) == checked["summary"]["identifiability"]


def test_find_reports_the_exact_gate_work_function_fixed_charge_degeneracy(checked, cfg):
    rep = checked["summary"]["identifiability"]
    (flag,) = rep["find"]["flagged"]
    assert flag["exact"] and flag["parameters"] == ["phi_G", "Q_f"]
    comb = flag["determined_combination"]
    assert comb["slope"] == pytest.approx(Q / cfg.constants.c_ox, rel=1e-3)  # V_T depends on phi_G - q Q_f / C_ox
    truth = cfg.truth_params
    assert comb["value"] == pytest.approx(truth["phi_G"] - comb["slope"] * truth["Q_f"], abs=0.01)
    assert all(rep["verdict"][n] == "not flagged" for n in ("phi_SD", "mu0", "N_tail"))


def test_profile_likelihood_spans_the_degenerate_range(checked, cfg):
    rep, final = checked["summary"]["identifiability"], checked["summary"]["final_params"]
    q_f, phi_g = rep["verify"]["Q_f"], rep["verify"]["phi_G"]
    assert q_f["reaches_upper_bound"] and not q_f["reaches_lower_bound"]
    assert phi_g["reaches_lower_bound"] and not phi_g["reaches_upper_bound"]
    # below this Q_f, phi_G would have to leave its approved range to keep V_T
    q_edge = final["Q_f"] + (cfg.specs["phi_G"].lower - final["phi_G"]) * cfg.constants.c_ox / Q
    assert q_f["interval_95"][0] == pytest.approx(q_edge, rel=0.02)


def test_only_oxide_thickness_splits_resolve_the_degeneracy(checked):
    design = checked["summary"]["identifiability"]["design"]
    assert len(design["candidates"]) == 8
    for r in design["candidates"]:
        (info,) = r["information_gain_along_flagged"]
        assert r["resolves"] == ("tox=" in r["condition"]) == (info > 12.0), r
        assert info < 1e-6 or info > 1e3, r  # the V_D and L tests add nothing along phi_G - Q_f
    assert design["add_to_calibration"] == {"length_um": 1.0, "vd": [0.05], "t_ox_nm": 14.0}


def test_the_recommended_split_resolves_the_degeneracy(checked, resolved, cfg):
    s = resolved["summary"]
    rep = s["identifiability"]
    assert rep["find"]["flagged"] == [] and set(rep["verdict"].values()) == {"not flagged"}
    assert rep["design"]["recommended"] is None
    assert s["final_params"]["Q_f"] == pytest.approx(cfg.truth_params["Q_f"], rel=0.1)
    # identifiable is not accurate: the controller ends with mu0 further from the truth than before
    mu0 = cfg.truth_params["mu0"]
    assert abs(s["final_params"]["mu0"] - mu0) > abs(checked["summary"]["final_params"]["mu0"] - mu0)


def test_several_exact_degeneracies_are_reported_separately():
    rng = np.random.default_rng(0)
    pairs = np.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -2.0]]).T  # two independent pairs
    mixed = np.linalg.qr(pairs @ rng.standard_normal((2, 2)))[0]  # an arbitrary basis, as eigh may return
    sparse = ident_mod._sparse_basis(mixed)
    supports = sorted(tuple(np.nonzero(np.abs(v) > 1e-12)[0]) for v in sparse.T)
    assert supports == [(0, 1), (3, 4)]


def test_methodology_numbers_match_a_fresh_run(checked, resolved):
    """Keeps the rounded numbers of docs/methodology.md section 7 honest, like the README test does."""
    text = (ROOT / "docs" / "methodology.md").read_text()
    rep = checked["summary"]["identifiability"]
    comb = rep["find"]["flagged"][0]["determined_combination"]
    block = rep["design"]["add_to_calibration"]
    before, after = checked["summary"]["final_params"], resolved["summary"]["final_params"]
    for expected in (f"{comb['slope'] * 1e14:.3f}×10⁻¹⁴ eV cm² × Q<sub>f</sub> ≈ {comb['value']:.2f} eV",
                     f"from about {rep['verify']['Q_f']['interval_95'][0] / 1e12:.1f}×10¹² cm⁻²",
                     f"to about {rep['verify']['phi_G']['interval_95'][1]:.2f} eV",
                     f"t<sub>ox</sub> = {block['t_ox_nm']:g} nm at V<sub>D</sub> = {block['vd'][0]:g} V",
                     f"Q<sub>f</sub> ≈ {after['Q_f'] / 1e12:.1f}×10¹² cm⁻² and φ<sub>G</sub> ≈ {after['phi_G']:.2f} eV",
                     f"from about {before['mu0']:.0f} to {after['mu0']:.0f} cm²/Vs",
                     f"gives μ<sub>0</sub> ≈ {resolved['summary']['identifiability']['best_fit']['mu0']:.0f} cm²/Vs"):
        assert expected in text, expected


def test_simulators_must_declare_thickness_support(checked, cfg, monkeypatch):
    other = load_config(CONFIG)
    other.simulator = Legacy(other.simulator)
    other.splits["calibration"] = other.splits["calibration"] + [Condition(0.05, 1.0, 14.0)]
    with pytest.raises(ValueError, match="supports_thickness_splits"):
        run_demo(other, None)
    monkeypatch.setattr(checked["workspace"], "simulator", Legacy(cfg.simulator))
    monkeypatch.setattr(ident_mod, "_profile_likelihood", lambda *a, **k: {  # not under test here
        "interval_95": [0.0, 1.0], "unit": "", "reaches_lower_bound": True, "reaches_upper_bound": True})
    design = recheck(checked, cfg)["design"]
    assert design["recommended"] is None
    assert all(not r["usable"] for r in design["candidates"] if "tox=" in r["condition"])


def test_a_thickness_split_leaves_the_other_synthetic_curves_unchanged(cfg):
    split = Condition(0.05, 1.0, t_ox_nm=14.0)
    other = load_config(CONFIG)
    other.splits["calibration"] = [split] + other.splits["calibration"]
    assert all_conditions(other) == all_conditions(cfg) + [split]


def test_thickness_split_device():
    dev, active = CompactTFT(), frozenset({"fixed_charge"})
    nominal = Condition(1.0, 1.0)
    assert dev.for_condition(nominal) is dev and nominal.label == "L=1um,VD=1V"
    same = Condition(1.0, 1.0, t_ox_nm=dev.const.t_ox_nm)
    assert np.array_equal(dev.simulate(P, active, same, VG).id, dev.simulate(P, active, nominal, VG).id)
    thick = dev.for_condition(Condition(1.0, 1.0, t_ox_nm=14.0))
    dvt = Q * P["Q_f"] * (1.0 / dev.const.c_ox - 1.0 / thick.const.c_ox)
    assert thick.threshold_voltage(P, active) - dev.threshold_voltage(P, active) == pytest.approx(dvt)
    assert thick.const.n_interface == pytest.approx(2.0 * dev.const.n_interface)  # C_it is kept
