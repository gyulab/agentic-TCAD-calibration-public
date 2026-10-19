"""Load the YAML run configuration into typed objects."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .controller import Component, ControllerSettings
from .device import CompactTFT, Condition, DeviceConstants, Simulator
from .objective import ObjectiveSpec
from .parameters import ParameterGroup, ParameterSpec


@dataclass
class DemoConfig:
    raw: dict
    seed: int
    vg: np.ndarray
    constants: DeviceConstants
    truth: CompactTFT
    truth_params: dict[str, float]
    truth_active: frozenset[str]
    noise: dict[str, float]
    simulator: Simulator
    initial: dict[str, float]
    specs: dict[str, ParameterSpec]
    groups: dict[str, ParameterGroup]
    components: dict[str, Component]
    splits: dict[str, list[Condition]]
    objective: ObjectiveSpec
    controller: ControllerSettings
    ofat_step: float
    review_mode: str
    approved_components: tuple[str, ...]


def _conditions(block: dict) -> list[Condition]:
    return [Condition(float(vd), float(block["length_um"])) for vd in block["vd"]]


def _load_plugin(entry: str, options: dict) -> Simulator:
    module, _, cls = entry.partition(":")
    return getattr(importlib.import_module(module), cls)(**options)


def load_config(path: str | Path) -> DemoConfig:
    raw = yaml.safe_load(Path(path).read_text())
    sweep = raw["sweep"]
    n = int(round((sweep["vg_max"] - sweep["vg_min"]) / sweep["vg_step"])) + 1
    vg = np.round(np.linspace(sweep["vg_min"], sweep["vg_max"], n), 10)
    constants = DeviceConstants(**raw.get("device", {}))

    truth_block = raw["synthetic_truth"]
    truth = CompactTFT(constants, mobility_gamma=float(truth_block["mobility_gamma"]),
                       dibl_v_per_v_um=float(truth_block.get("dibl_v_per_v_um", 0.0)))

    sim_block = raw.get("simulator", {"kind": "synthetic"})
    if sim_block["kind"] == "synthetic":
        simulator: Simulator = CompactTFT(constants, mobility_gamma=float(raw["calibration_model"]["mobility_gamma"]))
    elif sim_block["kind"] == "plugin":
        simulator = _load_plugin(sim_block["entry"], sim_block.get("options", {}))
    else:
        raise ValueError(f"unknown simulator kind {sim_block['kind']!r}")

    specs = {name: ParameterSpec(name, float(p["lower"]), float(p["upper"]), p.get("scale", "linear"),
                                 p.get("unit", ""), p.get("component"))
             for name, p in raw["parameters"].items()}
    groups = {}
    for name, g in raw["groups"].items():
        if isinstance(g, list):
            groups[name] = ParameterGroup(name, tuple(g))
        else:
            groups[name] = ParameterGroup(name, tuple(g["parameters"]), g.get("component"))
    components = {name: Component(name, c["addresses"], c["trigger"],
                                  {k: float(v) for k, v in c["initial"].items()},
                                  tuple(c.get("controls", ())), c.get("description", ""))
                  for name, c in raw["components"].items()}
    initial = {k: float(v) for k, v in raw["calibration_model"]["initial"].items()}
    # Parameters of inactive components still need a value; they have no effect until activated.
    for name, c in components.items():
        for k, v in c.initial.items():
            initial.setdefault(k, v)

    obj = raw.get("objective", {})
    objective = ObjectiveSpec(
        weights={k: float(v) for k, v in obj.get("weights", ObjectiveSpec().weights).items()},
        normalization={k: float(v) for k, v in obj.get("normalization", ObjectiveSpec().normalization).items()},
        measurement_floor_a=float(obj.get("measurement_floor_A", 10e-12)),
    )
    ctrl = dict(raw.get("controller", {}))
    ofat_step = float(ctrl.pop("ofat_step", 0.02))
    review = raw.get("review", {})
    return DemoConfig(
        raw=raw, seed=int(raw["run"]["seed"]), vg=vg, constants=constants, truth=truth,
        truth_params={k: float(v) for k, v in truth_block["parameters"].items()},
        truth_active=frozenset(truth_block.get("active_components", [])),
        noise={k: float(v) for k, v in truth_block["noise"].items()},
        simulator=simulator, initial=initial, specs=specs, groups=groups, components=components,
        splits={name: _conditions(block) for name, block in raw["splits"].items()},
        objective=objective, controller=ControllerSettings(**ctrl), ofat_step=ofat_step,
        review_mode=review.get("mode", "policy"),
        approved_components=tuple(review.get("approve_components", [])),
    )
