"""Bounded calibration parameters, their normalized coordinates, and parameter groups."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    scale: str = "linear"  # "linear" or "log"
    unit: str = ""
    component: str | None = None  # None = base model; otherwise the model component that owns it

    def to_x(self, value: float) -> float:
        """Physical value -> normalized coordinate in [0, 1]."""
        if self.scale == "log":
            return (np.log10(value) - np.log10(self.lower)) / (np.log10(self.upper) - np.log10(self.lower))
        return (value - self.lower) / (self.upper - self.lower)

    def from_x(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        if self.scale == "log":
            value = 10 ** (np.log10(self.lower) + x * (np.log10(self.upper) - np.log10(self.lower)))
        else:
            value = self.lower + x * (self.upper - self.lower)
        return float(np.clip(value, self.lower, self.upper))  # guard against round-off at the bounds

    def within_bounds(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    parameters: tuple[str, ...]
    component: str | None = None  # group only becomes available once this component is active


def params_hash(params: Mapping[str, float], active: frozenset[str]) -> str:
    blob = json.dumps({"params": {k: repr(float(v)) for k, v in sorted(params.items())},
                       "active": sorted(active)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
