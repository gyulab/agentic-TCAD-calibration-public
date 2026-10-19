"""Structured, append-only action log (JSON Lines)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Make values JSON-safe and stable: round floats to 6 significant digits, map nan to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.6g}")
    return value


class ActionLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[dict] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def record(self, event: str, **payload: Any) -> dict:
        entry = _clean({"seq": len(self.events), "event": event, **payload})
        self.events.append(entry)
        if self.path is not None:
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def of_type(self, event: str) -> list[dict]:
        return [e for e in self.events if e["event"] == event]
