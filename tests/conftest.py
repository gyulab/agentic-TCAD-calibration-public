from pathlib import Path

import pytest

from agentic_tcad_demo.config import load_config
from agentic_tcad_demo.runner import run_demo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "synthetic_demo.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="session")
def demo_run():
    """One full default run (rule planner, no files written), shared by read-only tests."""
    return run_demo(load_config(CONFIG), None)
