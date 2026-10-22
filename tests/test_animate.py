"""The video is rebuilt from the recorded run, so its numbers must match the run's own summary."""

import json

import numpy as np
import pytest

from agentic_tcad_demo.animate import Renderer, Story, _readable

from .conftest import ROOT

RUN = ROOT / "examples" / "planner_runs" / "claude-opus-5-5"


@pytest.fixture(scope="module")
def story(cfg):
    return Story(cfg, RUN / "action_log.jsonl")


def test_story_reconstructs_the_recorded_run(story):
    summary = json.loads((RUN / "summary.json").read_text())
    assert story.initial.j == pytest.approx(summary["J_initial"], rel=1e-4)
    assert story.final.j == pytest.approx(summary["J_final"], rel=1e-3)
    for k, v in summary["final_params"].items():
        assert story.final.params[k] == pytest.approx(v, rel=1e-5)
    assert sorted(story.final.active) == summary["final_active_components"]
    kinds = [s.kind for s in story.steps]
    assert kinds.count("activate_component") == 2 and kinds[-1] == "stop"
    rejected = [t for s in story.steps for t in s.trials if not t[1]]
    assert len(rejected) == summary["rejected_trials"]


def test_heldout_numbers_match_the_summary(story):
    summary = json.loads((RUN / "summary.json").read_text())
    rows = story.heldout_rows()
    for split in ("heldout_drain", "heldout_length"):
        vmax = max(abs(r[3]) for r in rows if r[0] == split)
        assert vmax == pytest.approx(summary["evaluation"][split]["final"]["max_abs_dVth_mV"], abs=0.1)


def test_captions_are_typeset_and_short():
    text = _readable("I_on is about 1.4 decades low and the S/D work function (4.75 eV) implies a large "
                     "contact barrier, while phi_G has only 0.11 eV left before its bound; fix the contact next.")
    assert "I$_{on}$" in text and "phi_" not in text and ";" not in text
    assert len(text) <= 168 and text.count("(") == text.count(")") and text.endswith(".")
    long = _readable("Use phi_G alone (phi_G and Q_f are degenerate, so moving both would only drift along the "
                     "null space of the two descriptors, which no measurement here can resolve at all).")
    assert long.count("(") == long.count(")")


def test_a_frame_renders(story, tmp_path):
    r = Renderer(story, "title", "footer")
    path = tmp_path / "frame.png"
    r.frame(path, story.final, caption=["Step 1 · test", "Why: test.", "✓ Kept.", "New values: none"])
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000
    assert np.isfinite(story.final.j)
