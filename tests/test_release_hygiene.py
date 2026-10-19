"""Guards against overclaiming and accidental leakage in the public files."""

import subprocess
import sys

from .conftest import ROOT


def read(rel):
    return (ROOT / rel).read_text()


def test_readme_states_what_is_and_is_not_reproduced():
    text = read("README.md")
    assert "not a reproduction of the" in text and "synthetic" in text.lower()
    assert "arxiv.org/abs/2609.12184" in text and "10.48550/arXiv.2609.12184" in text
    assert "not affiliated with or endorsed by Synopsys" in text
    assert "[MIT](LICENSE)" in text and (ROOT / "LICENSE").read_text().startswith("MIT License")


PAPER_SECTIONS = ("## paper vs. this public demo", "## what the paper reports")


def test_paper_numbers_appear_only_as_paper_results():
    """The paper's headline numbers may appear only in the explicit paper sections or in lines naming the paper."""
    for rel in ("README.md", "docs/paper_vs_demo.md", "docs/methodology.md"):
        heading, fenced = "", False
        for line in read(rel).splitlines():
            if line.startswith("```"):
                fenced = not fenced
                continue
            if not fenced and line.startswith("## "):
                heading = line.lower()
            if any(num in line for num in ("14.3", "2.589", "0.181", "36.1", "46.2")):
                assert heading.startswith(PAPER_SECTIONS) or "paper" in line.lower(), (rel, line)


def test_release_audit_passes_on_the_working_tree():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "release_audit.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout


def shipped_files():
    """Files that would be published: git-tracked files, or (outside git) everything except local envs/outputs."""
    proc = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        return [ROOT / line for line in proc.stdout.splitlines()]
    skip = {".git", ".venv", "venv", "outputs", "__pycache__", ".pytest_cache"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


def test_no_simulator_decks_or_data_files_are_shipped():
    bad = {".cmd", ".par", ".tdr", ".plt", ".csv", ".so"}
    assert [p for p in shipped_files() if p.suffix in bad] == []
