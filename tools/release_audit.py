"""Pre-publication audit of this repository.

Scans every tracked file (or, outside git, every file not ignored below) and, with --history, every blob
reachable from any ref. Exits non-zero if anything is found. Matched values are never printed in full.

    python tools/release_audit.py                         # working tree
    python tools/release_audit.py --history               # also all git history
    python tools/release_audit.py --denylist PRIVATE.txt  # extra private substrings (keep that file untracked)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EMAILS = {"kyur1ter1a@gmail.com"}
BINARY_ALLOW = re.compile(r"^docs/(figures/[\w\-]+\.png|media/[\w\-]+\.(png|gif|mp4))$")
MEDIA = re.compile(r"^docs/media/[\w\-]+\.(png|gif|mp4)$")
MAX_MEDIA_BYTES = 10_000_000  # GitHub's inline limit for uploaded media on free plans
FORBIDDEN_SUFFIXES = {".cmd", ".par", ".tdr", ".plt", ".grd", ".sav", ".dat", ".so", ".linux64", ".log",
                      ".csv", ".xlsx", ".xls", ".npz", ".pkl", ".pt", ".ckpt", ".env", ".pem", ".key"}
MAX_BYTES = 1_000_000
SKIP_DIRS = {".git", "outputs", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}

RULES = {
    "absolute_user_path": r"(?<![\w.])/(?:home|Users)/[\w.\-]+",  # site-specific paths: use --denylist
    "windows_user_path": r"[A-Za-z]:\\\\Users\\\\",
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    "license_server": r"\b\d{4,5}@[A-Za-z][\w.\-]+|SNPSLMD_LICENSE_FILE\s*=|LM_LICENSE_FILE\s*=",
    "private_ip": r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "api_key": r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}"
               r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|nvapi-[\w\-]{20,}|hf_[A-Za-z0-9]{30,}|xox[abpors]-[\w\-]{10,})",
    "secret_assignment": r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']",
    "vendor_copyright": r"(?i)copyright[^\n]{0,60}synopsys|synopsys,? inc\.",
    "deck_syntax": r"(?m)^\s*(?:Physics|Electrode|Math|Solve|Plot)\s*\{",
}
COMPILED = {k: re.compile(v) for k, v in RULES.items()}


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def in_git() -> bool:
    try:
        return git("rev-parse", "--is-inside-work-tree").strip() == b"true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def tree_files() -> dict[str, bytes]:
    if in_git():
        names = [n for n in git("ls-files", "-z").decode().split("\0") if n]
    else:
        names = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                 if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
                 and not any(part.endswith(".egg-info") for part in p.parts)]
    return {n: (ROOT / n).read_bytes() for n in sorted(names)}


def history_blobs() -> dict[str, bytes]:
    out = {}
    for line in git("rev-list", "--all", "--objects").decode().splitlines():
        sha, _, path = line.partition(" ")
        if path and git("cat-file", "-t", sha).strip() == b"blob":
            out[f"{path}@{sha[:10]}"] = git("cat-file", "-p", sha)
    return out


def redact(s: str) -> str:
    return s[:3] + "***" if len(s) > 3 else "***"


def scan(files: dict[str, bytes], deny: list[str]) -> list[str]:
    findings = []
    for name, data in files.items():
        path = name.split("@")[0]
        if path == "tools/release_audit.py":  # this file defines the patterns themselves
            continue
        suffix = Path(path).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES or Path(path).name.startswith(".env") and path != ".env.example":
            findings.append(f"{name}: forbidden file type")
        limit = MAX_MEDIA_BYTES if MEDIA.match(path) else MAX_BYTES
        if len(data) > limit:
            findings.append(f"{name}: larger than {limit} bytes")
        if b"\0" in data[:8192]:
            if not BINARY_ALLOW.match(path):
                findings.append(f"{name}: binary file outside docs/figures/ and docs/media/")
            if b"tEXt" in data or b"iTXt" in data:
                findings.append(f"{name}: PNG carries text metadata")
            continue
        text = data.decode("utf-8", errors="replace")
        for rule, rx in COMPILED.items():
            for m in rx.finditer(text):
                if rule == "email" and m.group(0).lower() in ALLOWED_EMAILS:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                findings.append(f"{name}:{line}: {rule} ({redact(m.group(0))})")
        low = text.lower()
        for word in deny:
            if word.lower() in low:
                findings.append(f"{name}: private denylist term ({redact(word)})")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="also scan every blob in git history")
    ap.add_argument("--denylist", type=Path, help="file with one private substring per line (# comments)")
    args = ap.parse_args()
    deny = []
    if args.denylist:
        deny = [w.strip() for w in args.denylist.read_text().splitlines() if w.strip() and not w.startswith("#")]
    files = tree_files()
    findings = scan(files, deny)
    scope = f"{len(files)} tracked files"
    if args.history:
        blobs = history_blobs()
        findings += scan(blobs, deny)
        scope += f", {len(blobs)} history blobs"
    for f in findings:
        print("FINDING", f)
    print(f"release audit: {scope}, {len(deny)} denylist terms, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
