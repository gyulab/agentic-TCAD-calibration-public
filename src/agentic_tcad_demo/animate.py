"""Render a calibration run (action_log.jsonl) as a short MP4 and GIF for talks, posters and READMEs.

Every frame is drawn from the recorded decisions and the simulator, so the video can be rebuilt
exactly from the repository:

    python -m agentic_tcad_demo animate --log examples/planner_runs/claude-opus-5-5/action_log.jsonl

Each step shows what a device engineer looks at:
- the I_D-V_G fit in log and linear scale
- the V_th, SS and I_on errors
- the calibrated TCAD descriptors inside their allowed ranges
- the agent's action and reason, and whether the simulator accepted it
The run ends with the fixed model on held-out drain biases and channel length.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DemoConfig  # noqa: E402
from .device import Condition, synthesize_measurement  # noqa: E402
from .objective import objective, score_curve  # noqa: E402
from .plots import BLUE, GRID, INK, INK_2, ORANGE, SURFACE  # noqa: E402
from .runner import all_conditions  # noqa: E402

GREEN, RED = "#1baf7a", "#e34948"
W_IN, H_IN, DPI = 12.8, 7.2, 150  # 1920 x 1080 px

PARAM_ROWS = [  # (name, label, unit, formatter)
    ("phi_SD", "S/D work function φ$_{SD}$", "eV", lambda v: f"{v:.2f}"),
    ("phi_G", "Gate work function φ$_G$", "eV", lambda v: f"{v:.2f}"),
    ("mu0", "Mobility μ$_0$", "cm²/Vs", lambda v: f"{v:.1f}"),
    ("Q_f", "Fixed charge Q$_f$", "cm⁻²", lambda v: _sci(v)),
    ("N_tail", "Tail states N$_{tail}$", "cm⁻³eV⁻¹", lambda v: _sci(v)),
]
ACTION_TEXT = {
    "electrostatics": "shifts the gate work function",
    "contact": "lowers the S/D contact barrier",
    "transport": "adjusts the channel mobility",
    "fixed_charge": "adjusts gate work function and fixed charge",
    "tail_states": "adjusts the tail-state density",
}
NEW_PHYSICS = {"fixed_charge": "gate-interface fixed charge Q$_f$", "tail_states": "tail-state DOS (subthreshold slope)"}
SYMBOLS = {"phi_SD": "φ$_{SD}$", "phi_G": "φ$_G$", "mu0": "μ$_0$", "N_tail": "N$_{tail}$", "Q_f": "Q$_f$"}


def _sci(v: float) -> str:
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{v / 10**e:.1f}×10{sup}"


TOKENS = (("phi_SD", "φ$_{SD}$"), ("phi_G", "φ$_G$"), ("mu0", "μ$_0$"), ("N_tail", "N$_{tail}$"), ("Q_f", "Q$_f$"),
          ("V_th", "V$_{th}$"), ("I_on", "I$_{on}$"), ("cm2/Vs", "cm²/Vs"), ("~", "≈"), ("<=", "≤"), ("trust-region ", ""))


def _readable(text: str, limit: int = 150) -> str:
    """The agent's own reason, cut to whole sentences that fit two caption lines, typeset for the frame."""
    sentences = []
    for clause in (c.strip() for c in text.split(";") if c.strip()):
        sentences += [x.strip() for x in re.split(r"(?<=\.)\s+", clause) if x.strip()]
    out = ""
    for snt in sentences:
        if snt[0].isascii() and snt[0].isalpha():
            snt = snt[0].upper() + snt[1:]
        candidate = f"{out} {snt.rstrip('.')}.".strip()
        if len(candidate) > limit:
            break
        out = candidate
    if not out and sentences:  # a single over-long sentence: cut at a clause boundary
        first = sentences[0]
        for cut in (", while", ", so ", ", and "):
            k = first.find(cut, 50)
            if 0 < k <= limit:
                first = first[:k]
                break
        else:
            first = first[:limit].rsplit(" ", 1)[0]
        if first.count("(") > first.count(")"):
            first = first[:first.rfind("(")].rstrip()
        out = first.rstrip(".,") + "."
    for a, b in TOKENS:
        out = out.replace(a, b)
    return out


# --------------------------------------------------------------------------------------------
@dataclass
class State:
    params: dict
    active: frozenset
    curves: dict  # Condition -> I_D array (calibration conditions)
    j: float
    errors: dict  # dVth_V, dSS_mVdec, dIon_dec (mean over calibration biases)


@dataclass
class Step:
    index: int
    kind: str  # update_group | activate_component | stop
    target: str | None
    rationale: str
    trials: list = field(default_factory=list)  # (State, accepted: bool, J_before, J_trial)
    activation: dict | None = None


class Story:
    def __init__(self, cfg: DemoConfig, log_path: Path):
        self.cfg = cfg
        conds = all_conditions(cfg)
        self.vg = cfg.vg
        self.measured = synthesize_measurement(cfg.truth, cfg.truth_params, cfg.truth_active, conds, cfg.vg,
                                               cfg.seed, cfg.noise["lognormal_sigma"], cfg.noise["floor_noise_A"])
        self.cal = cfg.splits["calibration"]
        events = [json.loads(x) for x in Path(log_path).read_text().splitlines() if x.strip()]
        base = next(e for e in events if e["event"] == "baseline")
        self.initial = self.state(base["params"], frozenset(base["active"]))
        self.steps: list[Step] = []
        current = self.initial
        proposals = [e for e in events if e["event"] == "proposal"]
        for k, p in enumerate(proposals):
            nxt = proposals[k + 1]["seq"] if k + 1 < len(proposals) else 10**9
            block = [e for e in events if p["seq"] < e["seq"] < nxt]
            step = Step(k + 1, p["action"], p["target"], p["rationale"])
            for e in block:
                if e["event"] == "trial":
                    params = {**current.params, **{n: c["to"] for n, c in e["changes"].items()}}
                    comp = cfg.groups[e["group"]].component
                    active = current.active | ({comp} if comp else set())
                    trial = self.state(params, frozenset(active))
                    ok = e["decision"] == "accepted"
                    step.trials.append((trial, ok, e["J_before"], e["J_trial"]))
                    if ok:
                        current = trial
                elif e["event"] == "approval":
                    step.activation = {"component": e["component"], "reason": e["evidence"]["reason"]}
            self.steps.append(step)
        self.final = current
        self.heldout = {k: v for k, v in cfg.splits.items() if k != "calibration"}

    def state(self, params: dict, active: frozenset) -> State:
        curves = {c: self.cfg.simulator.simulate(params, active, c, self.vg).id for c in self.cal}
        scores = [score_curve(self.vg, curves[c], self.measured[c], self.cfg.objective) for c in self.cal]
        def worst(values):  # signed value with the largest magnitude over the calibration biases
            values = list(values)
            return float(values[int(np.argmax(np.abs(values)))])

        errors = {
            "dVth_V": worst(s.sim.vth - s.meas.vth for s in scores),
            "dSS_mVdec": worst(1e3 * (s.sim.ss - s.meas.ss) for s in scores),
            "dIon_dec": worst(np.log10(s.sim.ion / s.meas.ion) for s in scores),
        }
        return State(dict(params), active, curves, objective(scores), errors)

    def heldout_rows(self):
        rows = []
        for split, conds in self.heldout.items():
            for c in conds:
                sim = self.cfg.simulator.simulate(self.final.params, self.final.active, c, self.vg).id
                s = score_curve(self.vg, sim, self.measured[c], self.cfg.objective)
                rows.append((split, c, sim, 1e3 * (s.sim.vth - s.meas.vth), float(np.log10(s.sim.ion / s.meas.ion))))
        return rows


# --------------------------------------------------------------------------------------------
def _style(ax, size=10):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=size)


class Renderer:
    def __init__(self, story: Story, title: str, footer: str):
        self.s, self.title, self.footer = story, title, footer
        self.j_hist = [story.initial.j]
        self.marks: list[int] = []

    def _figure(self):
        fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
        fig.text(0.02, 0.955, self.title, fontsize=17, fontweight="bold", color=INK, va="center")
        fig.text(0.98, 0.955, "synthetic device data", fontsize=11, color=INK_2, va="center", ha="right")
        fig.text(0.02, 0.018, self.footer, fontsize=9.5, color=INK_2)
        return fig

    def _iv(self, ax_log, ax_lin, curves, ghost=None):
        vg = self.s.vg
        for c in self.s.cal:
            meas = self.s.measured[c]
            for ax, f in ((ax_log, 1.0), (ax_lin, 1e6)):
                ax.plot(vg[::4], meas[::4] * f, "o", ms=3.2, mfc="none", mec=INK_2, mew=0.7)
                if ghost is not None:
                    ax.plot(vg, ghost[c] * f, "--", color=ORANGE, lw=1.8)
                ax.plot(vg, curves[c] * f, color=BLUE, lw=2.4)
            ax_log.annotate(f"V$_D$ = {c.vd:g} V", (vg[-1], meas[-1]), xytext=(-2, 5), textcoords="offset points",
                            ha="right", fontsize=10, color=INK_2)
        ax_log.set_yscale("log")
        ax_log.set_ylim(1e-13, 3e-3)
        ax_log.set_ylabel("I$_D$ (A)", fontsize=12, color=INK)
        ax_lin.set_ylabel("I$_D$ (µA)", fontsize=12, color=INK)
        for ax in (ax_log, ax_lin):
            ax.set_xlabel("V$_G$ (V)", fontsize=12, color=INK)
            _style(ax)
        ax_log.plot([], [], "o", mfc="none", mec=INK_2, label="target (measurement)")
        ax_log.plot([], [], color=BLUE, lw=2.4, label="TCAD model")
        if ghost is not None:
            ax_log.plot([], [], "--", color=ORANGE, lw=1.8, label="rejected trial")
        ax_log.legend(fontsize=9.5, frameon=False, loc="lower right")
        ax_log.set_title("I$_D$–V$_G$, L = 1 µm (calibration data)", fontsize=12, color=INK, loc="left")
        ax_lin.set_title("linear scale", fontsize=12, color=INK, loc="left")

    def _readout(self, ax, errors):
        ax.axis("off")
        ax.set_title("Fit error vs. target (worst V$_D$)", fontsize=12, color=INK, loc="left")
        dv = errors["dVth_V"]
        rows = [("V$_{th}$ error", f"{dv:+.2f} V" if abs(dv) >= 0.1 else f"{1e3 * dv:+.0f} mV"),
                ("SS error", f"{errors['dSS_mVdec']:+.0f} mV/dec"),
                ("I$_{on}$ error", f"{errors['dIon_dec']:+.2f} dec")]
        for i, (k, v) in enumerate(rows):
            y = 0.78 - 0.3 * i
            ax.text(0.0, y, k, fontsize=13, color=INK_2, transform=ax.transAxes, va="center")
            ax.text(1.0, y, v, fontsize=17, color=INK, fontweight="bold", transform=ax.transAxes, va="center",
                    ha="right")

    def _params(self, ax, params, active, changed=()):
        ax.axis("off")
        ax.set_title("TCAD descriptors · allowed range", fontsize=12, color=INK, loc="left")
        specs = self.s.cfg.specs
        for i, (name, label, unit, fmt) in enumerate(PARAM_ROWS):
            y = 0.86 - 0.2 * i
            spec = specs[name]
            on = spec.component is None or spec.component in active
            color = INK if on else "#b9b8b2"
            weight = "bold" if name in changed else "normal"
            ax.text(0.0, y + 0.05, label, fontsize=10.5, color=color, fontweight=weight, transform=ax.transAxes)
            ax.text(1.0, y + 0.05, f"{fmt(params[name])} {unit}" if on else "not in model", fontsize=10,
                    color=color, fontweight=weight, ha="right", transform=ax.transAxes)
            ax.plot([0, 1], [y - 0.03, y - 0.03], color=GRID, lw=4, transform=ax.transAxes, solid_capstyle="round")
            if on:
                x = spec.to_x(params[name])
                at_limit = x <= 1e-3 or x >= 1 - 1e-3
                ax.plot([x], [y - 0.03], "o", ms=9, color=ORANGE if at_limit else BLUE, transform=ax.transAxes)
                if at_limit:
                    ax.text(x + (0.03 if x < 0.5 else -0.03), y - 0.09, "at limit", fontsize=9, color=ORANGE,
                            ha="left" if x < 0.5 else "right", transform=ax.transAxes)

    def _jplot(self, ax):
        n = len(self.j_hist)
        ax.plot(range(n), self.j_hist, color=BLUE, lw=2, marker="o", ms=3.5)
        for k in self.marks:
            ax.plot(k, self.j_hist[k], "o", ms=9, mfc="none", mec=ORANGE, mew=2)
        ax.set_yscale("log")
        ax.set_xlim(-0.5, max(10.5, n - 0.5))
        ax.set_ylim(0.2, 40)
        ax.set_title("Fit error J (paper Eq. 1)", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("accepted update", fontsize=10, color=INK_2)
        _style(ax, 9)

    def frame(self, path, state, ghost=None, caption=(), banner=None, changed=(), banner_color=ORANGE):
        """caption = (headline, why, result). `banner` highlights the headline (new physics, rejection)."""
        fig = self._figure()
        gs = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.9, bottom=0.075, height_ratios=[1.95, 1.2],
                              width_ratios=[1.25, 1.0, 0.95], hspace=0.42, wspace=0.28)
        ax_log, ax_lin = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        right = gs[0, 2].subgridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0.25)
        self._iv(ax_log, ax_lin, state.curves, ghost)
        self._readout(fig.add_subplot(right[0]), state.errors)
        self._params(fig.add_subplot(right[1]), state.params, state.active, changed)
        self._jplot(fig.add_subplot(gs[1, 2]))
        cap = fig.add_subplot(gs[1, 0:2])
        cap.axis("off")
        y = 0.97
        for i, line in enumerate(caption):
            if i == 0:
                style = dict(fontsize=15, fontweight="bold")
                if banner:
                    style.update(color="white", bbox=dict(boxstyle="round,pad=0.3", fc=banner_color, ec="none"))
                else:
                    style.update(color=INK)
                cap.text(0.0, y, line, transform=cap.transAxes, va="top", **style)
                y -= 0.25
                continue
            max_lines = 2 if i == 1 or i == len(caption) - 1 else 1
            for wrapped in textwrap.wrap(line, 84, break_long_words=False)[:max_lines]:
                cap.text(0.0, y, wrapped, fontsize=13, color=INK if i == 1 else INK_2, transform=cap.transAxes,
                         va="top")
                y -= 0.17
            y -= 0.03 if i == 1 else 0.0
        fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
        plt.close(fig)

    def heldout_frame(self, path, rows):
        fig = self._figure()
        gs = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.9, bottom=0.07, height_ratios=[2.15, 1],
                              width_ratios=[1.0, 1.0, 1.1], hspace=0.42, wspace=0.28)
        vg = self.s.vg
        for col, split, title in ((0, "heldout_drain", "New V$_D$ = 0.1, 0.2, 0.5 V (L = 1 µm)"),
                                  (1, "heldout_length", "New channel length L = 2 µm")):
            ax = fig.add_subplot(gs[0, col])
            for sp, c, sim, _, _ in rows:
                if sp != split:
                    continue
                meas = self.s.measured[c]
                ax.plot(vg[::4], meas[::4], "o", ms=3.2, mfc="none", mec=INK_2, mew=0.7)
                ax.plot(vg, sim, color=BLUE, lw=2.4)
                if split == "heldout_length":
                    ax.annotate(f"V$_D$ = {c.vd:g} V", (vg[-1], meas[-1]), xytext=(-2, 5),
                                textcoords="offset points", ha="right", fontsize=9.5, color=INK_2)
            ax.set_yscale("log")
            ax.set_ylim(1e-13, 3e-3)
            ax.set_title(title, fontsize=12, color=INK, loc="left")
            ax.set_xlabel("V$_G$ (V)", fontsize=12, color=INK)
            ax.set_ylabel("I$_D$ (A)", fontsize=12, color=INK)
            _style(ax)
        tab = fig.add_subplot(gs[0, 2])
        tab.axis("off")
        tab.set_title("Error on held-out conditions", fontsize=12, color=INK, loc="left")
        tab.text(0.0, 0.93, "L,  V$_D$", fontsize=10.5, color=INK_2, transform=tab.transAxes)
        tab.text(0.68, 0.93, "ΔV$_{th}$", fontsize=10.5, color=INK_2, ha="right", transform=tab.transAxes)
        tab.text(1.0, 0.93, "ΔI$_{on}$", fontsize=10.5, color=INK_2, ha="right", transform=tab.transAxes)
        for i, (sp, c, _, dvth, dion) in enumerate(rows):
            y = 0.8 - 0.13 * i
            tab.text(0.0, y, f"{c.length_um:g} µm,  {c.vd:g} V", fontsize=11.5, color=INK, transform=tab.transAxes)
            tab.text(0.68, y, f"{dvth:+.1f} mV", fontsize=11.5, color=INK, ha="right", transform=tab.transAxes)
            tab.text(1.0, y, f"{dion:+.3f} dec", fontsize=11.5, color=INK, ha="right", transform=tab.transAxes)
        cap = fig.add_subplot(gs[1, :])
        cap.axis("off")
        vmax = {sp: max(abs(r[3]) for r in rows if r[0] == sp) for sp in ("heldout_drain", "heldout_length")}
        cap.text(0.0, 0.92, "Transfer test: the calibrated model is held fixed (no retuning)", fontsize=15,
                 fontweight="bold", color="white", transform=cap.transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc=GREEN, ec="none"))
        cap.text(0.0, 0.5, f"max |ΔV$_{{th}}$| = {vmax['heldout_drain']:.1f} mV on new drain biases and "
                 f"{vmax['heldout_length']:.1f} mV on the 2 µm channel.", fontsize=14, color=INK,
                 transform=cap.transAxes, va="top")
        cap.text(0.0, 0.25, "The remaining error is physics this compact model does not contain "
                 "(drain-dependent V$_{th}$, gate-dependent mobility).", fontsize=12.5, color=INK_2,
                 transform=cap.transAxes, va="top")
        fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
        plt.close(fig)

    def card(self, path, lines, qr_url=None):
        fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
        if qr_url:
            try:
                import qrcode
                img = np.array(qrcode.make(qr_url, border=1).convert("L"))
                ax = fig.add_axes([0.435, 0.08, 0.13, 0.13 * W_IN / H_IN])
                ax.imshow(img, cmap="gray", interpolation="nearest")
                ax.axis("off")
            except ImportError:
                pass
        y = 0.8
        for text, size, weight, color in lines:
            fig.text(0.5, y, text, fontsize=size, fontweight=weight, color=color, ha="center", va="center")
            y -= 0.075 if size < 20 else 0.11
        fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
        plt.close(fig)


def _blend(a: State, b: State, t: float, vg_conds) -> State:
    """Interpolate curves (in log current) and parameters (in normalized coordinates) for a smooth move."""
    curves = {c: 10 ** ((1 - t) * np.log10(np.maximum(a.curves[c], 1e-30)) + t * np.log10(np.maximum(b.curves[c], 1e-30)))
              for c in vg_conds}
    return State(b.params if t > 0.5 else a.params, b.active if t > 0.5 else a.active, curves,
                 (1 - t) * a.j + t * b.j, {k: (1 - t) * a.errors[k] + t * b.errors[k] for k in a.errors})


# --------------------------------------------------------------------------------------------
def render(cfg: DemoConfig, log_path: Path, out_dir: Path, name: str, title: str, links: list[str],
           agent_note: str, gif: bool = True, morph_frames: int = 6, qr_url: str | None = None) -> dict:
    story = Story(cfg, log_path)
    footer = ("Synthetic device data and an open compact model stand in for the measured BG-IWO device and Sentaurus.  "
              "Method: arXiv:2609.12184")
    r = Renderer(story, title, footer)
    work = Path(tempfile.mkdtemp(prefix="tcad_anim_"))
    clips: list[tuple[Path, float]] = []  # (frame, seconds)

    def add(path, seconds):
        clips.append((path, seconds))

    n = 0

    def fpath():
        nonlocal n
        n += 1
        return work / f"f{n:04d}.png"

    p = fpath()
    r.card(p, [(title, 26, "bold", INK),
               ("An AI agent calibrates a TCAD transistor model to I$_D$–V$_G$ data", 16, "normal", INK),
               ("Each step: residual diagnosis → bounded parameter update or new physics → simulator check", 14,
                "normal", INK_2),
               ("A step is kept only if the simulation converges and the fit improves", 14, "normal", INK_2),
               ("synthetic device data · open-source demo", 12, "normal", INK_2)])
    add(p, 4.0)
    p = fpath()
    e0 = story.initial.errors
    r.frame(p, story.initial, caption=[
        "Start: an uncalibrated TCAD model",
        f"V$_{{th}}$ is off by {e0['dVth_V']:+.2f} V and I$_{{on}}$ by {e0['dIon_dec']:+.1f} decades.",
        "The agent reads these residuals and chooses which physical knob to change."])
    add(p, 4.0)

    current = story.initial
    for step in story.steps:
        if step.kind == "stop":
            p = fpath()
            r.frame(p, current, caption=[
                f"Step {step.index} · The agent stops",
                "Why: every remaining update would improve the fit by less than 2 %.",
                "The rest of the mismatch is physics this compact model lacks (drain-dependent V$_{th}$, "
                "gate-dependent mobility), so more fitting would only chase noise."])
            add(p, 4.5)
            continue
        for trial, ok, j_before, j_trial in step.trials:
            changed = [k for k in trial.params if trial.params[k] != current.params[k]]
            change_txt = ", ".join(
                f"{SYMBOLS[k]} {PARAM_ROWS_FMT[k](current.params[k])} → {PARAM_ROWS_FMT[k](trial.params[k])} {UNITS[k]}"
                if current.params[k] != 0 else f"{SYMBOLS[k]} = {PARAM_ROWS_FMT[k](trial.params[k])} {UNITS[k]}"
                for k in changed)
            new_phys = step.kind == "activate_component"
            if step.kind == "activate_component":
                comp = step.activation["component"] if step.activation else step.target
                head = f"Step {step.index} · New physics: {NEW_PHYSICS.get(comp, comp)}"
                banner = "new"
                why = "Why: " + _activation_reason(comp, step.activation, current)
            else:
                head = f"Step {step.index} · Agent {ACTION_TEXT.get(step.target, step.target)}"
                banner = None
                why = "Why: " + _readable(step.rationale)
            if ok:
                result = f"✓ Kept: simulation converged and the fit improved (J {j_before:.3g} → {j_trial:.3g})."
                for k in range(1, morph_frames + 1):
                    p = fpath()
                    r.frame(p, _blend(current, trial, k / morph_frames, story.cal), caption=[head, why],
                            banner=banner, changed=changed)
                    add(p, 0.08)
                r.j_hist.append(trial.j)
                if new_phys:
                    r.marks.append(len(r.j_hist) - 1)
                p = fpath()
                r.frame(p, trial, caption=[head, why, result, "New values:  " + change_txt], banner=banner,
                        changed=changed)
                add(p, 4.5 if banner else 3.0)
                current = trial
            else:
                result = f"✗ Rejected: the fit got worse (J {j_before:.3g} → {j_trial:.3g}), so the step is halved and retried."
                p = fpath()
                r.frame(p, current, ghost=trial.curves, caption=[head, why, result], banner=banner, changed=changed,
                        banner_color=RED if not banner else ORANGE)
                add(p, 3.5)
    rows = story.heldout_rows()
    p = fpath()
    r.heldout_frame(p, rows)
    add(p, 6.0)
    p = fpath()
    r.card(p, [("Agentic TCAD calibration", 24, "bold", INK)] +
           [(line, 14, "normal", INK) for line in links] +
           [(agent_note, 12, "normal", INK_2)], qr_url=qr_url)
    add(p, 5.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    concat = work / "list.txt"
    concat.write_text("".join(f"file '{f}'\nduration {d}\n" for f, d in clips) + f"file '{clips[-1][0]}'\n")
    mp4 = out_dir / f"{name}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                    "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact", str(mp4)], check=True)
    poster = out_dir / f"{name}_poster.png"
    shutil.copy(clips[-3][0] if len(clips) >= 3 else clips[-1][0], poster)
    outputs = {"mp4": mp4, "poster": poster, "seconds": sum(d for _, d in clips), "frames": n}
    if gif:
        g = out_dir / f"{name}.gif"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", str(g)], check=True)
        outputs["gif"] = g
    shutil.rmtree(work, ignore_errors=True)
    return outputs


PARAM_ROWS_FMT = {name: fmt for name, _, _, fmt in PARAM_ROWS}
UNITS = {name: unit for name, _, unit, _ in PARAM_ROWS}


def _activation_reason(component: str, activation: dict | None, before: State) -> str:
    """Device-language reason for adding a model term, from the recorded eligibility evidence."""
    reason = (activation or {}).get("reason", "")
    if "bound" in reason:
        dv = before.errors["dVth_V"]
        off = f"{dv:+.2f} V" if abs(dv) >= 0.1 else f"{1e3 * dv:+.0f} mV"
        return f"φ$_G$ reached its allowed limit while a {off} V$_{{th}}$ offset remains."
    if "dr_ss" in reason:
        return (f"the SS error ({before.errors['dSS_mVdec']:+.0f} mV/dec) persists and no existing parameter "
                "can change the subthreshold slope.")
    return reason


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()
