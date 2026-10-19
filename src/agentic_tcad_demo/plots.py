"""Static PNG figures for a demo run (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .objective import TERMS  # noqa: E402

# Validated categorical slots (light surface); measured data is neutral ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TERM_COLORS = dict(zip(TERMS, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))
TERM_LABELS = {"log": "log-current RMSE", "lin": "linear-current RMSE", "vth": "|ΔV_th|",
               "ion": "|Δlog I_on|", "ss": "|ΔSS|"}
SAVE = dict(dpi=150, facecolor=SURFACE, metadata={"Software": None})


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def _transfer(fig_path, vg, measured, curves, title, sim_label, sim_color):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, scale in zip(axes, ("log", "linear")):
        for cond, meas in measured.items():
            mfc = "none"
            ax.plot(vg, meas * (1 if scale == "log" else 1e6), "o", ms=3.5, mfc=mfc, mec=INK_2, mew=0.8,
                    label="synthetic measurement" if cond is next(iter(measured)) else None)
            ax.plot(vg, curves[cond] * (1 if scale == "log" else 1e6), color=sim_color, lw=2,
                    label=sim_label if cond is next(iter(measured)) else None)
            if scale == "log":
                ax.annotate(f"V_D={cond.vd:g} V", (vg[-1], meas[-1]), xytext=(-4, 6),
                            textcoords="offset points", ha="right", fontsize=7, color=INK_2)
        ax.set_yscale(scale)
        ax.set_xlabel("V_G (V)", fontsize=9, color=INK)
        ax.set_ylabel("I_D (A)" if scale == "log" else "I_D (µA)", fontsize=9, color=INK)
        if scale == "log":
            ax.set_ylim(1e-13, None)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, **SAVE)
    plt.close(fig)


def write_figures(out: dict, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    ws, res, s = out["workspace"], out["result"], out["summary"]
    cal = ws.calibration
    measured = {c: ws.measured[c] for c in cal}
    paths = []

    p = fig_dir / "fig1_initial_vs_target.png"
    _transfer(p, ws.vg, measured, res.initial.curves,
              f"Initial model vs synthetic target (L = 1 µm), J = {res.j_history[0]:.3g}", "initial model", ORANGE)
    paths.append(p)
    p = fig_dir / "fig2_final_vs_target.png"
    _transfer(p, ws.vg, measured, res.final.curves,
              f"Calibrated model vs synthetic target (L = 1 µm), J = {res.j_history[-1]:.3g}", "calibrated model", BLUE)
    paths.append(p)

    # J vs accepted update, with model activations labelled directly.
    p = fig_dir / "fig3_objective_trajectory.png"
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    steps = np.arange(len(res.j_history))
    ax.plot(steps, res.j_history, color=BLUE, lw=2, marker="o", ms=4)
    for k, acc in enumerate(res.accepted, start=1):
        if acc.get("activation"):
            ax.plot(k, res.j_history[k], "o", ms=9, mfc="none", mec=ORANGE, mew=2)
            ax.annotate(f"activate {acc['activation']}", (k, res.j_history[k]), xytext=(6, 8),
                        textcoords="offset points", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.set_xlabel("accepted update (0 = initial model)", fontsize=9, color=INK)
    ax.set_ylabel("objective J (Eq. 1)", fontsize=9, color=INK)
    ax.set_title(f"J {res.j_history[0]:.3g} → {res.j_history[-1]:.3g} over {len(res.accepted)} accepted updates "
                 f"({s['rejected_trials']} rejected trials)", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(p, **SAVE)
    plt.close(fig)
    paths.append(p)

    # Share of J carried by each weighted term after every accepted update (stacked to 1).
    # This shows which residual dominates, the signal that drives each action.
    p = fig_dir / "fig4_residual_components.png"
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    totals = np.array([sum(h.values()) for h in res.term_history])
    bottom = np.zeros(len(res.term_history))
    for t in TERMS:
        vals = np.array([h[t] for h in res.term_history]) / totals
        ax.bar(steps, vals, bottom=bottom, color=TERM_COLORS[t], edgecolor=SURFACE, linewidth=1.0,
               width=0.8, label=TERM_LABELS[t])
        bottom += vals
    for k, acc in enumerate(res.accepted, start=1):
        if acc.get("activation"):
            ax.annotate(acc["activation"], (k, 1.0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(steps)
    ax.set_xlabel("accepted update (0 = initial model)", fontsize=9, color=INK)
    ax.set_ylabel("share of J", fontsize=9, color=INK)
    ax.set_title("Which residual dominates J after each accepted update", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax)
    fig.tight_layout()
    fig.savefig(p, **SAVE)
    plt.close(fig)
    paths.append(p)

    # Held-out evaluation: two measures -> two panels, never a dual axis.
    p = fig_dir / "fig5_heldout_evaluation.png"
    rows = []
    for split in ("calibration", "heldout_drain", "heldout_length"):
        for r in s["evaluation_rows"][split]["final"]:
            rows.append((split, r))
    labels = [f"{'cal' if sp == 'calibration' else 'held-out'}\nL={r['length_um']:g}\nV_D={r['vd']:g}" for sp, r in rows]
    colors = [INK_2 if sp == "calibration" else BLUE for sp, _ in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, unit in zip(axes, ("dVth_mV", "dlogIon_dec"), ("ΔV_th (mV)", "Δlog10 I_on (dec)")):
        vals = [r[key] for _, r in rows]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, width=0.7, edgecolor=SURFACE, linewidth=1.0)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3g}", (xi, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
        ax.axhline(0, color=INK_2, lw=0.8)
        lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
        pad = 0.15 * (hi - lo or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(f"sim − meas, {unit}", fontsize=9, color=INK)
        _style(ax)
    fig.suptitle("Frozen calibrated model on calibration (gray) and held-out (blue) conditions, no retuning (L in µm, V_D in V)",
                 fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(p, **SAVE)
    plt.close(fig)
    paths.append(p)
    return paths
