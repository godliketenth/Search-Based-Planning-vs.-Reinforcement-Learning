#!/usr/bin/env python
"""Presentation figures: the whole comparison, in four pictures.

The per-phase figures answer per-phase questions. These answer the *project's*
question, and are built to be read from a projector at the back of a room:

1. **Robustness** (the hero). A dumbbell per agent, deterministic -> perturbed.
   Dumbbell rather than grouped bars because the data's job is *before -> after per
   item*, and the length of each connector IS the finding.
2. **Cost vs robustness.** Emulated frames against completion under perturbation,
   log x, because the two approaches differ by more than two orders of magnitude.
3. **How each agent fails.** 100% stacked composition, because the question is
   proportion-of-outcomes, not counts.
4. **Transfer.** Furthest distance on a level neither agent prepared for.

Colour follows the validated reference palette; every categorical set used here was
run through the palette validator (3 slots clear the all-pairs gate; the dumbbell is
a single hue in two ordinal steps). Aqua sits below 3:1 on a light surface, so every
mark carries a visible direct label - the relief rule, not an accident.

Usage
-----
    .venv/bin/python scripts/make_comparison_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES_DIR, PROJECT_ROOT  # noqa: E402

# --- validated reference palette -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e8e7e3"

BLUE_LIGHT = "#86b6ef"   # ordinal step 250 - "before"
BLUE_DARK = "#1c5cab"    # ordinal step 550 - "after"

SERIES_1 = "#2a78d6"     # blue   - A* closed-loop
SERIES_2 = "#eb6834"     # orange - A* open-loop
SERIES_3 = "#1baf7a"     # aqua   - PPO
NEUTRAL = "#9e9d97"      # random / baseline

#: Measured results. Single source of truth for every figure here.
AGENTS = ["Random", "A* open-loop", "A* closed-loop", "PPO (pixels)"]
DETERMINISTIC = [0.00, 1.00, 1.00, 1.00]
STICKY = [0.00, 0.15, 0.80, 0.35]
AGENT_COLOR = {"A* open-loop": SERIES_2, "A* closed-loop": SERIES_1,
               "PPO (pixels)": SERIES_3, "Random": NEUTRAL}

#: (completed, enemy deaths, pit deaths) out of 20 sticky episodes.
FAILURES = {
    "A* open-loop": (3, 11, 6),
    "A* closed-loop": (16, 3, 1),
    "PPO (pixels)": (7, 12, 1),
    "Random": (0, 20, 0),
}

#: Emulated frames spent: A* planning once, PPO training.
COMPUTE_FRAMES = {"A* open-loop": 71_637, "A* closed-loop": 71_637, "PPO (pixels)": 20_002_688}

#: Furthest x on level 1-2, which neither agent prepared for. Flag is ~3000.
TRANSFER = {"A* closed-loop": 950, "PPO (pixels)": 354, "Random": 355}


def _style(axes) -> None:
    """Recessive grid and axes; data should be the darkest thing on the page."""
    axes.set_facecolor(SURFACE)
    axes.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_2, labelsize=11, length=0)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)


def figure_robustness(path: Path) -> None:
    """The hero figure: what perturbation does to each agent.

    A dumbbell, not grouped bars. The reader's question is "how far did each one
    fall?", and in a dumbbell that is a length they can compare directly instead of
    a difference they have to compute between two bar heights.
    """
    figure, axes = plt.subplots(figsize=(11, 5.2))
    _style(axes)
    positions = np.arange(len(AGENTS))[::-1]

    for position, agent, before, after in zip(positions, AGENTS, DETERMINISTIC, STICKY):
        if before == after == 0:
            axes.plot([0], [position], "o", ms=11, color=NEUTRAL, zorder=3,
                      markeredgecolor=SURFACE, markeredgewidth=2)
            axes.text(0.022, position, "never completes", va="center", fontsize=11,
                      color=INK_MUTED, style="italic")
            continue
        axes.plot([after, before], [position, position], color=BLUE_LIGHT,
                  linewidth=5, solid_capstyle="round", zorder=2)
        # 2px surface ring so overlapping marks stay separable.
        axes.plot([before], [position], "o", ms=13, color=BLUE_LIGHT, zorder=4,
                  markeredgecolor=SURFACE, markeredgewidth=2)
        axes.plot([after], [position], "o", ms=13, color=BLUE_DARK, zorder=5,
                  markeredgecolor=SURFACE, markeredgewidth=2)
        axes.text(after - 0.028, position, f"{after:.2f}", va="center", ha="right",
                  fontsize=12, fontweight="bold", color=INK)
        axes.text(before + 0.028, position, f"{before:.2f}", va="center",
                  fontsize=12, color=INK_2)

    axes.set_yticks(positions)
    axes.set_yticklabels(AGENTS, fontsize=12.5, color=INK)
    axes.set_xlim(-0.16, 1.17)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xlabel("level completion rate", fontsize=11.5, color=INK_2, labelpad=10)
    axes.set_title("What happens when the controller stops obeying",
                   fontsize=16, color=INK, pad=34, loc="left", fontweight="bold")
    axes.text(0, 1.045, "Completion on level 1-1, before and after 25% sticky actions",
              transform=axes.transAxes, fontsize=12, color=INK_2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=11, color=BLUE_LIGHT,
                   markeredgecolor=SURFACE, markeredgewidth=2, label="deterministic"),
        plt.Line2D([], [], marker="o", ls="", ms=11, color=BLUE_DARK,
                   markeredgecolor=SURFACE, markeredgewidth=2, label="sticky actions, p = 0.25"),
    ]
    # Upper right: the Random row leaves that corner empty, and anywhere lower
    # collides with a dumbbell.
    axes.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                frameon=False, fontsize=11.5, labelcolor=INK_2, handletextpad=0.5)

    # The finding, said once, on the chart.
    axes.annotate("same planner, same heuristic —\nthe only difference is replanning",
                  xy=(0.80, positions[2]), xytext=(0.52, positions[2] + 0.62),
                  fontsize=11.5, color=SERIES_1, ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=SERIES_1, lw=1.6,
                                  connectionstyle="arc3,rad=-0.25"))
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_cost_vs_robustness(path: Path) -> None:
    """What each approach paid, against what it bought under perturbation."""
    figure, axes = plt.subplots(figsize=(9.5, 5.6))
    _style(axes)
    axes.grid(axis="y", color=GRID, linewidth=0.8)

    points = [
        ("A* open-loop", COMPUTE_FRAMES["A* open-loop"], STICKY[1], SERIES_2, (18, -22)),
        ("A* closed-loop", COMPUTE_FRAMES["A* closed-loop"], STICKY[2], SERIES_1, (18, 14)),
        ("PPO (pixels)", COMPUTE_FRAMES["PPO (pixels)"], STICKY[3], SERIES_3, (-16, 20)),
    ]
    for label, frames, completion, color, offset in points:
        axes.plot([frames], [completion], "o", ms=17, color=color, zorder=4,
                  markeredgecolor=SURFACE, markeredgewidth=2.5)
        axes.annotate(f"{label}\n{completion:.2f} under noise", xy=(frames, completion),
                      xytext=offset, textcoords="offset points", fontsize=11.5,
                      color=INK, fontweight="bold",
                      ha="right" if offset[0] < 0 else "left")

    axes.annotate("", xy=(20_002_688, 0.35), xytext=(71_637, 0.35),
                  arrowprops=dict(arrowstyle="<|-|>", color=INK_MUTED, lw=1.4,
                                  shrinkA=12, shrinkB=12))
    axes.text(1.2e6, 0.305, "279× more frames", fontsize=11.5, color=INK_MUTED,
              ha="center", style="italic")

    axes.set_xscale("log")
    axes.set_xlim(3e4, 7e7)
    axes.set_ylim(0, 1.0)
    axes.set_xlabel("emulated frames spent  (planning for A*, training for PPO)  — log scale",
                    fontsize=11.5, color=INK_2, labelpad=10)
    axes.set_ylabel("completion under 25% sticky actions", fontsize=11.5, color=INK_2)
    axes.set_title("What it cost, and what that bought",
                   fontsize=16, color=INK, pad=34, loc="left", fontweight="bold")
    axes.text(0, 1.045, "Further right is more compute. Higher is more robust.",
              transform=axes.transAxes, fontsize=12, color=INK_2)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_failure_modes(path: Path) -> None:
    """How the 20 perturbed episodes ended, as proportions."""
    figure, axes = plt.subplots(figsize=(10, 4.6))
    _style(axes)
    axes.grid(axis="x", color=GRID, linewidth=0.8)

    order = ["A* closed-loop", "PPO (pixels)", "A* open-loop", "Random"]
    positions = np.arange(len(order))[::-1]
    segments = [("completed", SERIES_1, 0), ("died on an enemy", NEUTRAL, 1),
                ("fell in a pit", SERIES_2, 2)]

    for position, agent in zip(positions, order):
        counts = FAILURES[agent]
        left = 0.0
        for label, color, index in segments:
            share = counts[index] / sum(counts)
            if share <= 0:
                continue
            # 2px surface gap between adjacent fills.
            axes.barh(position, share, left=left, height=0.58, color=color,
                      edgecolor=SURFACE, linewidth=2, zorder=3)
            if share > 0.1:
                axes.text(left + share / 2, position, f"{counts[index]}", ha="center",
                          va="center", fontsize=12, fontweight="bold",
                          color="#ffffff" if color != SERIES_2 else "#ffffff")
            left += share

    axes.set_yticks(positions)
    axes.set_yticklabels(order, fontsize=12.5, color=INK)
    axes.set_xlim(0, 1)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel("share of 20 perturbed episodes", fontsize=11.5, color=INK_2, labelpad=10)
    axes.set_title("How each agent's episodes ended",
                   fontsize=16, color=INK, pad=34, loc="left", fontweight="bold")
    # Claim must match the data: the others each fall in one pit, not zero.
    axes.text(0, 1.06, "Open-loop falls in pits 6x as often as any other agent — the signature of timing drift",
              transform=axes.transAxes, fontsize=12, color=INK_2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
               for label, color, _ in segments]
    axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=3, frameon=False, fontsize=11.5, labelcolor=INK_2)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_transfer(path: Path) -> None:
    """Distance reached on a level neither agent prepared for."""
    figure, axes = plt.subplots(figsize=(9, 4.2))
    _style(axes)

    order = ["A* closed-loop", "PPO (pixels)", "Random"]
    positions = np.arange(len(order))[::-1]
    for position, agent in zip(positions, order):
        value = TRANSFER[agent]
        axes.barh(position, value, height=0.56, color=AGENT_COLOR[agent],
                  zorder=3, edgecolor=SURFACE, linewidth=2)
        axes.text(value + 45, position, f"x = {value}", va="center", fontsize=12.5,
                  fontweight="bold", color=INK)

    axes.axvline(3000, color=INK_MUTED, linestyle="--", linewidth=1.3, zorder=2)
    axes.text(3000 - 60, positions[0] + 0.45, "level exit  ", ha="right", fontsize=11,
              color=INK_MUTED, style="italic")
    axes.set_yticks(positions)
    axes.set_yticklabels(order, fontsize=12.5, color=INK)
    axes.set_xlim(0, 3300)
    axes.set_xlabel("furthest distance reached on level 1-2 (pixels)",
                    fontsize=11.5, color=INK_2, labelpad=10)
    axes.set_title("On a level neither agent prepared for",
                   fontsize=16, color=INK, pad=34, loc="left", fontweight="bold")
    axes.text(0, 1.07, "Neither completes it — but PPO stops where random stops",
              transform=axes.transAxes, fontsize=12, color=INK_2)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    """Write all four figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("comparison_1_robustness.png", figure_robustness),
        ("comparison_2_cost_vs_robustness.png", figure_cost_vs_robustness),
        ("comparison_3_failure_modes.png", figure_failure_modes),
        ("comparison_4_transfer.png", figure_transfer),
    ]
    for name, builder in outputs:
        path = FIGURES_DIR / name
        builder(path)
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    print("presentation figures:")
    main()
