"""Figures for the report.

Three figures, each answering one of the questions the metrics were chosen for:

* **completion by agent and condition** -- the headline, and where the open-loop vs
  closed-loop gap under perturbation becomes visible;
* **distance distributions** -- where each agent actually stops, which separates "fails
  early" from "fails at the last obstacle";
* **compute cost** -- emulated frames spent by each approach, on a log axis because the
  two differ by more than two orders of magnitude.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.metrics import DeathCause, EpisodeResult  # noqa: E402

#: Stable colours, so the same agent is the same colour across every figure.
AGENT_COLORS = {
    "random": "#9e9e9e",
    "astar_open_loop": "#e07b39",
    "astar_closed_loop": "#2a6f97",
    "ppo": "#5a9e5a",
}

#: Readable labels for the report.
AGENT_LABELS = {
    "random": "Random",
    "astar_open_loop": "A* open-loop",
    "astar_closed_loop": "A* closed-loop",
    "ppo": "PPO (pixels)",
}


def _ordered_agents(results: list[EpisodeResult]) -> list[str]:
    """Agent names in a stable, meaningful order."""
    preferred = ["random", "astar_open_loop", "astar_closed_loop", "ppo"]
    present = {result.agent for result in results}
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def plot_completion(results: list[EpisodeResult], path: Path, level: str) -> None:
    """Grouped bars: completion rate per agent, per condition.

    The pair of bars for A* is the project's central result -- how much of the open-loop
    plan's success survives perturbation, and how much closed-loop replanning recovers.
    """
    subset = [r for r in results if r.level == level]
    if not subset:
        return
    agents = _ordered_agents(subset)
    conditions = sorted({r.condition for r in subset})

    width = 0.8 / len(conditions)
    figure, axes = plt.subplots(figsize=(9, 4.5))
    for index, condition in enumerate(conditions):
        rates, errors = [], []
        for agent in agents:
            cell = [r for r in subset if r.agent == agent and r.condition == condition]
            rate = sum(r.completed for r in cell) / len(cell) if cell else 0.0
            rates.append(rate)
            # Binomial standard error, to make small-n honest about its uncertainty.
            errors.append(
                np.sqrt(rate * (1 - rate) / len(cell)) if cell else 0.0
            )
        offsets = np.arange(len(agents)) + index * width - 0.4 + width / 2
        axes.bar(offsets, rates, width, yerr=errors, capsize=3,
                 label=condition, alpha=0.65 + 0.35 * index,
                 color=[AGENT_COLORS.get(a, "#777") for a in agents],
                 edgecolor="black", linewidth=0.6)

    axes.set_xticks(np.arange(len(agents)))
    axes.set_xticklabels([AGENT_LABELS.get(a, a) for a in agents])
    axes.set_ylabel("completion rate")
    axes.set_ylim(0, 1.08)
    axes.set_title(f"Level completion on {level}, by agent and condition")
    axes.legend(title="condition")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_distance(results: list[EpisodeResult], path: Path, level: str,
                  goal_x: int = 3161) -> None:
    """Strip plot of furthest x reached, per agent and condition.

    Shows *where* agents stop, not just how often they finish. A cluster well short of the
    flag means a specific obstacle; a spread means inconsistent control.
    """
    subset = [r for r in results if r.level == level]
    if not subset:
        return
    agents = _ordered_agents(subset)
    conditions = sorted({r.condition for r in subset})

    figure, axes = plt.subplots(figsize=(9, 4.5))
    rng = np.random.default_rng(0)
    for index, condition in enumerate(conditions):
        for position, agent in enumerate(agents):
            cell = [r.max_x for r in subset
                    if r.agent == agent and r.condition == condition]
            if not cell:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(cell))
            x = position + (index - (len(conditions) - 1) / 2) * 0.28 + jitter
            axes.scatter(x, cell, s=28, alpha=0.7,
                         color=AGENT_COLORS.get(agent, "#777"),
                         marker="o" if index == 0 else "^",
                         edgecolor="black", linewidth=0.4,
                         label=condition if position == 0 else None)

    axes.axhline(goal_x, color="crimson", linestyle="--", linewidth=1,
                 label=f"flagpole (x={goal_x})")
    axes.axhline(594, color="grey", linestyle=":", linewidth=1,
                 label="first pipe (x=594)")
    axes.set_xticks(np.arange(len(agents)))
    axes.set_xticklabels([AGENT_LABELS.get(a, a) for a in agents])
    axes.set_ylabel("furthest x reached")
    axes.set_title(f"Distance reached on {level} (circle = first condition, triangle = second)")
    axes.legend(loc="lower right", fontsize=8)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_compute_cost(results: list[EpisodeResult], path: Path) -> None:
    """Emulated frames spent by each approach, on a log axis.

    The point of the figure is the order of magnitude, not the precise bar heights: A* and
    PPO differ by more than 100x in frames burned, for the same completion rate on 1-1.
    """
    agents = _ordered_agents(results)
    frames = []
    for agent in agents:
        cell = [r.compute_frames for r in results if r.agent == agent]
        frames.append(max(cell) if cell else 0)

    figure, axes = plt.subplots(figsize=(8, 4))
    bars = axes.bar(
        [AGENT_LABELS.get(a, a) for a in agents], [max(f, 1) for f in frames],
        color=[AGENT_COLORS.get(a, "#777") for a in agents],
        edgecolor="black", linewidth=0.6,
    )
    axes.set_yscale("log")
    axes.set_ylabel("emulated frames (log scale)")
    axes.set_title("Compute cost: frames burned by planning (A*) or training (PPO)")
    for bar, value in zip(bars, frames):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                  f"{value:,}" if value else "0", ha="center", fontsize=9)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_death_causes(results: list[EpisodeResult], path: Path, level: str) -> None:
    """Stacked bars of how each agent failed.

    Whether perturbation shifts failures from enemies to pits says something a completion
    rate cannot: pits punish mistimed commitment, enemies punish mistimed reaction.
    """
    subset = [r for r in results if r.level == level and not r.completed]
    if not subset:
        return
    agents = _ordered_agents(subset)
    conditions = sorted({r.condition for r in subset})
    causes = [DeathCause.ENEMY, DeathCause.PIT, DeathCause.TIMEOUT]
    cause_colors = {"enemy": "#c0392b", "pit": "#34495e", "timeout": "#bdc3c7"}

    labels = [f"{AGENT_LABELS.get(a, a)}\n{c}" for a in agents for c in conditions]
    bottoms = np.zeros(len(labels))
    figure, axes = plt.subplots(figsize=(10, 4.5))
    for cause in causes:
        counts = [
            sum(1 for r in subset
                if r.agent == agent and r.condition == condition and r.death_cause == cause)
            for agent in agents for condition in conditions
        ]
        axes.bar(labels, counts, bottom=bottoms, label=cause,
                 color=cause_colors[cause], edgecolor="black", linewidth=0.5)
        bottoms += np.array(counts, dtype=float)

    axes.set_ylabel("failed episodes")
    axes.set_title(f"How each agent failed on {level}")
    axes.legend(title="cause")
    axes.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
