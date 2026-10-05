#!/usr/bin/env python
"""Phase 4: turn a training run's TensorBoard logs into the report's figure and CSV.

Reads the run's event files rather than stdout, so the curve is reproducible from what the
run durably wrote. Training may span several resumes -- each writes its own event file --
and these are merged in step order, which is why the step counter is preserved across
resumes in the first place.

The plotted metric is **completion rate**, not episode return. Return conflates "died late
with a high score" with "finished the level"; the project's question is whether the agent
solves the level, so that is what the figure shows.

Usage
-----
    .venv/bin/python scripts/phase4_curve.py
    .venv/bin/python scripts/phase4_curve.py --run-name ppo_1-1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR  # noqa: E402

#: Scalars pulled out of the event files.
TAGS = ("mario/completion_rate", "mario/mean_max_x", "rollout/ep_rew_mean", "time/fps")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="ppo_1-1")
    parser.add_argument("--smooth", type=int, default=20, help="rolling-mean window")
    return parser.parse_args()


def read_scalars(tensorboard_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Merge scalars from every event file under ``tensorboard_dir``.

    Args:
        tensorboard_dir: the run's TensorBoard directory.

    Returns:
        ``tag -> [(step, value), ...]``, sorted by step and de-duplicated. Resumed runs
        overlap slightly at the seam, so later writes win for a repeated step.
    """
    from tensorboard.backend.event_processing import event_accumulator

    merged: dict[str, dict[int, float]] = {tag: {} for tag in TAGS}
    event_files = sorted(tensorboard_dir.rglob("events.out.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files under {tensorboard_dir}")

    for path in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(path), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        for tag in TAGS:
            if tag in available:
                for event in accumulator.Scalars(tag):
                    merged[tag][event.step] = event.value

    return {tag: sorted(points.items()) for tag, points in merged.items()}


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Smooth a noisy series with a trailing rolling mean."""
    if window <= 1:
        return values
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def write_csv(scalars: dict[str, list[tuple[int, float]]], path: Path) -> None:
    """Write the merged curve as a CSV, one row per logged step."""
    steps = sorted({step for series in scalars.values() for step, _ in series})
    lookup = {tag: dict(series) for tag, series in scalars.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *(tag.split("/")[-1] for tag in TAGS)])
        for step in steps:
            writer.writerow([step, *(lookup[tag].get(step, "") for tag in TAGS)])


def plot(scalars: dict[str, list[tuple[int, float]]], smooth: int, path: Path) -> None:
    """Draw completion rate and distance against training steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps, values = zip(*scalars["mario/completion_rate"])
    top.plot(steps, values, color="lightsteelblue", linewidth=0.8, label="per update")
    top.plot(steps, rolling_mean(list(values), smooth), color="navy", linewidth=1.8,
             label=f"rolling mean ({smooth})")
    top.set_ylabel("completion rate")
    top.set_ylim(-0.05, 1.05)
    top.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="A* open-loop (1.00)")
    top.legend(loc="lower right")
    top.set_title("PPO on Super Mario Bros 1-1 -- level completion vs. training steps")

    steps, values = zip(*scalars["mario/mean_max_x"])
    bottom.plot(steps, values, color="lightsteelblue", linewidth=0.8)
    bottom.plot(steps, rolling_mean(list(values), smooth), color="darkgreen", linewidth=1.8)
    bottom.axhline(3161, color="crimson", linestyle="--", linewidth=1, label="flagpole (x=3161)")
    bottom.axhline(594, color="grey", linestyle=":", linewidth=1,
                   label="random agent ceiling (x=594)")
    bottom.set_ylabel("mean max x")
    bottom.set_xlabel("training steps (agent decisions)")
    bottom.legend(loc="lower right")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    """Entry point."""
    args = parse_args()
    tensorboard_dir = PROJECT_ROOT / "runs" / args.run_name / "tensorboard"
    scalars = read_scalars(tensorboard_dir)

    completion = scalars["mario/completion_rate"]
    final = rolling_mean([value for _, value in completion], args.smooth)[-1]
    print(f"run {args.run_name}: {len(completion)} logged updates, "
          f"final step {completion[-1][0]:,}")
    print(f"final completion rate (rolling mean of {args.smooth}): {final:.3f}")

    csv_path = TABLES_DIR / f"phase4_curve_{args.run_name}.csv"
    write_csv(scalars, csv_path)
    print(f"curve CSV : {csv_path.relative_to(PROJECT_ROOT)}")

    figure_path = FIGURES_DIR / f"phase4_curve_{args.run_name}.png"
    plot(scalars, args.smooth, figure_path)
    print(f"figure    : {figure_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
