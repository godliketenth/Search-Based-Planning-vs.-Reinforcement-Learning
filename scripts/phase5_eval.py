#!/usr/bin/env python
"""Phase 5: evaluate every agent under both conditions, through one harness.

The experiment
--------------
Four agents -- random, A* open-loop, A* closed-loop, PPO -- are run through **the same**
environment construction, stepping loop, frame accounting and metrics. The only thing that
differs is the integer each returns from ``act``.

Two conditions, because the emulator is deterministic and an open-loop plan would otherwise
score 100% trivially:

* **deterministic** -- the upper bound. Repeats here *verify* reproducibility rather than
  estimate a rate.
* **sticky actions, p = 0.25** -- each frame the previously executed action may repeat
  instead of the commanded one. This is actuation uncertainty: the planner's model is still
  exact, but the executed control is not the commanded one, so a plan that was feasible at
  plan time is not feasible at execution time.

The gap between open-loop and closed-loop A* under sticky actions is the central result.

Plus a transfer test on 1-2: PPO has never seen it, and A* runs on it unchanged.

Usage
-----
    .venv/bin/python scripts/phase5_eval.py
    .venv/bin/python scripts/phase5_eval.py --episodes-sticky 5 --levels 1-1   # quick
    .venv/bin/python scripts/phase5_eval.py --agents astar_open_loop astar_closed_loop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR, load_config  # noqa: E402
from src.agents.astar_closed_loop import AStarClosedLoopAgent  # noqa: E402
from src.agents.astar_open_loop import AStarOpenLoopAgent  # noqa: E402
from src.agents.ppo_agent import PPOEvalAgent  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.env.factories import make_planner_env  # noqa: E402
from src.eval import plots  # noqa: E402
from src.eval.harness import (  # noqa: E402
    condition_name, run_cell, write_results_csv, write_summary_csv,
)
from src.eval.metrics import summarise  # noqa: E402
from src.planning.primitives import primitives_from_config  # noqa: E402

ALL_AGENTS = ("random", "astar_open_loop", "astar_closed_loop", "ppo")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="eval.yaml")
    parser.add_argument("--astar-config", default="astar.yaml")
    parser.add_argument("--agents", nargs="+", choices=ALL_AGENTS, default=list(ALL_AGENTS))
    parser.add_argument("--levels", nargs="+", help="override: levels to evaluate")
    parser.add_argument("--episodes-sticky", type=int, dest="episodes_sticky")
    parser.add_argument("--episodes-deterministic", type=int, dest="episodes_deterministic")
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    return parser.parse_args()


def build_factory(name: str, level: str, config: dict, astar_config: dict):
    """Return a zero-argument callable building a fresh agent, or ``None`` if unavailable.

    A fresh agent per episode matters: a leftover plan queue or policy state would silently
    corrupt the next episode's result.
    """
    if name == "random":
        return lambda: RandomAgent(seed=config["seed"])

    if name == "astar_open_loop":
        plan_path = PROJECT_ROOT / config["plans"].get(level, "")
        if not plan_path.exists():
            return None
        return lambda: AStarOpenLoopAgent(plan_path)

    if name == "astar_closed_loop":
        primitives = primitives_from_config(astar_config.get("primitives"))
        settings = config["closed_loop"]
        goal_x = config["goal_x"][level]

        def factory():
            planner_env = make_planner_env(level)
            return AStarClosedLoopAgent(
                planner_env=planner_env,
                primitives=primitives,
                goal_x=goal_x,
                replan_every=settings["replan_every"],
                horizon_px=settings["horizon_px"],
                weight=settings["weight"],
                max_nodes=settings["max_nodes"],
                max_seconds=settings["max_seconds"],
                lattice=astar_config.get("lattice"),
            )

        return factory

    if name == "ppo":
        ppo = config["ppo"]
        model = PROJECT_ROOT / ppo["model"]
        if not model.exists():
            return None
        return lambda: PPOEvalAgent(
            model_path=model,
            device=ppo["device"],
            deterministic=ppo["deterministic"],
            training_frames=ppo["training_frames"],
            training_seconds=ppo["training_seconds"],
        )

    raise ValueError(f"unknown agent {name!r}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(
        args.config,
        {
            "levels": args.levels,
            "episodes_sticky": args.episodes_sticky,
            "episodes_deterministic": args.episodes_deterministic,
        },
    )
    astar_config = load_config(args.astar_config)

    print("=" * 72)
    print("Phase 5 evaluation -- agents x conditions x levels, one harness")
    print("=" * 72)

    results = []
    skipped: list[str] = []
    started = time.perf_counter()

    for level in config["levels"]:
        for agent_name in args.agents:
            factory = build_factory(agent_name, level, config, astar_config)
            if factory is None:
                skipped.append(f"{agent_name} on {level} (missing plan or model)")
                continue

            for sticky_p in config["conditions"]:
                # The transfer level is evaluated deterministically only: the question there
                # is whether the approach generalises at all, not how it handles noise.
                if level == config["transfer_level"] and sticky_p > 0:
                    continue
                if level == config["transfer_level"]:
                    n_episodes = config["episodes_transfer"]
                elif sticky_p <= 0:
                    n_episodes = config["episodes_deterministic"]
                else:
                    n_episodes = config["episodes_sticky"]
                print(f"\n  {agent_name}  |  {level}  |  {condition_name(sticky_p)}  "
                      f"({n_episodes} episodes)")
                results.extend(
                    run_cell(
                        agent_factory=factory,
                        level=level,
                        sticky_p=sticky_p,
                        n_episodes=n_episodes,
                        base_seed=config["seed"],
                        decision_every=config["decision_every"].get(agent_name, 1),
                        max_frames=config["max_frames"],
                    )
                )

    wall = time.perf_counter() - started
    if not results:
        print("\nno results produced; check that plans and the PPO model exist")
        sys.exit(1)

    tag = args.tag or "main"
    results_path = TABLES_DIR / f"phase5_episodes_{tag}.csv"
    summary_rows = summarise(results)
    summary_path = TABLES_DIR / f"phase5_summary_{tag}.csv"
    write_results_csv(results, results_path)
    write_summary_csv(summary_rows, summary_path)

    print("\n" + "=" * 72)
    print(f"{'agent':20s} {'level':5s} {'condition':14s} {'n':>3s} {'compl':>6s} "
          f"{'mean x':>7s} {'ms/dec':>7s}")
    print("-" * 72)
    for row in summary_rows:
        print(f"{row['agent']:20s} {row['level']:5s} {row['condition']:14s} "
              f"{row['episodes']:3d} {row['completion_rate']:6.2f} "
              f"{row['mean_max_x']:7.0f} {row['decision_ms_mean']:7.2f}")
    print("=" * 72)

    for level in config["levels"]:
        goal_x = config["goal_x"][level]
        plots.plot_completion(results, FIGURES_DIR / f"phase5_completion_{level}_{tag}.png", level)
        plots.plot_distance(results, FIGURES_DIR / f"phase5_distance_{level}_{tag}.png",
                            level, goal_x=goal_x)
        plots.plot_death_causes(results, FIGURES_DIR / f"phase5_deaths_{level}_{tag}.png", level)
    plots.plot_compute_cost(results, FIGURES_DIR / f"phase5_compute_{tag}.png")

    print(f"\nepisodes CSV : {results_path.relative_to(PROJECT_ROOT)}")
    print(f"summary CSV  : {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"figures      : {FIGURES_DIR.relative_to(PROJECT_ROOT)}/phase5_*_{tag}.png")
    if skipped:
        print("\nskipped cells:")
        for item in skipped:
            print(f"  - {item}")
    print(f"\ntotal wall clock: {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
