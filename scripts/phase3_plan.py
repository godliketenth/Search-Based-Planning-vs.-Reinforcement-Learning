#!/usr/bin/env python
"""Phase 3: plan a level with A* over motion primitives, then verify the plan.

Planning is only half the job. A plan that cannot be replayed is worthless, so this script
always does three things:

1. **Search** -- A* over motion primitives, with exact forward simulation as both the
   transition function and the collision checker.
2. **Verify by replay** -- re-run the saved byte sequence from a fresh ``reset()`` and check
   it reproduces the planned trajectory exactly. This is what makes the "deterministic"
   evaluation condition meaningful, and it is the property sticky actions will destroy in
   Phase 5.
3. **Visualise** -- an overlay GIF with the plan drawn ahead of Mario, and a static figure
   of every configuration A* expanded, plotted over the level.

Usage
-----
    .venv/bin/python scripts/phase3_plan.py
    .venv/bin/python scripts/phase3_plan.py --goal-x 900 --no-gif      # quick partial plan
    .venv/bin/python scripts/phase3_plan.py --weight 2.0               # weighted A*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RAW_DIR, load_config  # noqa: E402
from src.env import ram_map  # noqa: E402
from src.env.factories import make_planner_env  # noqa: E402
from src.env.state import MarioState, WorldState  # noqa: E402
from src.planning.astar import SearchBudget, plan_level  # noqa: E402
from src.planning.heuristic import DistanceHeuristic  # noqa: E402
from src.planning.plan import Plan  # noqa: E402
from src.planning.primitives import primitives_from_config  # noqa: E402
from src.planning.simulator import ForwardSimulator  # noqa: E402
from src.viz.debug_overlay import render_overlay  # noqa: E402

GIF_FPS = 30

#: How far ahead of Mario to draw the remaining plan, in frames.
PATH_LOOKAHEAD_FRAMES = 90


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="astar.yaml")
    parser.add_argument("--env-config", default="env.yaml")
    parser.add_argument("--level", help="override: 'world-stage'")
    parser.add_argument("--seed", type=int, help="override: reset seed")
    parser.add_argument("--goal-x", type=int, dest="goal_x", help="override: heuristic goal x")
    parser.add_argument("--weight", type=float, help="override: heuristic inflation")
    parser.add_argument("--max-seconds", type=float, help="override: wall-clock budget")
    parser.add_argument("--scale", type=int, default=2, help="GIF upscaling")
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def verify_replay(level: str, seed: int, plan: Plan) -> dict:
    """Replay the plan from a fresh reset and compare against the planned outcome.

    A plan is an *open-loop* control sequence: it is executed blind. On a deterministic
    emulator that is exact, which this check confirms.

    Args:
        level: level to replay on.
        seed: reset seed the plan was built from.
        plan: the plan to verify.

    Returns:
        A dict describing what the replay produced.
    """
    env = make_planner_env(level)
    try:
        env.reset(seed=seed)
        ram = env.ram
        died_at = None
        for frame, byte in enumerate(plan.bytes_):
            env._frame_advance(byte)
            if ram_map.read_is_dying(ram) and died_at is None:
                died_at = frame
        info = env._get_info()
        return {
            "final_x": ram_map.read_x(ram),
            "flag_get": bool(info["flag_get"]),
            "died_at_frame": died_at,
            "matches_plan": (
                ram_map.read_x(ram) == plan.final_x
                and bool(info["flag_get"]) == plan.flag_get
                and died_at is None
            ),
        }
    finally:
        env.close()


def render_plan_gif(level: str, seed: int, plan: Plan, path: Path,
                    stride: int, scale: int) -> int:
    """Replay the plan, drawing the remaining trajectory ahead of Mario.

    Drawing the plan is what makes planning observable instead of an invisible internal
    detail -- the same overlay the Phase 5b live demo uses.

    Returns:
        How many frames were written.
    """
    import imageio.v2 as imageio

    env = make_planner_env(level, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        # First pass: where will Mario be on every frame of the plan?
        env.reset(seed=seed)
        ram = env.ram
        positions: list[tuple[int, int]] = []
        for byte in plan.bytes_:
            env._frame_advance(byte)
            positions.append((ram_map.read_x(ram), ram_map.read_y_screen(ram) + 16))

        # Second pass: render, drawing the not-yet-executed part of that trajectory.
        env.reset(seed=seed)
        for frame_index, byte in enumerate(plan.bytes_):
            if frame_index % stride == 0:
                info = env._get_info()
                world = WorldState.from_env(env, info["flag_get"], info["time"])
                ahead = positions[frame_index:frame_index + PATH_LOOKAHEAD_FRAMES]
                frames.append(
                    render_overlay(
                        env.render(), world, path=ahead, scale=scale,
                        hud={
                            "frame": f"{frame_index}/{len(plan)}",
                            "nodes": plan.stats.nodes_expanded,
                            "plan": "A* open-loop",
                        },
                    )
                )
            env._frame_advance(byte)
    finally:
        env.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000.0 / GIF_FPS, loop=0)
    return len(frames)


def render_search_figure(expanded: list[MarioState], plan: Plan | None, path: Path) -> None:
    """Plot every configuration A* expanded, over the level's x-y plane.

    This is the figure for the report: it shows the search *exploring* -- dense clusters of
    expansions where the planner had to work out a jump, sparse runs where momentum carried
    it -- rather than only the final path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [state.x for state in expanded]
    ys = [state.y for state in expanded]
    order = np.arange(len(expanded))

    figure, axes = plt.subplots(figsize=(14, 4))
    scatter = axes.scatter(xs, ys, c=order, cmap="viridis", s=8, alpha=0.75,
                           label="expanded configurations")
    if plan is not None:
        axes.axvline(plan.final_x, color="crimson", linestyle="--", linewidth=1,
                     label=f"goal reached (x={plan.final_x})")
    axes.invert_yaxis()  # y increases downwards in the C-space
    axes.set_xlabel("level x (pixels)")
    axes.set_ylabel("level y (pixels, downwards)")
    axes.set_title(
        f"A* expanded node set -- {len(expanded)} configurations"
        + (f", plan {len(plan)} frames" if plan else "")
    )
    figure.colorbar(scatter, ax=axes, label="expansion order")
    axes.legend(loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    """Entry point."""
    args = parse_args()
    env_config = load_config(args.env_config, {"level": args.level, "seed": args.seed})
    config = load_config(args.config, {"goal_x": args.goal_x, "weight": args.weight})
    level, seed = env_config["level"], env_config["seed"]

    budget_config = dict(config["budget"])
    if args.max_seconds is not None:
        budget_config["max_seconds"] = args.max_seconds
    budget = SearchBudget(**budget_config)
    primitives = primitives_from_config(config.get("primitives"))

    print("=" * 72)
    print(f"Phase 3 A* planner -- level {level}, goal_x {config['goal_x']}, "
          f"weight {config['weight']}")
    print("=" * 72)
    print(f"primitives: {', '.join(f'{p.name}({p.frames}f)' for p in primitives)}")

    env = make_planner_env(level)
    try:
        simulator = ForwardSimulator(env)
        heuristic = DistanceHeuristic(goal_x=config["goal_x"])

        def progress(count: int, state: MarioState) -> None:
            print(f"  {count:6d} expanded   best x = {state.x:5d}", flush=True)

        print("\n[1/4] Searching")
        plan, stats, expanded = plan_level(
            simulator, primitives, heuristic,
            level=level, seed=seed,
            goal_x=args.goal_x,
            weight=config["weight"],
            budget=budget,
            lattice=config.get("lattice"),
            on_progress=progress,
        )
    finally:
        env.close()

    print(f"\n  goal reached   : {stats.goal_reached}")
    print(f"  nodes expanded : {stats.nodes_expanded}")
    print(f"  edges simulated: {stats.nodes_generated}  ({stats.deaths} fatal, "
          f"{stats.nodes_pruned} pruned as duplicates)")
    print(f"  frames emulated: {stats.frames_simulated:,}")
    print(f"  peak frontier  : {stats.peak_frontier} snapshots "
          f"(~{stats.peak_frontier_mb:.0f} MB)")
    print(f"  wall clock     : {stats.wall_s:.1f} s  ({stats.edges_per_s:.0f} edges/s)")

    if plan is None:
        print("\nNo plan found inside the budget. Try --weight 2.0 or a larger budget.")
        sys.exit(1)

    lower_bound = heuristic(MarioState(x=40, y=0, vx=0, vy=0, on_ground=True, powerup=0))
    print(f"\n  plan length    : {len(plan)} frames")
    print(f"  lower bound    : {lower_bound:.0f} frames (admissible heuristic from start)")
    print(f"  ratio to bound : {len(plan) / lower_bound:.3f}")

    print("\n[2/4] Verifying by open-loop replay from reset()")
    replay = verify_replay(level, seed, plan)
    print(f"  final x        : {replay['final_x']} (planned {plan.final_x})")
    print(f"  flag reached   : {replay['flag_get']}")
    print(f"  died           : {replay['died_at_frame'] is not None}")
    print(f"  REPLAY {'MATCHES' if replay['matches_plan'] else 'DIVERGED'}")

    tag = f"{level}_w{config['weight']:.1f}_seed{seed}"
    plan_path = plan.save(RAW_DIR / f"phase3_plan_{tag}.json")
    print(f"\n[3/4] Plan saved  : {plan_path.relative_to(Path.cwd())}")

    print("[4/4] Artefacts")
    if not args.no_figure:
        figure_path = FIGURES_DIR / f"phase3_search_{tag}.png"
        render_search_figure(expanded, plan, figure_path)
        print(f"  search figure  : {figure_path.relative_to(Path.cwd())}")
    if not args.no_gif:
        gif_path = FIGURES_DIR / f"phase3_plan_{tag}.gif"
        count = render_plan_gif(level, seed, plan, gif_path, args.gif_stride, args.scale)
        print(f"  plan GIF       : {gif_path.relative_to(Path.cwd())} ({count} frames)")

    if not replay["matches_plan"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
