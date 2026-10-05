#!/usr/bin/env python
"""Phase 5b: the live demo -- watch the planner plan.

Runs one agent in a window with the planner's trajectory drawn on the game itself: a dotted
path ahead of Mario, flashing each time it is recomputed. An mp4 is always recorded
alongside, as a backup for presenting.

The control worth knowing is **`s`**, which toggles sticky actions mid-run. It reproduces the
project's headline result live: with perturbation on, an open-loop plan visibly desynchronises
and Mario dies, while the closed-loop planner redraws its path and recovers. That is the
difference between 0.15 and 0.80 completion, demonstrated instead of tabulated.

Controls
--------
    space  pause / resume          [ ]  slower / faster
    n      step one frame           s   toggle sticky actions
    o      cycle overlay layers     r   restart the episode
    q      quit

    human agent: d right · f run right · w jump right · e run-jump · j jump · a left · x release

Usage
-----
    .venv/bin/python scripts/demo.py --agent astar_closed_loop
    .venv/bin/python scripts/demo.py --agent astar_open_loop --sticky 0.25
    .venv/bin/python scripts/demo.py --agent ppo --headless --frames 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.agents.astar_closed_loop import AStarClosedLoopAgent  # noqa: E402
from src.agents.astar_open_loop import AStarOpenLoopAgent  # noqa: E402
from src.agents.human_agent import HumanAgent  # noqa: E402
from src.agents.ppo_agent import PPOEvalAgent  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.env.factories import make_planner_env  # noqa: E402
from src.planning.plan import Plan  # noqa: E402
from src.planning.primitives import primitives_from_config  # noqa: E402
from src.planning.simulator import ForwardSimulator  # noqa: E402
from src.viz.live_demo import DemoRunner  # noqa: E402

AGENTS = ("astar_open_loop", "astar_closed_loop", "ppo", "random", "human")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[12:24]),
    )
    parser.add_argument("--config", default="demo.yaml")
    parser.add_argument("--astar-config", default="astar.yaml")
    parser.add_argument("--agent", choices=AGENTS, default="astar_closed_loop")
    parser.add_argument("--level", help="override: 'world-stage'")
    parser.add_argument("--sticky", type=float, dest="sticky_p",
                        help="override: sticky probability used when the toggle is on")
    parser.add_argument("--scale", type=int, help="override: window upscaling")
    parser.add_argument("--headless", action="store_true",
                        help="no window; still writes the mp4")
    parser.add_argument("--frames", type=int, help="headless only: frames to record")
    parser.add_argument("--no-record", action="store_true", help="skip the mp4")
    parser.add_argument("--out", type=Path, help="mp4 path (default: media/demo_<agent>.mp4)")
    return parser.parse_args()


def build_agent(name: str, level: str, config: dict, astar_config: dict):
    """Construct the requested agent, loading any cached plan or trained model.

    Returns:
        ``(agent, open_loop_plan_or_None)``. The plan is returned so the caller can
        precompute its on-screen trajectory.

    Raises:
        SystemExit: if a required artefact is missing, with the command that produces it.
    """
    if name == "random":
        return RandomAgent(seed=config["seed"]), None

    if name == "human":
        return HumanAgent(), None

    if name == "astar_open_loop":
        plan_path = PROJECT_ROOT / config["plans"].get(level, "")
        if not plan_path.exists():
            raise SystemExit(
                f"no cached plan for {level} at {plan_path}.\n"
                f"  produce one with: .venv/bin/python scripts/phase3_plan.py --level {level}"
            )
        plan = Plan.load(plan_path)
        return AStarOpenLoopAgent(plan), plan

    if name == "astar_closed_loop":
        settings = config["closed_loop"]
        return AStarClosedLoopAgent(
            planner_env=make_planner_env(level),
            primitives=primitives_from_config(astar_config.get("primitives")),
            goal_x=config["goal_x"][level],
            replan_every=settings["replan_every"],
            horizon_px=settings["horizon_px"],
            weight=settings["weight"],
            max_nodes=settings["max_nodes"],
            max_seconds=settings["max_seconds"],
            lattice=astar_config.get("lattice"),
        ), None

    if name == "ppo":
        ppo = config["ppo"]
        model = PROJECT_ROOT / ppo["model"]
        if not model.exists():
            raise SystemExit(
                f"no trained model at {model}.\n"
                "  train one with: .venv/bin/python scripts/phase4_train.py"
            )
        return PPOEvalAgent(model, device=ppo["device"],
                            deterministic=ppo["deterministic"]), None

    raise ValueError(f"unknown agent {name!r}")


def precompute_trajectory(plan: Plan, level: str, seed: int) -> list[tuple[int, int]]:
    """Replay a cached plan once to get the positions it will visit.

    Done up front rather than per frame: drawing the plan needs its predicted path, and
    producing that means running the whole plan through an emulator.
    """
    env = make_planner_env(level)
    try:
        simulator = ForwardSimulator(env)
        _, snapshot = simulator.root(seed=seed)
        return simulator.trajectory(snapshot, list(plan.bytes_))
    finally:
        env.close()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(
        args.config,
        {"level": args.level, "sticky_p": args.sticky_p, "scale": args.scale},
    )
    astar_config = load_config(args.astar_config)
    level = config["level"]

    agent, plan = build_agent(args.agent, level, config, astar_config)

    if plan is not None:
        print(f"precomputing the plan's trajectory ({len(plan)} frames)...")
        agent.set_trajectory(precompute_trajectory(plan, level, config["seed"]))

    record_path = None
    if not args.no_record:
        record_path = args.out or (
            PROJECT_ROOT / config["record_dir"] / f"demo_{args.agent}_{level}.mp4"
        )

    runner = DemoRunner(
        agent=agent,
        level=level,
        sticky_p=config["sticky_p"],
        seed=config["seed"],
        scale=config["scale"],
        decision_every=config["decision_every"].get(args.agent, 1),
        lookahead=config["lookahead"],
        max_frames=config["max_frames"],
        record_path=record_path,
        headless=args.headless,
    )

    print("=" * 68)
    print(f"Phase 5b demo -- {args.agent} on {level}")
    print("=" * 68)
    print(f"  sticky when toggled on : {config['sticky_p'] or 0.25}")
    print(f"  recording              : {record_path or 'disabled'}")
    if not args.headless:
        print("\n  space pause | n step | [ ] speed | s sticky | o overlay | r reset | q quit")
        print("  Press 's' mid-run to see open-loop break and closed-loop recover.\n")

    frames = runner.run_headless(args.frames) if args.headless else runner.run()
    print(f"\nrendered {frames} frames")
    if record_path:
        path = Path(record_path).resolve()
        # A user-supplied --out may sit outside the project, so fall back to the full path.
        try:
            shown = path.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = path
        print(f"mp4: {shown}")


if __name__ == "__main__":
    main()
