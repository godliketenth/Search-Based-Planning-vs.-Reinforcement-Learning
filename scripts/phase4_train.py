#!/usr/bin/env python
"""Phase 4: train PPO on Super Mario Bros from pixels.

PPO is the model-free counterpart to the Phase 3 planner. Where A* was handed a perfect
simulator and privileged RAM state and spent 54 s planning, PPO gets 84x84 grayscale frames
and has to discover the dynamics from millions of samples. The point of running it is to
measure that trade, not to win it.

Training is **resumable**: checkpoints are written periodically and ``--resume`` continues
from the newest one with the step counter intact, so a long run survives interruption.

Usage
-----
    .venv/bin/python scripts/phase4_train.py                        # fresh run
    .venv/bin/python scripts/phase4_train.py --resume               # continue
    .venv/bin/python scripts/phase4_train.py --timesteps 50000 --n-envs 4
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, RAW_DIR, load_config  # noqa: E402
from src.rl.train import TrainingPaths, resolve_device, train  # noqa: E402
from src.rl.vec_env import make_training_vec_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="ppo.yaml")
    parser.add_argument("--run-name", default="ppo_1-1", help="subdirectory under runs/")
    parser.add_argument("--level", help="override: 'world-stage'")
    parser.add_argument("--seed", type=int, help="override: seed")
    parser.add_argument("--n-envs", type=int, dest="n_envs", help="override: parallel workers")
    parser.add_argument("--device", help="override: cpu | mps | cuda | auto")
    parser.add_argument(
        "--timesteps", type=int, dest="total_timesteps",
        help="override: timesteps to run (additional, when resuming)",
    )
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(
        args.config,
        {
            "level": args.level,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "device": args.device,
            "total_timesteps": args.total_timesteps,
        },
    )

    paths = TrainingPaths(PROJECT_ROOT / "runs" / args.run_name)
    paths.ensure()
    device = resolve_device(config["device"])

    print("=" * 72)
    print(f"Phase 4 PPO training -- level {config['level']}, {config['n_envs']} workers, "
          f"device {device}")
    print("=" * 72)
    print(f"  timesteps   : {config['total_timesteps']:,}")
    print(f"  rollout     : {config['n_steps']} x {config['n_envs']} = "
          f"{config['n_steps'] * config['n_envs']:,} per update")
    print(f"  run dir     : {paths.run_dir.relative_to(PROJECT_ROOT)}")
    print(f"  tensorboard : tensorboard --logdir {paths.tensorboard.relative_to(PROJECT_ROOT)}")
    print()

    venv = make_training_vec_env(
        n_envs=config["n_envs"],
        level=config["level"],
        sticky_p=config["sticky_p"],
        seed=config["seed"],
        max_episode_steps=config["max_episode_steps"],
        skip=config["skip"],
        frame_size=config["frame_size"],
        n_stack=config["n_stack"],
    )
    started = time.perf_counter()
    try:
        model = train(
            venv=venv,
            config=config,
            paths=paths,
            total_timesteps=config["total_timesteps"],
            resume=args.resume,
        )
    finally:
        venv.close()
    wall_s = time.perf_counter() - started

    steps = int(model.num_timesteps)
    summary = {
        "run_name": args.run_name,
        "level": config["level"],
        "device": device,
        "n_envs": config["n_envs"],
        "total_timesteps": steps,
        "wall_s": wall_s,
        "agent_steps_per_s": config["total_timesteps"] / wall_s if wall_s else 0.0,
        "frames_per_s": config["total_timesteps"] * config["skip"] / wall_s if wall_s else 0.0,
        "final_model": str(paths.final_model),
    }
    summary_path = RAW_DIR / f"phase4_train_{args.run_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\ntrained to {steps:,} timesteps in {wall_s / 60:.1f} min "
          f"({summary['agent_steps_per_s']:,.0f} agent steps/s, "
          f"{summary['frames_per_s']:,.0f} frames/s)")
    print(f"model   : {paths.final_model.relative_to(PROJECT_ROOT)}")
    print(f"summary : {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
