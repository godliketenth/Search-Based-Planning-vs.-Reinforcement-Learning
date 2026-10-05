#!/usr/bin/env python
"""Phase 2 inspector: verify the C-space extraction and render the debug overlay.

A tile grid pulled out of RAM cannot be checked by reading numbers. This script checks it
two ways:

1. **Visually** -- it writes a GIF with the obstacle map, enemies, Mario's collision box and
   his velocity vector drawn over the real frame. A one-column error in the tile-buffer ring
   arithmetic is immediately obvious.
2. **Automatically** (``--verify``) -- it asserts properties that can only hold if the
   addresses are right:

   * ``vx`` predicts the measured change in ``x_pos``, and ``vy`` the change in screen y;
   * whenever ``on_ground`` is set, the tile under Mario's feet is **solid** (this is the
     real test of the grid's alignment: a shifted grid fails it immediately);
   * the first Goomba's extracted position converges on the x where Mario dies.

The ``reactive`` policy is itself a check: a controller that jumps only because the tile
grid says something solid is ahead gets past the pipe that Phase 1's random agent could
never clear. That it works at all is evidence the obstacle map means what we think.

Usage
-----
    .venv/bin/python scripts/phase2_inspect.py --verify
    .venv/bin/python scripts/phase2_inspect.py --level 1-2 --policy reactive
    .venv/bin/python scripts/phase2_inspect.py --no-gif --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RAW_DIR, load_config  # noqa: E402
from src.env import ram_map  # noqa: E402
from src.env.actions import NOOP, RIGHT_A_B, RIGHT_B  # noqa: E402
from src.env.factories import make_planner_env  # noqa: E402
from src.env.state import MarioState, WorldState  # noqa: E402
from src.viz.debug_overlay import render_overlay  # noqa: E402

GIF_FPS = 30

#: How far ahead the reactive policy looks for an obstacle, in pixels.
LOOKAHEAD_PX = 24


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="env.yaml")
    parser.add_argument("--level", help="override config: 'world-stage'")
    parser.add_argument("--seed", type=int, help="override config: RNG seed")
    parser.add_argument("--steps", type=int, default=1200, help="frames to emulate")
    parser.add_argument(
        "--policy",
        choices=("right", "reactive"),
        default="reactive",
        help="'right' holds run-right; 'reactive' jumps using the tile grid (default)",
    )
    parser.add_argument("--scale", type=int, default=2, help="GIF upscaling (default: 2)")
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--verify", action="store_true", help="run the extraction checks")
    return parser.parse_args()


def reactive_action(world: WorldState) -> int:
    """Run right; jump when the tile grid says something solid is ahead, or a pit is.

    This is **not** a planner -- it has no lookahead beyond one tile and no model of the
    dynamics. It exists to exercise and validate the obstacle map.

    Args:
        world: the current state.

    Returns:
        A joypad byte.
    """
    mario = world.mario
    if not mario.on_ground:
        # Keep holding right+A so an in-progress jump completes: Mario's air control is
        # limited, so releasing mid-jump cannot be undone. This is the kinodynamic
        # constraint showing up even in a trivial controller.
        return RIGHT_A_B

    feet_y = world.mario_y_screen + 32
    ahead_x = mario.x + LOOKAHEAD_PX
    ram_tiles = world.tiles

    def solid_at(x_level: int, y_screen: int) -> bool:
        col = x_level // ram_map.TILE_SIZE - ram_tiles.col0
        row = (y_screen - ram_map.TILE_TOP_Y) // ram_map.TILE_SIZE
        if not (0 <= col < ram_map.TILE_COLS and 0 <= row < ram_map.TILE_ROWS):
            return False
        return bool(ram_map.is_solid(int(ram_tiles.tiles[row, col])))

    wall_ahead = solid_at(ahead_x, feet_y - 8)
    pit_ahead = not solid_at(ahead_x, feet_y)
    enemy_ahead = any(0 < enemy.x - mario.x < 40 and abs(enemy.y - feet_y) < 32
                      for enemy in world.enemies)
    if wall_ahead or pit_ahead or enemy_ahead:
        return RIGHT_A_B
    return RIGHT_B


def run(level: str, seed: int, steps: int, policy: str, capture: bool,
        gif_stride: int, scale: int) -> tuple[list[dict], list[np.ndarray]]:
    """Run a policy, recording extracted state and overlay frames.

    Returns:
        ``(records, frames)``.
    """
    env = make_planner_env(level, render_mode="rgb_array")
    records: list[dict] = []
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=seed)
        info = env._get_info()
        previous: MarioState | None = None
        previous_y_screen: int | None = None

        for step in range(steps):
            world = WorldState.from_env(env, info["flag_get"], info["time"])
            action = RIGHT_B if policy == "right" else reactive_action(world)

            if capture and step % gif_stride == 0:
                frames.append(
                    render_overlay(
                        env.render(), world,
                        hud={"step": step, "policy": policy}, scale=scale,
                    )
                )

            feet_tile = ram_map.tile_at(env.ram, world.mario.x + 8, world.mario_y_screen + 32)
            records.append(
                {
                    "step": step,
                    "x": world.mario.x,
                    "y": world.mario.y,
                    "y_screen": world.mario_y_screen,
                    "vx": world.mario.vx,
                    "vy": world.mario.vy,
                    "on_ground": world.mario.on_ground,
                    "powerup": world.mario.powerup,
                    "is_dying": world.is_dying,
                    "n_enemies": len(world.enemies),
                    "nearest_enemy_dx": min(
                        (enemy.x - world.mario.x for enemy in world.enemies),
                        key=abs, default=None,
                    ),
                    "feet_tile": feet_tile,
                    "dx": None if previous is None else world.mario.x - previous.x,
                    "dy_screen": (
                        None if previous_y_screen is None
                        else world.mario_y_screen - previous_y_screen
                    ),
                    "prev_vx": None if previous is None else previous.vx,
                    "prev_vy": None if previous is None else previous.vy,
                }
            )
            previous, previous_y_screen = world.mario, world.mario_y_screen

            env._frame_advance(action)
            info = env._get_info()
            if world.is_dying:
                break
    finally:
        env.close()
    return records, frames


def verify(records: list[dict], level: str) -> dict[str, dict]:
    """Check the extracted state against independently observable behaviour.

    Returns:
        ``check name -> {passed, detail}``.
    """
    checks: dict[str, dict] = {}

    # --- velocity predicts displacement -------------------------------------------------
    vx_errors = [
        abs(r["dx"] - r["prev_vx"] / ram_map.VX_SUBPIXELS_PER_PIXEL)
        for r in records
        if r["dx"] is not None and abs(r["dx"]) < 100  # skip the page-wrap frame
    ]
    mean_vx_error = float(np.mean(vx_errors)) if vx_errors else float("inf")
    checks["vx predicts dx"] = {
        "passed": mean_vx_error < 1.0,
        "detail": f"mean |dx - vx/16| = {mean_vx_error:.3f} px (sub-pixel accumulation "
                  f"makes a perfect match impossible)",
    }

    vy_errors = [
        abs(r["dy_screen"] - r["prev_vy"])
        for r in records
        if r["dy_screen"] is not None and abs(r["dy_screen"]) < 100
    ]
    mean_vy_error = float(np.mean(vy_errors)) if vy_errors else float("inf")
    checks["vy predicts dy"] = {
        "passed": mean_vy_error < 1.0,
        "detail": f"mean |dy_screen - vy| = {mean_vy_error:.3f} px",
    }

    # --- the tile grid is aligned -------------------------------------------------------
    grounded = [r for r in records if r["on_ground"] and not r["is_dying"]]
    solid_under_feet = [r for r in grounded if ram_map.is_solid(r["feet_tile"])]
    ratio = len(solid_under_feet) / len(grounded) if grounded else 0.0
    checks["ground under feet is solid"] = {
        "passed": ratio > 0.98,
        "detail": f"{len(solid_under_feet)}/{len(grounded)} grounded frames "
                  f"({ratio:.1%}) have a solid tile under Mario's feet",
    }

    # --- on_ground agrees with vertical motion ------------------------------------------
    grounded_moving = [r for r in grounded if r["vy"] != 0]
    checks["on_ground implies vy == 0"] = {
        "passed": len(grounded_moving) / max(len(grounded), 1) < 0.02,
        "detail": f"{len(grounded_moving)}/{len(grounded)} grounded frames have vy != 0",
    }

    # --- enemies are where the collision happens ----------------------------------------
    deaths = [r for r in records if r["is_dying"]]
    if deaths and deaths[0]["nearest_enemy_dx"] is not None:
        gap = abs(deaths[0]["nearest_enemy_dx"])
        checks["enemy adjacent at death"] = {
            "passed": gap < 32,
            "detail": f"nearest enemy was {gap} px away on the death frame",
        }
    else:
        checks["enemy adjacent at death"] = {
            "passed": True,
            "detail": "no death in this run (nothing to check)",
        }

    # --- the tile vocabulary is non-degenerate ------------------------------------------
    tiles_seen = {r["feet_tile"] for r in records}
    checks["tile reads are non-degenerate"] = {
        "passed": len(tiles_seen) > 1,
        "detail": f"distinct tile bytes under feet on {level}: "
                  f"{sorted(hex(t) for t in tiles_seen)}",
    }
    return checks


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config, {"level": args.level, "seed": args.seed})
    level, seed = config["level"], config["seed"]

    print("=" * 72)
    print(f"Phase 2 inspector -- level {level}, policy {args.policy}, seed {seed}")
    print("=" * 72)

    records, frames = run(
        level=level, seed=seed, steps=args.steps, policy=args.policy,
        capture=not args.no_gif, gif_stride=args.gif_stride, scale=args.scale,
    )
    final = records[-1]
    print(f"\nframes run      : {len(records)}")
    print(f"max x reached   : {max(r['x'] for r in records)}")
    print(f"ended dying     : {final['is_dying']}")

    if args.verify:
        print("\nExtraction checks")
        checks = verify(records, level)
        for name, result in checks.items():
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"  [{mark}] {name}")
            print(f"         {result['detail']}")
        failed = [n for n, r in checks.items() if not r["passed"]]
        print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
    else:
        checks = {}

    tag = f"{level}_{args.policy}_seed{seed}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = RAW_DIR / f"phase2_inspect_{tag}.json"
    report.write_text(json.dumps({"checks": checks, "final": final}, indent=2))
    print(f"\nreport          : {report.relative_to(Path.cwd())}")

    if frames:
        import imageio.v2 as imageio

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        gif_path = FIGURES_DIR / f"phase2_overlay_{tag}.gif"
        imageio.mimsave(gif_path, frames, duration=1000.0 / GIF_FPS, loop=0)
        print(f"overlay GIF     : {gif_path.relative_to(Path.cwd())} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
