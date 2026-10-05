#!/usr/bin/env python
"""Phase 1 sanity check: drive Super Mario Bros with a random policy.

Purpose
-------
1. Prove the environment runs end-to-end on this machine through the full Gymnasium 5-tuple
   API, and that both env factories (wrapped for RL, raw for the planner) work.
2. Re-measure the emulation throughput that the whole A* design depends on. Phase 0 measured
   ~1,200 frames/s per core; because an A* edge costs one primitive's worth of frames plus a
   snapshot, that number is what forces closed-loop replanning to use a bounded receding
   horizon instead of replanning the whole level. Confirming it here means the Phase 3
   design rests on a measurement reproduced by committed code, not a one-off spike.
3. Produce the project's first artefact: a GIF of a random agent, which is also the
   "do-nothing" baseline that both real agents have to beat.

A random policy is the degenerate planner: it samples controls with no model of the
dynamics and no lookahead. It establishes how far an agent gets on momentum alone -- in
practice it dies on the first Goomba at x~315 -- which is the floor for Phase 5's table.

Usage
-----
    .venv/bin/python scripts/phase1_sanity.py
    .venv/bin/python scripts/phase1_sanity.py --level 1-2 --steps 1500 --sticky 0.25
    .venv/bin/python scripts/phase1_sanity.py --no-gif        # fast, CI-friendly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RAW_DIR, load_config  # noqa: E402
from src.env.actions import RIGHT_B, name_for_action  # noqa: E402
from src.env.factories import make_planner_env, make_rl_env  # noqa: E402

#: Frames of the GIF per second. The NES runs at ~60 fps; 30 keeps the file small.
GIF_FPS = 30

#: Frames to emulate when benchmarking the raw planner env.
BENCHMARK_FRAMES = 500


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Unset flags fall back to the YAML config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="env.yaml", help="YAML in configs/ (default: env.yaml)")
    parser.add_argument("--level", help="override config: 'world-stage', e.g. 1-1")
    parser.add_argument("--seed", type=int, help="override config: RNG seed")
    parser.add_argument(
        "--sticky",
        type=float,
        dest="sticky_p",
        help="override config: sticky-action probability (0.0 or 0.25)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="total agent steps (default: 1000)")
    parser.add_argument("--gif", type=Path, help="GIF output path (default: results/figures/...)")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF writing")
    parser.add_argument(
        "--gif-stride",
        type=int,
        default=2,
        help="keep every Nth frame in the GIF (default: 2)",
    )
    return parser.parse_args()


def benchmark_planner_env(level: str, n_frames: int = BENCHMARK_FRAMES) -> dict[str, float]:
    """Measure raw emulator throughput and snapshot cost on the planner's env.

    These three numbers are the Phase 3 cost model: an A* edge is
    ``primitive_length * frame_ms + dump_ms + load_ms``.

    Args:
        level: level to benchmark on.
        n_frames: frames to emulate for the throughput estimate.

    Returns:
        Dict with ``frame_ms``, ``frames_per_s``, ``dump_ms``, ``load_ms``.
    """
    env = make_planner_env(level)
    try:
        env.reset(seed=0)
        start = time.perf_counter()
        for _ in range(n_frames):
            env._frame_advance(RIGHT_B)
        frame_s = (time.perf_counter() - start) / n_frames

        # Snapshot round-trip cost, averaged over a handful of calls.
        n_snap = 200
        start = time.perf_counter()
        snapshots = [env.dump_state() for _ in range(n_snap)]
        dump_s = (time.perf_counter() - start) / n_snap
        start = time.perf_counter()
        for snapshot in snapshots:
            env.load_state(snapshot)
        load_s = (time.perf_counter() - start) / n_snap
    finally:
        env.close()

    return {
        "frame_ms": frame_s * 1e3,
        "frames_per_s": 1.0 / frame_s,
        "dump_ms": dump_s * 1e3,
        "load_ms": load_s * 1e3,
    }


def run_random_policy(
    level: str,
    seed: int,
    sticky_p: float,
    max_episode_steps: int,
    render_mode: str,
    steps: int,
    capture_frames: bool,
    gif_stride: int,
) -> tuple[list[dict], list[np.ndarray], dict]:
    """Act uniformly at random through the wrapped RL env for a fixed step budget.

    Episodes that end (death, timeout or flag) are reset and the budget continues, so the
    run reports a death *rate* rather than a single anecdote.

    Args:
        level: level to play.
        seed: seed for both the env and the action sampler.
        sticky_p: sticky-action probability.
        max_episode_steps: truncation limit, in agent steps.
        render_mode: passed to the factory; must be ``"rgb_array"`` to capture frames.
        steps: total agent steps across all episodes.
        capture_frames: whether to collect frames for the GIF.
        gif_stride: keep every Nth frame when capturing.

    Returns:
        ``(records, frames, summary)`` -- per-step log rows, captured RGB frames, and
        aggregate statistics.
    """
    env = make_rl_env(
        level=level,
        sticky_p=sticky_p,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    frames: list[np.ndarray] = []

    episode = 0
    deaths = 0
    flags = 0
    truncations = 0
    episode_max_x: list[int] = []
    # Accumulated across episodes: the sampler's counters are cleared by its own reset(), so
    # reading them only at the end would report 0 whenever the run ends on an episode
    # boundary -- which it does whenever the step budget equals max_episode_steps.
    sticky_commands = 0
    sticky_overrides = 0

    try:
        _, info = env.reset(seed=seed)
        episode_start_step = 0
        best_x = int(info["x_pos"])
        wall_start = time.perf_counter()

        for step in range(steps):
            action = int(rng.integers(env.action_space.n))
            _, reward, terminated, truncated, info = env.step(action)
            best_x = max(best_x, int(info["x_pos"]))

            records.append(
                {
                    "step": step,
                    "episode": episode,
                    "action": action,
                    "action_name": name_for_action(action),
                    "reward": float(reward),
                    "x_pos": int(info["x_pos"]),
                    "y_pos": int(info["y_pos"]),
                    "player_state": int(info["player_state"]),
                    "is_dying": bool(info["is_dying"]),
                    "flag_get": bool(info["flag_get"]),
                    "time": int(info["time"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if capture_frames and step % gif_stride == 0:
                frame = env.render()
                if frame is not None:
                    # `render()` returns a *view* into the emulator's screen buffer and
                    # hands back the same array object every call, so it must be copied.
                    # Without the copy every captured frame aliases the final screen and
                    # the GIF collapses to a single image. Phase 5b's mp4 recorder has to
                    # do the same.
                    frames.append(np.array(frame, copy=True))

            if terminated or truncated:
                # `flag_get` is the only success signal; `is_dying` distinguishes a death
                # from the in-game timer running out. x_pos freezes during the death
                # animation (Phase 0, Finding 6), so never infer death from position.
                if info["flag_get"]:
                    flags += 1
                elif info["is_dying"] or info["player_state"] == 11:
                    deaths += 1
                else:
                    truncations += 1
                episode_max_x.append(best_x)
                episode += 1
                episode_start_step = step + 1
                if sticky_p > 0.0:
                    sticky_commands += env.sampler.n_commands
                    sticky_overrides += env.sampler.n_overridden
                _, info = env.reset()
                best_x = int(info["x_pos"])

        wall_s = time.perf_counter() - wall_start
        if episode_start_step < steps:  # final, unfinished episode
            episode_max_x.append(best_x)
        if sticky_p > 0.0:
            # Fold in the final, unfinished episode.
            sticky_commands += env.sampler.n_commands
            sticky_overrides += env.sampler.n_overridden
    finally:
        env.close()

    summary = {
        "level": level,
        "seed": seed,
        "sticky_p": sticky_p,
        "steps": steps,
        "episodes": len(episode_max_x),
        "deaths": deaths,
        "flags": flags,
        "timeouts_or_truncations": truncations,
        "max_x_overall": max(episode_max_x) if episode_max_x else 0,
        "mean_episode_max_x": float(np.mean(episode_max_x)) if episode_max_x else 0.0,
        "mean_episode_length": steps / len(episode_max_x) if episode_max_x else float(steps),
        "wall_s": wall_s,
        "steps_per_s": steps / wall_s,
        "sticky_realised": (
            sticky_overrides / sticky_commands if sticky_commands else None
        ),
    }
    return records, frames, summary


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write per-step records as JSON Lines (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_gif(frames: list[np.ndarray], path: Path, fps: int = GIF_FPS) -> None:
    """Write captured RGB frames to an animated GIF.

    Args:
        frames: RGB arrays. Must be independent copies, not views into the emulator's
            screen buffer -- see the note in :func:`run_random_policy`.
        path: output path; parent directories are created.
        fps: playback rate. Converted to a per-frame duration in milliseconds because
            imageio's ``fps`` argument is deprecated.
    """
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)


def main() -> None:
    """Entry point: benchmark, run the random policy, write artefacts, print a summary."""
    args = parse_args()
    config = load_config(
        args.config,
        {"level": args.level, "seed": args.seed, "sticky_p": args.sticky_p},
    )
    level = config["level"]

    print("=" * 72)
    print(f"Phase 1 sanity check -- level {level}, seed {config['seed']}, "
          f"sticky_p {config['sticky_p']}")
    print("=" * 72)

    print("\n[1/3] Raw emulator benchmark (the planner's forward simulator)")
    bench = benchmark_planner_env(level)
    print(f"  frame_advance : {bench['frame_ms']:.3f} ms  ({bench['frames_per_s']:,.0f} frames/s)")
    print(f"  dump_state    : {bench['dump_ms']:.3f} ms")
    print(f"  load_state    : {bench['load_ms']:.3f} ms")
    edge_ms = 12 * bench["frame_ms"] + bench["dump_ms"] + bench["load_ms"]
    print(f"  => a 12-frame A* edge costs ~{edge_ms:.1f} ms  (~{1e3 / edge_ms:.0f} edges/s)")
    print("     Closed-loop replanning must therefore use a bounded receding horizon.")

    print(f"\n[2/3] Random policy, {args.steps} agent steps")
    records, frames, summary = run_random_policy(
        level=level,
        seed=config["seed"],
        sticky_p=config["sticky_p"],
        max_episode_steps=config["max_episode_steps"],
        render_mode=config["render_mode"],
        steps=args.steps,
        capture_frames=not args.no_gif,
        gif_stride=args.gif_stride,
    )
    print(f"  episodes      : {summary['episodes']}")
    print(f"  deaths        : {summary['deaths']}")
    print(f"  flags reached : {summary['flags']}")
    print(f"  truncated     : {summary['timeouts_or_truncations']}")
    print(f"  max x_pos     : {summary['max_x_overall']}"
          f"  (mean per episode {summary['mean_episode_max_x']:.1f})")
    print(f"  throughput    : {summary['steps_per_s']:,.0f} agent steps/s")

    print("\n[3/3] Artefacts")
    tag = f"{level}_sticky{config['sticky_p']:.2f}_seed{config['seed']}"
    jsonl_path = RAW_DIR / f"phase1_random_{tag}.jsonl"
    write_jsonl(records, jsonl_path)
    print(f"  per-step log  : {jsonl_path.relative_to(Path.cwd())}")

    if args.no_gif:
        print("  GIF           : skipped (--no-gif)")
    elif not frames:
        print("  GIF           : no frames captured (is render_mode 'rgb_array'?)")
    else:
        gif_path = args.gif or FIGURES_DIR / f"phase1_random_{tag}.gif"
        write_gif(frames, gif_path)
        print(f"  GIF           : {gif_path.relative_to(Path.cwd())} ({len(frames)} frames)")

    summary_path = RAW_DIR / f"phase1_summary_{tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"summary": summary, "benchmark": bench}, indent=2))
    print(f"  summary       : {summary_path.relative_to(Path.cwd())}")
    print("\nDone.")


if __name__ == "__main__":
    main()
