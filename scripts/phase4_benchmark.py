#!/usr/bin/env python
"""Phase 4 benchmarks: worker scaling and MPS vs CPU.

Two numbers the project promised to measure rather than assume:

1. **How far ``SubprocVecEnv`` actually scales.** Phase 0 predicted roughly linear scaling
   to 8 workers. It does not, unless observations are reduced *before* they cross the pipe
   -- a raw 184 KB RGB frame per env per step makes IPC, not emulation, the bottleneck.
2. **Whether MPS beats CPU for an 84x84 NatureCNN.** The expectation was that it would not:
   the kernels are tiny, launch overhead dominates, and the real bottleneck is
   single-threaded emulation. This measures it.

Usage
-----
    .venv/bin/python scripts/phase4_benchmark.py
    .venv/bin/python scripts/phase4_benchmark.py --skip-scaling    # device benchmark only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR  # noqa: E402
from src.rl.vec_env import make_training_vec_env  # noqa: E402

#: Steps per worker-count measurement, after a warm-up step.
SCALING_STEPS = 300

#: Forward/backward passes for the device benchmark.
DEVICE_ITERS = 60


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-device", action="store_true")
    return parser.parse_args()


def benchmark_scaling(worker_counts: list[int], steps: int = SCALING_STEPS) -> list[dict]:
    """Measure throughput against worker count.

    Returns:
        One row per worker count, with agent steps/s and emulated frames/s.
    """
    rows: list[dict] = []
    baseline: float | None = None
    for n_envs in worker_counts:
        venv = make_training_vec_env(n_envs=n_envs, max_episode_steps=100_000)
        try:
            venv.reset()
            actions = np.full(n_envs, 3)
            venv.step(actions)  # warm up; excluded from the timing
            started = time.perf_counter()
            for _ in range(steps):
                venv.step(actions)
            elapsed = time.perf_counter() - started
        finally:
            venv.close()

        agent_steps_per_s = steps * n_envs / elapsed
        frames_per_s = agent_steps_per_s * 4
        baseline = baseline or frames_per_s
        rows.append(
            {
                "n_envs": n_envs,
                "agent_steps_per_s": agent_steps_per_s,
                "frames_per_s": frames_per_s,
                "speedup": frames_per_s / baseline,
            }
        )
        print(f"  {n_envs:3d} workers: {agent_steps_per_s:7.0f} agent steps/s  "
              f"{frames_per_s:8.0f} frames/s  {frames_per_s / baseline:5.2f}x", flush=True)
    return rows


def benchmark_devices(batch_size: int, iters: int = DEVICE_ITERS) -> list[dict]:
    """Time a NatureCNN forward+backward pass on each available device.

    Uses SB3's actual feature extractor on the actual observation shape, so the number
    describes this project rather than a generic benchmark.

    Returns:
        One row per device.
    """
    from stable_baselines3.common.torch_layers import NatureCNN
    import gymnasium as gym

    space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    rows: list[dict] = []
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        extractor = NatureCNN(space, features_dim=512).to(device)
        optimiser = torch.optim.Adam(extractor.parameters(), lr=1e-4)
        batch = torch.rand(batch_size, 4, 84, 84, device=device)

        for _ in range(5):  # warm up: first MPS call compiles kernels
            optimiser.zero_grad()
            extractor(batch).sum().backward()
            optimiser.step()
        if device == "mps":
            torch.mps.synchronize()

        started = time.perf_counter()
        for _ in range(iters):
            optimiser.zero_grad()
            extractor(batch).sum().backward()
            optimiser.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "device": device,
                "ms_per_update": elapsed / iters * 1e3,
                "updates_per_s": iters / elapsed,
            }
        )
        print(f"  {device:5s}: {elapsed / iters * 1e3:7.2f} ms/update  "
              f"({iters / elapsed:6.1f} updates/s)", flush=True)
    return rows


def main() -> None:
    """Entry point."""
    args = parse_args()
    results: dict = {}

    print("=" * 72)
    print("Phase 4 benchmarks")
    print("=" * 72)

    if not args.skip_scaling:
        print("\n[1/2] SubprocVecEnv scaling (preprocessing inside each worker)")
        results["scaling"] = benchmark_scaling(args.workers)
        best = max(results["scaling"], key=lambda row: row["frames_per_s"])
        print(f"\n  best: {best['n_envs']} workers at {best['frames_per_s']:,.0f} frames/s "
              f"({best['speedup']:.2f}x)")

    if not args.skip_device:
        print(f"\n[2/2] NatureCNN forward+backward, batch {args.batch_size}")
        results["devices"] = benchmark_devices(args.batch_size)
        fastest = min(results["devices"], key=lambda row: row["ms_per_update"])
        print(f"\n  fastest: {fastest['device']} at {fastest['ms_per_update']:.2f} ms/update")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "phase4_benchmark.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
