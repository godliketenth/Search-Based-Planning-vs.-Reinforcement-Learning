"""Building the parallel training environment.

Phase 0 found that ``nes_py``'s ``NativeVectorEmulator`` is *not* parallel -- it is a serial
C++ loop, flat at ~1,200 frames/s from 1 to 24 envs -- so parallelism has to come from real
OS processes via SB3's ``SubprocVecEnv``.

Measured scaling on an M1 Max (10 cores), with preprocessing inside each worker:

===========  ==========  =========
workers      frames/s    speedup
===========  ==========  =========
1            1,175       1.00x
4            4,204       3.58x
**8**        **7,679**   **6.54x**
10           6,382       5.43x
12           6,360       5.41x
===========  ==========  =========

Two things that table decides:

* **8 workers, not 10.** The machine has 10 cores but the main process needs one for the
  PPO update, so oversubscribing *loses* throughput.
* **Preprocess in the worker.** Shipping raw RGB instead caps 8-worker throughput at 2,823
  frames/s (2.69x). The pipe, not the emulator, is the bottleneck.

Frame stacking is applied at the **vector** level rather than per-env, so one 6.9 KB frame
crosses the pipe per step instead of a 27.6 KB stack.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from src.env.factories import make_rl_env
from src.rl.wrappers import DEFAULT_FRAME_SIZE, DEFAULT_SKIP, EpisodeStatistics, FrameSkip, NativeGrayResize

#: Worker count that maximised throughput on the development machine.
DEFAULT_N_ENVS = 8

#: Frames stacked to give the policy a sense of velocity. A single frame cannot show
#: whether Mario is rising or falling, which matters enormously here -- the same reason the
#: planner's C-space includes velocity.
DEFAULT_N_STACK = 4


def make_env_fn(
    level: str = "1-1",
    sticky_p: float = 0.0,
    seed: int = 0,
    rank: int = 0,
    max_episode_steps: int | None = None,
    skip: int = DEFAULT_SKIP,
    frame_size: int = DEFAULT_FRAME_SIZE,
    monitor: bool = True,
) -> Callable[[], gym.Env]:
    """Build a thunk that constructs one fully wrapped training environment.

    The wrapper order matters:

    1. ``make_rl_env`` -- ``JoypadSpace`` + ``TimeLimit`` + optional sticky actions;
    2. :class:`~src.rl.wrappers.FrameSkip` -- **outside** sticky, so each held frame is
       perturbed independently (frame-level perturbation, matching the planner's executor);
    3. :class:`~src.rl.wrappers.EpisodeStatistics` -- outcome fields for the metrics;
    4. :class:`~src.rl.wrappers.NativeGrayResize` -- the 26x reduction in pipe traffic;
    5. ``Monitor`` -- SB3's episode return/length logging.

    Args:
        level: ``"world-stage"``.
        sticky_p: sticky-action probability.
        seed: base seed; each worker uses ``seed + rank`` so they explore differently.
        rank: worker index.
        max_episode_steps: truncation limit in agent steps, or ``None``.
        skip: frames per agent step.
        frame_size: observation edge length.
        monitor: wrap with SB3's ``Monitor``.

    Returns:
        A zero-argument callable, as ``SubprocVecEnv`` requires.
    """

    def _init() -> gym.Env:
        env = make_rl_env(
            level=level,
            sticky_p=sticky_p,
            max_episode_steps=max_episode_steps,
            seed=seed + rank,
        )
        env = FrameSkip(env, skip=skip)
        env = EpisodeStatistics(env)
        env = NativeGrayResize(env, size=frame_size)
        if monitor:
            env = Monitor(env)
        return env

    return _init


def make_training_vec_env(
    n_envs: int = DEFAULT_N_ENVS,
    level: str = "1-1",
    sticky_p: float = 0.0,
    seed: int = 0,
    max_episode_steps: int | None = None,
    skip: int = DEFAULT_SKIP,
    frame_size: int = DEFAULT_FRAME_SIZE,
    n_stack: int = DEFAULT_N_STACK,
    force_dummy: bool = False,
) -> VecEnv:
    """Build the vectorised environment PPO trains against.

    Args:
        n_envs: parallel workers. See the module docstring for why 8 is the default.
        level: ``"world-stage"``.
        sticky_p: sticky-action probability.
        seed: base seed.
        max_episode_steps: truncation limit in agent steps.
        skip: frames per agent step.
        frame_size: observation edge length.
        n_stack: frames stacked at the vector level.
        force_dummy: use ``DummyVecEnv`` (single process). Faster to start, so tests use it;
            also the automatic choice for ``n_envs == 1``, where a subprocess only adds IPC.

    Returns:
        A channels-first, frame-stacked :class:`VecEnv`.
    """
    env_fns = [
        make_env_fn(
            level=level, sticky_p=sticky_p, seed=seed, rank=rank,
            max_episode_steps=max_episode_steps, skip=skip, frame_size=frame_size,
        )
        for rank in range(n_envs)
    ]
    vec_cls = DummyVecEnv if (force_dummy or n_envs == 1) else SubprocVecEnv
    venv: VecEnv = vec_cls(env_fns)
    venv = VecFrameStack(venv, n_stack=n_stack)
    # NatureCNN wants (C, H, W); the wrappers produce (H, W, C).
    venv = VecTransposeImage(venv)
    return venv
