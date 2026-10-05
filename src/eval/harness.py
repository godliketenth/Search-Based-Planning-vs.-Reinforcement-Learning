"""The evaluation harness: one loop, every agent.

The loop below is the project's fairness guarantee. It constructs the environment, steps it,
counts frames, detects termination and records metrics -- identically for every agent. An
agent's only influence is the integer it returns from ``act``.

Environment construction detail that matters
--------------------------------------------
Agents are evaluated **frame by frame**, not at frame-skip 4. The planner's plans are
per-frame byte sequences, so evaluating it at skip 4 would quantise its control and
handicap it. PPO was trained at skip 4, so it is given a fresh decision every 4 frames and
its action is held in between -- which is exactly the behaviour it was trained under. Both
therefore act at the frequency they were designed for, on the same clock, with the same
sticky perturbation applied per frame.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import numpy as np

from src.agents.base import Observation
from src.env import ram_map
from src.env.factories import make_rl_env
from src.eval.metrics import EpisodeResult, classify_death
from src.rl.wrappers import NativeGrayResize

#: Name used in results for the unperturbed condition.
CONDITION_DETERMINISTIC = "deterministic"


def condition_name(sticky_p: float) -> str:
    """Human-readable condition label for a sticky probability."""
    return CONDITION_DETERMINISTIC if sticky_p <= 0.0 else f"sticky_{sticky_p:g}"


class _PixelStack:
    """Maintains the 4-frame 84x84 stack PPO expects, in channels-first order.

    Rebuilt here rather than reused from the training vec-env because the harness steps one
    env frame by frame; ``VecFrameStack`` only exists at the vector level.

    **Temporal spacing matters and is easy to get wrong.** During training the stack is fed
    by a frame-skip-4 env, so consecutive slots are *four game frames* apart and the stack
    encodes roughly 16 frames of motion. The harness steps frame by frame, so it must push
    only on decision frames; pushing every frame yields four *consecutive* frames, a quarter
    of the expected motion between slots, and the policy -- which reads velocity from that
    spacing -- degrades badly without any error being raised.
    """

    def __init__(self, env: Any, n_stack: int = 4, size: int = 84) -> None:
        self._resizer = NativeGrayResize(env, size=size)
        self._frames = np.zeros((n_stack, size, size), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        """Zero the stack and push the current frame, matching VecFrameStack's reset."""
        self._frames[:] = 0
        return self.push()

    def push(self) -> np.ndarray:
        """Shift the stack and append the current frame. Newest is last."""
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = self._resizer.observation(None)[:, :, 0]
        return self._frames.copy()


def run_episode(
    agent: Any,
    level: str,
    sticky_p: float,
    seed: int,
    episode: int,
    max_frames: int = 12_000,
    decision_every: int = 1,
    render_mode: str | None = None,
    on_frame: Callable[[int, dict], None] | None = None,
) -> EpisodeResult:
    """Run one episode and return its metrics.

    Args:
        agent: anything satisfying the :class:`~src.agents.base.Agent` protocol.
        level: ``"world-stage"``.
        sticky_p: sticky-action probability for this condition.
        seed: reset seed.
        episode: index within the cell, recorded in the result.
        max_frames: hard cap, so a stalled agent terminates.
        decision_every: frames between agent decisions. ``1`` for the planner (its plans are
            per-frame), ``4`` for PPO (its training frame-skip). The action is held in
            between, so both face the same clock.
        render_mode: passed to the env factory.
        on_frame: optional callback for the live demo.

    Returns:
        The episode's :class:`EpisodeResult`.
    """
    env = make_rl_env(
        level=level, sticky_p=sticky_p, max_episode_steps=max_frames + 10,
        render_mode=render_mode, seed=seed,
    )
    try:
        _, info = env.reset(seed=seed)
        stack = _PixelStack(env)
        pixels = stack.reset()
        agent.reset()

        max_x = int(info.get("x_pos", 0))
        total_reward = 0.0
        latencies: list[float] = []
        action = 0
        frame = 0
        flag_get = False
        terminated = truncated = False

        while frame < max_frames:
            if frame % decision_every == 0:
                started = time.perf_counter()
                action = agent.act(
                    Observation(pixels=pixels, env=env, frame=frame, info=info)
                )
                latencies.append((time.perf_counter() - started) * 1e3)

            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            frame += 1
            if frame % decision_every == 0:
                # Advance the stack once per decision, so slot spacing matches training.
                pixels = stack.push()
            max_x = max(max_x, int(info.get("x_pos", 0)))
            flag_get = flag_get or bool(info.get("flag_get", False))

            if on_frame is not None:
                on_frame(frame, info)
            if terminated or truncated:
                break

        ram = env.unwrapped.ram
        death_cause = classify_death(ram, info, flag_get)
        diagnostics = getattr(agent, "diagnostics", None)
        sticky_realised = (
            env.sampler.realised_rate if hasattr(env, "sampler") else None
        )

        return EpisodeResult(
            agent=agent.name,
            level=level,
            condition=condition_name(sticky_p),
            episode=episode,
            seed=seed,
            completed=flag_get,
            max_x=max_x,
            frames=frame,
            death_cause=death_cause,
            reward=total_reward,
            decisions=len(latencies),
            decision_ms_mean=float(np.mean(latencies)) if latencies else 0.0,
            decision_ms_max=float(np.max(latencies)) if latencies else 0.0,
            planning_calls=getattr(diagnostics, "planning_calls", 0),
            planning_seconds=getattr(diagnostics, "planning_seconds", 0.0),
            nodes_expanded=getattr(diagnostics, "nodes_expanded", 0),
            compute_frames=getattr(diagnostics, "frames_simulated", 0),
            sticky_realised=sticky_realised,
            notes=dict(getattr(diagnostics, "notes", {})),
        )
    finally:
        env.close()


def run_cell(
    agent_factory: Callable[[], Any],
    level: str,
    sticky_p: float,
    n_episodes: int,
    base_seed: int = 0,
    decision_every: int = 1,
    max_frames: int = 12_000,
    verbose: bool = True,
) -> list[EpisodeResult]:
    """Run one (agent, level, condition) cell of the experimental grid.

    A fresh agent is built per episode via ``agent_factory``, so no state leaks between
    episodes -- a stale plan queue or policy state would quietly corrupt the next result.

    Args:
        agent_factory: builds the agent.
        level: ``"world-stage"``.
        sticky_p: sticky-action probability.
        n_episodes: episodes to run.
        base_seed: seeds are ``base_seed + episode``.
        decision_every: frames between decisions.
        max_frames: per-episode frame cap.
        verbose: print a line per episode.

    Returns:
        One :class:`EpisodeResult` per episode.
    """
    results: list[EpisodeResult] = []
    for episode in range(n_episodes):
        agent = agent_factory()
        result = run_episode(
            agent=agent, level=level, sticky_p=sticky_p, seed=base_seed + episode,
            episode=episode, decision_every=decision_every, max_frames=max_frames,
        )
        results.append(result)
        if verbose:
            status = "FLAG" if result.completed else result.death_cause.upper()
            print(
                f"    ep {episode:2d}  x={result.max_x:5d}  frames={result.frames:5d}  "
                f"{status:8s}  {result.decision_ms_mean:6.2f} ms/decision",
                flush=True,
            )
    return results


def write_results_csv(results: Iterable[EpisodeResult], path: Any) -> None:
    """Write per-episode results as a tidy CSV, one row per episode."""
    import csv
    from pathlib import Path

    results = list(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_row() for result in results]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: list[dict], path: Any) -> None:
    """Write the aggregated summary table."""
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
