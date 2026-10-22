"""Running two agents side by side, in lockstep.

The project's headline result is a *behavioural difference under perturbation*: A* open-loop
collapses from 1.00 completion to 0.15 under sticky actions, while the same planner replanning
every 15 frames holds 0.80. That is far more convincing watched than tabulated -- two agents on
the same level, from the same seed, with a slider that introduces the perturbation while they
run.

This module is the engine for that. It deliberately contains no UI code, so the stepping and
bookkeeping can be tested headlessly; the Streamlit app is a thin shell over it.

An honesty note encoded here rather than glossed over
-----------------------------------------------------
The two panels face the **same perturbation process, not the same noise realisation.** Sticky
actions repeat the *previously executed* action, so the override sequence depends on each
agent's own action history. Once two agents act differently -- which is the entire point -- their
override sequences necessarily diverge. Identical realisations are impossible here, not merely
unimplemented. :meth:`ComparisonSession.tick` therefore reports each side's *realised* override
rate, so a viewer can confirm both are being perturbed at the configured rate even though the
individual overrides differ.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.factory import build_agent
from src.viz.live_demo import OVERLAY_MODES, DemoRunner

#: Emulator frames advanced per tick. The NES runs at 60 fps and a browser tick is ~20 Hz, so
#: 3 frames per tick is roughly real time.
DEFAULT_FRAMES_PER_TICK = 3

#: Which side is which, in a fixed order so callers can rely on it.
SIDES: tuple[str, str] = ("left", "right")


class ComparisonSession:
    """Two :class:`DemoRunner` instances advanced together.

    Each side owns a full emulator, so :meth:`close` is **not** optional -- an abandoned session
    leaks a native emulator per side.
    """

    def __init__(
        self,
        left_name: str,
        right_name: str,
        config: dict[str, Any],
        astar_config: dict[str, Any],
        level: str = "1-1",
        sticky_p: float = 0.0,
        seed: int = 0,
        scale: int = 2,
        frames_per_tick: int = DEFAULT_FRAMES_PER_TICK,
    ) -> None:
        """Build both sides.

        Args:
            left_name: agent for the left panel.
            right_name: agent for the right panel.
            config: loaded demo config (plans, closed-loop settings, goals, PPO model).
            astar_config: loaded planner config (primitives, lattice).
            level: ``"world-stage"``; both sides play the same one.
            sticky_p: initial perturbation. Mutable later via :meth:`set_sticky`.
            seed: both sides reset from this seed, so any divergence is the agents', not luck.
            scale: integer upscaling of each panel's frame.
            frames_per_tick: emulator frames advanced per :meth:`tick`.

        Raises:
            FileNotFoundError: if an agent's cached plan or trained model is missing.
        """
        self.level = level
        self.seed = seed
        self.frames_per_tick = frames_per_tick
        self.names = {"left": left_name, "right": right_name}
        self.runners: dict[str, DemoRunner] = {}
        self.ticks = 0

        decision_every = config.get("decision_every", {})
        for side, name in self.names.items():
            agent, plan = build_agent(name, level, config, astar_config)
            runner = DemoRunner(
                agent=agent,
                level=level,
                sticky_p=sticky_p,
                seed=seed,
                scale=scale,
                decision_every=decision_every.get(name, 1),
                lookahead=config.get("lookahead", 90),
                max_frames=config.get("max_frames", 12_000),
                record_path=None,
                headless=True,
            )
            if plan is not None:
                # The open-loop agent can only draw its plan if given the positions it will
                # visit, which means replaying it once through an emulator.
                agent.set_trajectory(_plan_trajectory(plan, level, seed))
            self.runners[side] = runner

        self.reset()

    # ------------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        """Restart both episodes from the same seed."""
        for runner in self.runners.values():
            runner.reset()
        self.ticks = 0

    def close(self) -> None:
        """Release both emulators. Safe to call more than once."""
        for runner in self.runners.values():
            runner.close()
        self.runners = {}

    # -------------------------------------------------------------------------- controls

    def set_sticky(self, probability: float) -> None:
        """Change the perturbation on both sides, mid-episode.

        This is the demonstration: raise it while the agents are running and the open-loop plan
        starts failing within a second or two, while the closed-loop planner visibly redraws its
        path and keeps going.
        """
        for runner in self.runners.values():
            runner.env.sampler.p = float(probability)
            runner.state.sticky_on = probability > 0.0

    def set_overlay(self, index: int) -> None:
        """Set the overlay layer preset on both sides, so the panels stay comparable."""
        for runner in self.runners.values():
            runner.state.overlay_index = index % len(OVERLAY_MODES)

    # ------------------------------------------------------------------------- stepping

    def tick(self) -> dict[str, dict[str, Any]]:
        """Advance both sides by ``frames_per_tick`` frames and return frames plus stats.

        A side whose episode has ended stops advancing but is still rendered, so a dead agent
        freezes on its final frame while the other keeps going -- which is exactly the image the
        comparison is for.

        Returns:
            ``{"left": {"frame": ndarray, "stats": {...}}, "right": {...}}``.
        """
        out: dict[str, dict[str, Any]] = {}
        for side, runner in self.runners.items():
            frame: np.ndarray | None = None
            for _ in range(self.frames_per_tick):
                if runner.state.episode_over:
                    break
                frame = runner.step()
            if frame is None:
                frame = runner.render()
            out[side] = {"frame": frame, "stats": runner.stats(), "agent": self.names[side]}
        self.ticks += 1
        return out

    @property
    def finished(self) -> bool:
        """Whether both episodes have ended."""
        return all(runner.state.episode_over for runner in self.runners.values())


def _plan_trajectory(plan: Any, level: str, seed: int) -> list[tuple[int, int]]:
    """Replay a cached plan once to get the positions it visits, for drawing it on screen."""
    from src.env.factories import make_planner_env
    from src.planning.simulator import ForwardSimulator

    env = make_planner_env(level)
    try:
        simulator = ForwardSimulator(env)
        _, snapshot = simulator.root(seed=seed)
        return simulator.trajectory(snapshot, list(plan.bytes_))
    finally:
        env.close()
