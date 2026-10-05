"""A* executed open-loop: replay a cached plan, blind.

An open-loop plan is a memorised button sequence. It is executed with **no feedback at all**
-- the agent never looks at the observation, because by construction it already knows what
the world will do. On a deterministic emulator that is exact, and Phase 3 verified it:
replaying the cached 1056-frame plan from ``reset()`` reaches the flagpole every time.

That is also precisely why this agent is in the comparison. "A* completes the level 100% of
the time" is a statement about the emulator's determinism, not about planning. The moment
anything perturbs execution -- here, sticky actions -- the plan desynchronises from reality
and cannot recover, because there is nothing in the loop to notice. The gap between this
agent and :mod:`src.agents.astar_closed_loop` is the project's central measurement.

Plans are loaded from disk as **action sequences**. Emulator snapshots are not picklable, and
a sequence replayed from ``reset()`` is a complete description of a deterministic trajectory.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.base import AgentDiagnostics, Observation
from src.env.actions import NOOP, action_for_byte
from src.planning.plan import Plan


class AStarOpenLoopAgent:
    """Replays a precomputed plan, one action per frame, ignoring the world."""

    name = "astar_open_loop"

    def __init__(self, plan: Plan | str | Path, fallback_action: int | None = None) -> None:
        """Load the plan to replay.

        Args:
            plan: a :class:`Plan`, or a path to one saved as JSON.
            fallback_action: what to do once the plan is exhausted. Defaults to NOOP --
                deliberately inert, so that a plan running out is visible in the metrics as
                a stall rather than disguised by some reactive fallback.
        """
        self.plan = plan if isinstance(plan, Plan) else Plan.load(plan)
        self._actions = self.plan.actions
        self._fallback = (
            action_for_byte(NOOP) if fallback_action is None else fallback_action
        )
        self._cursor = 0
        self._trajectory: list[tuple[int, int]] = []
        self.diagnostics = AgentDiagnostics()

    def set_trajectory(self, points: list[tuple[int, int]]) -> None:
        """Attach the plan's predicted positions, for drawing it on screen.

        Supplied by the caller rather than computed here, because producing it means
        replaying the whole plan through an emulator and the agent does not own one. The
        live demo computes it once at startup.

        Args:
            points: ``(x_level, y_screen)`` per frame of the plan.
        """
        self._trajectory = list(points)

    def planned_path(self, lookahead: int = 90) -> list[tuple[int, int]]:
        """The not-yet-executed part of the plan, for the overlay.

        Args:
            lookahead: how many frames ahead to return.

        Returns:
            ``(x_level, y_screen)`` waypoints, empty if no trajectory was attached.
        """
        return self._trajectory[self._cursor:self._cursor + lookahead]

    def reset(self) -> None:
        """Rewind to the start of the plan."""
        self._cursor = 0
        self.diagnostics.reset()
        # The search was done once, offline; the episode itself costs no planning.
        self.diagnostics.planning_calls = 1
        self.diagnostics.planning_seconds = self.plan.stats.wall_s
        self.diagnostics.nodes_expanded = self.plan.stats.nodes_expanded
        self.diagnostics.frames_simulated = self.plan.stats.frames_simulated
        self.diagnostics.notes["plan_frames"] = len(self.plan)
        self.diagnostics.notes["plan_exhausted"] = False

    def act(self, observation: Observation) -> int:
        """Return the next action in the plan, regardless of what the world is doing."""
        if self._cursor >= len(self._actions):
            self.diagnostics.notes["plan_exhausted"] = True
            return self._fallback
        action = self._actions[self._cursor]
        self._cursor += 1
        return action
