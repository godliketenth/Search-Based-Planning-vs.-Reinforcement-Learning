"""A* executed closed-loop: replan from where Mario actually is.

This is the agent that makes the comparison a comparison. It runs a **receding horizon**:
every ``replan_every`` frames it looks at the real state, plans a short way ahead, executes
that, and throws the rest away. Perturbation is therefore corrected within one replan
interval instead of compounding forever.

How a replan works, and why it is safe
--------------------------------------
Phase 0 (Finding 4) established that a wrapped Gymnasium env cannot be stepped after
``load_state`` -- wrapper bookkeeping assumes time moves forward. But Phase 0 also found
snapshots are **portable across emulator instances**. So:

1. snapshot the **live** env (``dump_state``), which is only ever read, never restored into;
2. ``load_state`` that snapshot into a **separate planner env**;
3. search there, bounded;
4. return the resulting byte sequence to execute on the live env.

The live env's wrappers stay consistent because nothing ever rewinds it.

Why the horizon is bounded
--------------------------
A full-level replan costs ~54 s (Phase 3) and is unaffordable mid-episode. A bounded
horizon costs 84-500 ms depending on depth, because the heuristic drives almost straight at
the waypoint and only a handful of nodes are expanded. That is the practical reason
receding-horizon control exists, measured rather than asserted.
"""

from __future__ import annotations

import time
from typing import Any

from src.agents.base import AgentDiagnostics, Observation
from src.env import ram_map
from src.env.actions import NOOP, action_for_byte
from src.env.state import MarioState
from src.planning.astar import SearchBudget, plan_level
from src.planning.heuristic import DistanceHeuristic
from src.planning.primitives import MotionPrimitive
from src.planning.simulator import ForwardSimulator


class AStarClosedLoopAgent:
    """Receding-horizon A*: replan periodically from the observed state.

    Attributes:
        replan_every: frames between replans. Smaller corrects perturbation sooner but
            spends more compute; this is the control-frequency trade-off, made explicit.
        horizon_px: how far ahead each search plans, in pixels of level x.
    """

    name = "astar_closed_loop"

    def __init__(
        self,
        planner_env: Any,
        primitives: tuple[MotionPrimitive, ...],
        goal_x: int,
        replan_every: int = 30,
        horizon_px: int = 160,
        weight: float = 1.0,
        max_nodes: int = 600,
        max_seconds: float = 2.0,
        lattice: dict[str, int] | None = None,
        fallback_action: int | None = None,
    ) -> None:
        """Build the executor.

        Args:
            planner_env: a **separate** raw planner env used only for search. Must not be
                the env the harness is stepping.
            primitives: the motion-primitive set.
            goal_x: level x of the goal, for the heuristic.
            replan_every: frames between replans.
            horizon_px: lookahead distance per replan.
            weight: heuristic inflation for the bounded searches.
            max_nodes: per-replan node budget.
            max_seconds: per-replan wall-clock budget. A replan that overruns leaves the
                previous plan in place rather than stalling the episode.
            lattice: lattice resolution overrides.
            fallback_action: action used when no plan is available at all.
        """
        self.planner_env = planner_env
        self.simulator = ForwardSimulator(planner_env)
        self.primitives = primitives
        self.goal_x = goal_x
        self.replan_every = replan_every
        self.horizon_px = horizon_px
        self.weight = weight
        self.budget = SearchBudget(
            max_nodes=max_nodes, max_seconds=max_seconds, max_frontier=2_000
        )
        self.lattice = lattice or {}
        self._fallback = (
            action_for_byte(NOOP) if fallback_action is None else fallback_action
        )
        self._queue: list[int] = []
        self._frames_since_replan = 0
        self._path: list[tuple[int, int]] = []
        self.replanned_this_frame = False
        self.diagnostics = AgentDiagnostics()

    def planned_path(self, lookahead: int = 90) -> list[tuple[int, int]]:
        """The current horizon's predicted trajectory, for the overlay.

        Computed once per replan rather than per frame: it costs one re-simulation of the
        (short) plan, which is cheap beside the search that just produced it.
        """
        consumed = len(self._path) - len(self._queue)
        return self._path[max(0, consumed):max(0, consumed) + lookahead]

    def reset(self) -> None:
        """Drop any held plan and zero the counters."""
        self._queue = []
        self._path = []
        self.replanned_this_frame = False
        self._frames_since_replan = self.replan_every  # force a replan on the first act
        self.diagnostics.reset()
        self.diagnostics.notes["failed_replans"] = 0
        self.simulator.frames_simulated = 0

    def act(self, observation: Observation) -> int:
        """Return the next action, replanning first if the horizon has elapsed.

        The replan interval is honoured **even when planning fails**. An earlier version also
        replanned whenever the queue was empty, which looks harmless but is not: on a level
        the planner cannot solve, the queue is always empty, so that condition fires every
        frame and each attempt burns the full search budget. One episode then costs hours
        instead of minutes. A failed replan must back off like any other.
        """
        self.replanned_this_frame = False
        if self._frames_since_replan >= self.replan_every:
            self._replan(observation)
        self._frames_since_replan += 1

        if not self._queue:
            return self._fallback
        return action_for_byte(self._queue.pop(0))

    def _replan(self, observation: Observation) -> None:
        """Search a bounded horizon forward from the live environment's current state."""
        live = observation.env.unwrapped
        if ram_map.read_is_dying(live.ram):
            # Nothing to plan from: the trajectory has already failed.
            return

        snapshot = live.dump_state()
        self.planner_env.load_state(snapshot)
        state = MarioState.from_env(self.planner_env)
        target = min(state.x + self.horizon_px, self.goal_x)

        started = time.perf_counter()
        frames_before = self.simulator.frames_simulated
        plan, stats, _ = plan_level(
            self.simulator,
            self.primitives,
            DistanceHeuristic(goal_x=self.goal_x),
            goal_x=target,
            weight=self.weight,
            budget=self.budget,
            lattice=self.lattice,
            start=(state, snapshot),
        )
        elapsed = time.perf_counter() - started

        self.diagnostics.planning_calls += 1
        self.diagnostics.planning_seconds += elapsed
        self.diagnostics.nodes_expanded += stats.nodes_expanded
        self.diagnostics.frames_simulated += (
            self.simulator.frames_simulated - frames_before
        )
        self._frames_since_replan = 0

        if plan is None:
            # No safe trajectory found inside the budget. Keep whatever is queued -- a stale
            # plan is better than standing still, and the failure is counted, not hidden.
            self.diagnostics.notes["failed_replans"] += 1
            return
        self._queue = list(plan.bytes_)
        self.replanned_this_frame = True
        # Re-simulate the new plan to get its predicted positions, so the demo can draw the
        # trajectory the planner has just committed to. One short replay per replan.
        self._path = self.simulator.trajectory(snapshot, list(plan.bytes_))
