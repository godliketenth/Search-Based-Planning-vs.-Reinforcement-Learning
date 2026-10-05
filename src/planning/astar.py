"""A* over the motion-primitive lattice.

The graph
---------
* **Nodes** are configurations ``(x, y, vx, vy, on_ground, powerup)``, bucketed onto a
  lattice for duplicate detection.
* **Edges** are motion primitives, simulated exactly by the emulator.
* **Edge cost** is the primitive's duration in frames, so ``g`` is elapsed time and the
  planner minimises time-to-goal.
* **Collision checking** is the simulation itself: an edge is invalid iff Mario died on it.

Completeness, honestly stated
-----------------------------
This planner is **resolution-complete**, not complete, and optimal only *with respect to the
primitive set and the lattice resolution*. Two things cause that:

1. Only the primitives in the set can be executed, so a level needing a manoeuvre outside
   the set is unsolvable here however long the search runs.
2. States are merged by :meth:`MarioState.lattice_key`, so two genuinely different
   configurations in one cell are treated as one. A coarse lattice can therefore discard
   the only state from which a jump would have worked.

Weighted A* (``weight > 1``) inflates the heuristic to trade optimality for speed. The
result is **bounded suboptimal**: the returned plan costs at most ``weight x`` the optimal
plan. It is reported, never applied silently.

Memory
------
Snapshots are 272 KB each, so they live only for frontier nodes and are released at
expansion -- see :class:`~src.planning.simulator.SnapshotStore`. The closed set keeps only
lattice keys and ``g`` values.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.env.state import MarioState
from src.planning.heuristic import DistanceHeuristic
from src.planning.plan import Plan, SearchStats
from src.planning.primitives import MotionPrimitive
from src.planning.simulator import ForwardSimulator, SnapshotStore


@dataclass(slots=True)
class _Node:
    """One search node. Internal to the search; plans expose byte sequences instead."""

    state: MarioState
    g: int
    parent: int | None
    primitive: int | None


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Limits that stop a search rather than letting it run unbounded.

    Attributes:
        max_nodes: cap on nodes expanded.
        max_seconds: wall-clock cap.
        max_frontier: cap on live snapshots, i.e. a memory cap. At 272 KB each, 4,000 is
            roughly 1.1 GB.
    """

    max_nodes: int = 50_000
    max_seconds: float = 300.0
    max_frontier: int = 4_000


def plan_level(
    simulator: ForwardSimulator,
    primitives: Sequence[MotionPrimitive],
    heuristic: DistanceHeuristic,
    *,
    level: str = "1-1",
    seed: int = 0,
    goal_x: int | None = None,
    weight: float = 1.0,
    budget: SearchBudget | None = None,
    lattice: dict[str, int] | None = None,
    start: tuple[MarioState, Any] | None = None,
    on_progress: Callable[[int, MarioState], None] | None = None,
) -> tuple[Plan | None, SearchStats, list[MarioState]]:
    """Search for an open-loop plan from the level start to the goal.

    Args:
        simulator: the exact forward simulator.
        primitives: the edge set.
        heuristic: admissible frames-to-goal estimate.
        level: level name, recorded in the plan.
        seed: reset seed; a plan is only valid from this seed.
        goal_x: if given, reaching this x also counts as the goal, alongside the emulator's
            own ``flag_get``. Useful for planning to a waypoint.
        weight: heuristic inflation. ``1.0`` is optimal A*; above that is bounded
            suboptimal by this factor.
        budget: stopping limits.
        lattice: resolution overrides for :meth:`MarioState.lattice_key`.
        start: ``(state, snapshot)`` to search from, instead of the level start. This is what
            closed-loop replanning needs: the executor snapshots the *live* environment
            mid-episode and searches forward from wherever Mario actually is, which is no
            longer where the previous plan predicted. Omit for a full-level plan.
        on_progress: optional callback ``(nodes_expanded, best_state)``, for a progress bar.

    Returns:
        ``(plan, stats, expanded_states)``. ``plan`` is ``None`` if the budget ran out.
        ``expanded_states`` is every configuration expanded, for the search-visualisation
        figure.

    Raises:
        ValueError: if ``weight`` is below 1 (which would break admissibility outright).
    """
    if weight < 1.0:
        raise ValueError(f"weight must be >= 1.0; {weight} would make h inadmissible")
    budget = budget or SearchBudget()
    lattice = lattice or {}

    started = time.perf_counter()
    store = SnapshotStore()
    nodes: list[_Node] = []
    expanded_states: list[MarioState] = []

    start_state, start_snapshot = start if start is not None else simulator.root(seed=seed)
    nodes.append(_Node(state=start_state, g=0, parent=None, primitive=None))
    store.put(0, start_snapshot)

    # (f, tie-breaker, node_id). The counter keeps the heap total-ordered and makes
    # tie-breaking deterministic, which matters for reproducible search statistics.
    counter = 0
    open_heap: list[tuple[float, int, int]] = [(weight * heuristic(start_state), counter, 0)]
    best_g: dict[tuple, int] = {start_state.lattice_key(**lattice): 0}
    closed: set[tuple] = set()

    goal_node: int | None = None
    goal_flag_get = False
    deaths = 0
    pruned = 0
    generated = 0

    while open_heap:
        if len(nodes) and (
            len(expanded_states) >= budget.max_nodes
            or time.perf_counter() - started > budget.max_seconds
        ):
            break

        _, _, node_id = heapq.heappop(open_heap)
        node = nodes[node_id]
        key = node.state.lattice_key(**lattice)
        if key in closed:
            # A stale heap entry for a node we already settled. Release its snapshot.
            store.discard(node_id)
            pruned += 1
            continue
        closed.add(key)

        try:
            snapshot = store.take(node_id)
        except KeyError:  # pragma: no cover - would mean the consistency assumption broke
            pruned += 1
            continue

        expanded_states.append(node.state)
        if on_progress is not None and len(expanded_states) % 250 == 0:
            on_progress(len(expanded_states), node.state)

        for index, primitive in enumerate(primitives):
            transition = simulator.expand(snapshot, primitive)
            generated += 1

            if not transition.alive:
                deaths += 1
                continue

            child_key = transition.state.lattice_key(**lattice)
            child_g = node.g + transition.frames
            if child_key in closed or best_g.get(child_key, 1 << 30) <= child_g:
                pruned += 1
                continue

            nodes.append(
                _Node(state=transition.state, g=child_g, parent=node_id, primitive=index)
            )
            child_id = len(nodes) - 1
            best_g[child_key] = child_g
            store.put(child_id, transition.snapshot)

            if transition.flag_get or (goal_x is not None and transition.state.x >= goal_x):
                goal_node = child_id
                # Record how the goal was met: reaching an x waypoint is not the same as
                # touching the flag, and a plan that claims otherwise fails replay checks.
                goal_flag_get = transition.flag_get
                break

            counter += 1
            heapq.heappush(
                open_heap,
                (child_g + weight * heuristic(transition.state), counter, child_id),
            )

        # The parent's snapshot is dead the moment its children exist.
        del snapshot

        if goal_node is not None:
            break

        if len(store) > budget.max_frontier:
            # Memory guard: drop the worst frontier nodes rather than exhaust RAM.
            pruned += _trim_frontier(open_heap, store, budget.max_frontier)

    wall_s = time.perf_counter() - started
    stats = SearchStats(
        nodes_expanded=len(expanded_states),
        nodes_generated=generated,
        nodes_pruned=pruned,
        deaths=deaths,
        peak_frontier=store.peak_live,
        frames_simulated=simulator.frames_simulated,
        wall_s=wall_s,
        goal_reached=goal_node is not None,
        exhausted=goal_node is None,
    )

    if goal_node is None:
        return None, stats, expanded_states

    byte_sequence, names = _reconstruct(nodes, primitives, goal_node)
    plan = Plan(
        level=level,
        seed=seed,
        bytes_=tuple(byte_sequence),
        primitive_names=tuple(names),
        goal_x=heuristic.goal_x,
        final_x=nodes[goal_node].state.x,
        flag_get=goal_flag_get,
        stats=stats,
    )
    return plan, stats, expanded_states


def _trim_frontier(
    open_heap: list[tuple[float, int, int]], store: SnapshotStore, keep: int
) -> int:
    """Drop the highest-``f`` frontier nodes to stay inside the memory budget.

    This makes the search incomplete in the strict sense -- a discarded node might have led
    to the only solution -- so it is counted and reported rather than hidden.

    Returns:
        How many nodes were dropped.
    """
    if len(open_heap) <= keep:
        return 0
    open_heap.sort()
    dropped = open_heap[keep:]
    del open_heap[keep:]
    heapq.heapify(open_heap)
    for _, _, node_id in dropped:
        store.discard(node_id)
    return len(dropped)


def _reconstruct(
    nodes: list[_Node], primitives: Sequence[MotionPrimitive], goal_node: int
) -> tuple[list[int], list[str]]:
    """Walk parent pointers back to the root and concatenate the primitives' bytes."""
    byte_sequence: list[int] = []
    names: list[str] = []
    current: int | None = goal_node
    while current is not None:
        node = nodes[current]
        if node.primitive is not None:
            primitive = primitives[node.primitive]
            byte_sequence.extend(reversed(primitive.bytes_))
            names.append(primitive.name)
        current = node.parent
    byte_sequence.reverse()
    names.reverse()
    return byte_sequence, names
