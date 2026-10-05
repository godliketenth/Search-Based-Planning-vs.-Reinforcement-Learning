"""Tests for the forward simulator and the A* search.

The two properties worth protecting here are:

* **exactness** -- expanding the same edge twice must give the same child, or the search is
  exploring a graph that does not exist; and
* **snapshot discipline** -- at 272 KB each, a leaked snapshot per node is gigabytes.
"""

from __future__ import annotations

import pytest

from src.env.actions import RIGHT_B
from src.env.factories import make_planner_env
from src.env.state import MarioState
from src.planning.astar import SearchBudget, plan_level
from src.planning.heuristic import DistanceHeuristic
from src.planning.plan import Plan
from src.planning.primitives import DEFAULT_PRIMITIVES, MotionPrimitive, hold
from src.planning.simulator import ForwardSimulator, SnapshotStore

#: A near goal, so the search tests stay fast while still exercising a real search.
NEAR_GOAL_X = 400


@pytest.fixture(scope="module")
def simulator():
    """A forward simulator over a planner env."""
    env = make_planner_env("1-1")
    yield ForwardSimulator(env)
    env.close()


# ------------------------------------------------------------------ snapshot discipline


def test_snapshot_store_releases_on_take() -> None:
    """``take`` is the expansion step: holding the snapshot afterwards would leak 272 KB."""
    store = SnapshotStore()
    store.put(1, object())
    assert len(store) == 1
    store.take(1)
    assert len(store) == 0


def test_taking_twice_raises() -> None:
    """A second take means a node was expanded twice -- consistency would be violated."""
    store = SnapshotStore()
    store.put(1, object())
    store.take(1)
    with pytest.raises(KeyError):
        store.take(1)


def test_discard_is_idempotent() -> None:
    """Pruned nodes may or may not still hold a snapshot."""
    store = SnapshotStore()
    store.put(1, object())
    store.discard(1)
    store.discard(1)
    assert len(store) == 0


def test_peak_live_tracks_the_high_water_mark() -> None:
    """Peak frontier is the planner's reported memory figure."""
    store = SnapshotStore()
    for node_id in range(5):
        store.put(node_id, object())
    for node_id in range(5):
        store.take(node_id)
    assert store.peak_live == 5
    assert store.peak_bytes > 0


# ---------------------------------------------------------------------- forward simulation


def test_expanding_the_same_edge_twice_is_identical(simulator) -> None:
    """Exact forward simulation: the transition function must be a function."""
    _, snapshot = simulator.root(seed=0)
    primitive = DEFAULT_PRIMITIVES[0]
    first = simulator.expand(snapshot, primitive)
    second = simulator.expand(snapshot, primitive)
    assert first.state == second.state
    assert first.frames == second.frames
    assert first.alive == second.alive


def test_expanding_does_not_consume_the_parent(simulator) -> None:
    """A node is expanded once but has many children; the parent snapshot must survive."""
    _, snapshot = simulator.root(seed=0)
    results = [simulator.expand(snapshot, primitive) for primitive in DEFAULT_PRIMITIVES]
    again = simulator.expand(snapshot, DEFAULT_PRIMITIVES[0])
    assert again.state == results[0].state


def test_different_primitives_reach_different_states(simulator) -> None:
    """Otherwise the branching factor is a fiction and the search is a straight line."""
    _, snapshot = simulator.root(seed=0)
    states = {
        simulator.expand(snapshot, primitive).state
        for primitive in DEFAULT_PRIMITIVES
    }
    assert len(states) >= 4


def test_a_fatal_primitive_is_an_invalid_edge(simulator) -> None:
    """Collision checking *is* the simulation: running into the Goomba must kill the edge."""
    simulator.root(seed=0)
    env = simulator.env
    while env._get_info()["x_pos"] < 270:
        env._frame_advance(RIGHT_B)
    snapshot = env.dump_state()

    suicide = MotionPrimitive("suicide", hold(RIGHT_B, 60))
    transition = simulator.expand(snapshot, suicide)
    assert not transition.alive
    assert transition.snapshot is None, "a dead end must not cost 272 KB"
    assert transition.frames < suicide.frames, "simulation should stop early on death"


def test_surviving_primitive_returns_a_usable_snapshot(simulator) -> None:
    """Live edges must carry a snapshot, or their children cannot be expanded."""
    _, snapshot = simulator.root(seed=0)
    transition = simulator.expand(snapshot, DEFAULT_PRIMITIVES[0])
    assert transition.alive
    assert transition.snapshot is not None
    follow_on = simulator.expand(transition.snapshot, DEFAULT_PRIMITIVES[0])
    assert follow_on.state.x > transition.state.x


def test_trajectory_records_one_point_per_frame(simulator) -> None:
    """The overlay draws these; a mismatch would desynchronise the drawn plan."""
    _, snapshot = simulator.root(seed=0)
    sequence = [RIGHT_B] * 30
    points = simulator.trajectory(snapshot, sequence)
    assert len(points) == len(sequence)
    assert points[-1][0] > points[0][0], "Mario should have moved right"


# ------------------------------------------------------------------------------- search


def test_search_finds_and_verifies_a_short_plan(simulator) -> None:
    """End to end: A* produces a plan whose open-loop replay reproduces the outcome."""
    plan, stats, expanded = plan_level(
        simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=NEAR_GOAL_X),
        goal_x=NEAR_GOAL_X, budget=SearchBudget(max_nodes=400, max_seconds=60),
    )
    assert plan is not None
    assert stats.goal_reached
    assert plan.final_x >= NEAR_GOAL_X
    assert len(plan) > 0

    env = make_planner_env("1-1")
    try:
        env.reset(seed=0)
        for byte in plan.bytes_:
            env._frame_advance(byte)
        assert env._get_info()["x_pos"] == plan.final_x
    finally:
        env.close()


def test_plan_length_is_the_sum_of_its_primitives(simulator) -> None:
    """``g`` is measured in frames, so the plan's length must equal its cost."""
    plan, _, _ = plan_level(
        simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=NEAR_GOAL_X),
        goal_x=NEAR_GOAL_X, budget=SearchBudget(max_nodes=400, max_seconds=60),
    )
    assert plan is not None
    by_name = {primitive.name: primitive for primitive in DEFAULT_PRIMITIVES}
    expected = sum(by_name[name].frames for name in plan.primitive_names)
    assert len(plan) == expected


def test_plan_respects_the_admissible_lower_bound(simulator) -> None:
    """An optimal plan cannot be shorter than the admissible heuristic says."""
    heuristic = DistanceHeuristic(goal_x=NEAR_GOAL_X)
    plan, _, _ = plan_level(
        simulator, DEFAULT_PRIMITIVES, heuristic,
        goal_x=NEAR_GOAL_X, budget=SearchBudget(max_nodes=400, max_seconds=60),
    )
    assert plan is not None
    start = MarioState(x=40, y=432, vx=0, vy=0, on_ground=True, powerup=0)
    assert len(plan) >= heuristic(start) - 1e-9


def test_weight_below_one_is_rejected(simulator) -> None:
    """Deflating an admissible heuristic is never useful and breaks the bound."""
    with pytest.raises(ValueError):
        plan_level(
            simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=NEAR_GOAL_X),
            weight=0.5, budget=SearchBudget(max_nodes=10),
        )


def test_exhausted_budget_returns_no_plan(simulator) -> None:
    """The planner must fail honestly rather than return a partial plan as if complete."""
    plan, stats, _ = plan_level(
        simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=3161),
        budget=SearchBudget(max_nodes=3, max_seconds=30),
    )
    assert plan is None
    assert stats.exhausted
    assert not stats.goal_reached


def test_search_reports_the_work_it_did(simulator) -> None:
    """These statistics are the Phase 5 comparison's 'cost of planning' column."""
    _, stats, expanded = plan_level(
        simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=NEAR_GOAL_X),
        goal_x=NEAR_GOAL_X, budget=SearchBudget(max_nodes=400, max_seconds=60),
    )
    assert stats.nodes_expanded == len(expanded) > 0
    assert stats.nodes_generated >= stats.nodes_expanded
    assert stats.frames_simulated > 0
    assert stats.wall_s > 0
    assert stats.peak_frontier > 0
    assert stats.edges_per_s > 0


def test_plan_round_trips_through_json(simulator, tmp_path) -> None:
    """Plans are cached to disk as action sequences -- snapshots are not picklable."""
    plan, _, _ = plan_level(
        simulator, DEFAULT_PRIMITIVES, DistanceHeuristic(goal_x=NEAR_GOAL_X),
        goal_x=NEAR_GOAL_X, budget=SearchBudget(max_nodes=400, max_seconds=60),
    )
    assert plan is not None
    path = plan.save(tmp_path / "plan.json")
    reloaded = Plan.load(path)
    assert reloaded.bytes_ == plan.bytes_
    assert reloaded.final_x == plan.final_x
    assert reloaded.primitive_names == plan.primitive_names
    assert reloaded.actions == plan.actions
