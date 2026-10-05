"""Tests for the A* heuristic.

Admissibility and consistency are not stylistic preferences: admissibility is what makes
A* optimal, and consistency is what guarantees no node is re-expanded -- which is in turn
what lets the planner release each node's 272 KB snapshot on expansion. So these tests check
the *physical premise* (Mario cannot exceed the speed cap), not just the arithmetic.
"""

from __future__ import annotations

import pytest

from src.env.factories import make_planner_env
from src.env.ram_map import MAX_VX_PIXELS_PER_FRAME
from src.env.state import MarioState
from src.planning.heuristic import DistanceHeuristic
from src.planning.primitives import DEFAULT_PRIMITIVES
from src.planning.simulator import ForwardSimulator


def _state(x: int) -> MarioState:
    return MarioState(x=x, y=432, vx=0, vy=0, on_ground=True, powerup=0)


def test_heuristic_is_zero_at_and_beyond_the_goal() -> None:
    """A goal state must have h = 0, or A* could stop early or never stop."""
    heuristic = DistanceHeuristic(goal_x=1000)
    assert heuristic(_state(1000)) == 0.0
    assert heuristic(_state(1500)) == 0.0


def test_heuristic_is_never_negative() -> None:
    """A negative h would make f decrease along a path and break the priority ordering."""
    heuristic = DistanceHeuristic(goal_x=1000)
    for x in (0, 500, 999, 1000, 5000):
        assert heuristic(_state(x)) >= 0.0


def test_heuristic_decreases_with_progress() -> None:
    """Closer to the goal must mean a smaller estimate."""
    heuristic = DistanceHeuristic(goal_x=1000)
    assert heuristic(_state(0)) > heuristic(_state(500)) > heuristic(_state(900))


def test_heuristic_uses_the_measured_speed_cap() -> None:
    """h is in frames: distance divided by the fastest Mario can travel."""
    heuristic = DistanceHeuristic(goal_x=1000)
    assert heuristic(_state(400)) == pytest.approx(600 / MAX_VX_PIXELS_PER_FRAME)


def test_zero_or_negative_v_max_is_rejected() -> None:
    """Guards against a config that would divide by zero or invert the ordering."""
    with pytest.raises(ValueError):
        DistanceHeuristic(goal_x=1000, v_max=0.0)
    with pytest.raises(ValueError):
        DistanceHeuristic(goal_x=1000, v_max=-3.0)


def test_mario_never_outruns_the_speed_cap() -> None:
    """The physical premise of admissibility, checked against the real game.

    If any primitive could move Mario more than ``v_max`` pixels per frame, the heuristic
    would overestimate and A* would lose its optimality guarantee.
    """
    env = make_planner_env("1-1")
    try:
        simulator = ForwardSimulator(env)
        _, snapshot = simulator.root(seed=0)
        worst_ratio = 0.0
        frontier = [snapshot]
        for _ in range(3):  # three plies is plenty to reach top speed
            next_frontier = []
            for parent in frontier[:4]:
                for primitive in DEFAULT_PRIMITIVES:
                    before = simulator.env
                    before.load_state(parent)
                    x_before = MarioState.from_env(before).x
                    transition = simulator.expand(parent, primitive)
                    if not transition.alive:
                        continue
                    travelled = transition.state.x - x_before
                    worst_ratio = max(worst_ratio, travelled / transition.frames)
                    next_frontier.append(transition.snapshot)
            frontier = next_frontier
        assert worst_ratio <= MAX_VX_PIXELS_PER_FRAME, (
            f"Mario moved {worst_ratio:.2f} px/frame, above the assumed cap of "
            f"{MAX_VX_PIXELS_PER_FRAME}; the heuristic would be inadmissible"
        )
    finally:
        env.close()


def test_heuristic_is_consistent_along_real_edges() -> None:
    """``h(s) <= cost(s, s') + h(s')`` for every edge the planner can actually take.

    Consistency is what lets an expanded node be closed forever -- and therefore what lets
    its snapshot be freed.
    """
    env = make_planner_env("1-1")
    try:
        simulator = ForwardSimulator(env)
        start, snapshot = simulator.root(seed=0)
        heuristic = DistanceHeuristic(goal_x=3161)
        for primitive in DEFAULT_PRIMITIVES:
            transition = simulator.expand(snapshot, primitive)
            if not transition.alive:
                continue
            assert heuristic(start) <= transition.frames + heuristic(transition.state) + 1e-9
    finally:
        env.close()
