"""Tests for the agents and the interface they share.

The protocol conformance tests are the important ones: the harness's fairness claim rests on
every agent being driven through exactly the same two methods, so an agent that quietly needs
something extra would undermine the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.astar_open_loop import AStarOpenLoopAgent
from src.agents.base import Agent, AgentDiagnostics, Observation
from src.agents.random_agent import RandomAgent
from src.env.actions import N_ACTIONS, NOOP, action_for_byte
from src.planning.plan import Plan, SearchStats


def _observation(frame: int = 0) -> Observation:
    return Observation(pixels=np.zeros((4, 84, 84), np.uint8), env=None, frame=frame)


def _plan(n_frames: int = 6) -> Plan:
    from src.env.actions import RIGHT_B

    return Plan(
        level="1-1", seed=0, bytes_=(RIGHT_B,) * n_frames,
        primitive_names=("run",), goal_x=3161, final_x=100,
        stats=SearchStats(nodes_expanded=7, frames_simulated=123, wall_s=1.5),
    )


# ------------------------------------------------------------------------------- protocol


@pytest.mark.parametrize(
    "agent",
    [RandomAgent(seed=0), AStarOpenLoopAgent(_plan())],
    ids=["random", "astar_open_loop"],
)
def test_agents_satisfy_the_protocol(agent) -> None:
    """Every agent must be drivable through exactly ``reset`` and ``act``."""
    assert isinstance(agent, Agent)
    assert isinstance(agent.name, str) and agent.name


def test_actions_are_within_the_action_space() -> None:
    """An out-of-range action would raise deep inside the env, not at the agent."""
    for agent in (RandomAgent(seed=1), AStarOpenLoopAgent(_plan())):
        agent.reset()
        for frame in range(10):
            action = agent.act(_observation(frame))
            assert isinstance(action, int)
            assert 0 <= action < N_ACTIONS


# --------------------------------------------------------------------------------- random


def test_random_agent_is_seeded() -> None:
    """Reproducibility: the floor baseline must be repeatable from its seed."""
    first = RandomAgent(seed=7)
    second = RandomAgent(seed=7)
    first.reset()
    second.reset()
    assert [first.act(_observation()) for _ in range(20)] == [
        second.act(_observation()) for _ in range(20)
    ]


def test_random_agent_ignores_the_observation() -> None:
    """It is the floor precisely because it uses no information at all."""
    agent = RandomAgent(seed=0)
    agent.reset()
    blind = [agent.act(_observation()) for _ in range(10)]
    agent_again = RandomAgent(seed=0)
    agent_again.reset()
    noisy = [
        agent_again.act(
            Observation(pixels=np.full((4, 84, 84), 255, np.uint8), env=None, frame=i)
        )
        for i in range(10)
    ]
    assert blind == noisy


# -------------------------------------------------------------------------- A* open-loop


def test_open_loop_replays_the_plan_in_order() -> None:
    """Open-loop execution is literally the stored sequence, one action per frame."""
    plan = _plan(5)
    agent = AStarOpenLoopAgent(plan)
    agent.reset()
    assert [agent.act(_observation(i)) for i in range(5)] == list(plan.actions)


def test_open_loop_ignores_the_world_entirely() -> None:
    """The defining property: no feedback, so nothing can change what it does."""
    agent = AStarOpenLoopAgent(_plan(5))
    agent.reset()
    blind = [agent.act(_observation(i)) for i in range(5)]
    agent.reset()
    hostile = [
        agent.act(
            Observation(pixels=np.full((4, 84, 84), 9, np.uint8), env=None, frame=i,
                        info={"is_dying": True, "x_pos": 0})
        )
        for i in range(5)
    ]
    assert blind == hostile


def test_open_loop_falls_back_inertly_when_the_plan_runs_out() -> None:
    """A NOOP fallback makes a stall visible in the metrics instead of hiding it."""
    agent = AStarOpenLoopAgent(_plan(2))
    agent.reset()
    for _ in range(2):
        agent.act(_observation())
    assert agent.act(_observation()) == action_for_byte(NOOP)
    assert agent.diagnostics.notes["plan_exhausted"] is True


def test_open_loop_reset_rewinds_the_plan() -> None:
    """Episodes must not inherit a partly-consumed plan."""
    agent = AStarOpenLoopAgent(_plan(4))
    agent.reset()
    first = [agent.act(_observation()) for _ in range(4)]
    agent.reset()
    assert [agent.act(_observation()) for _ in range(4)] == first


def test_open_loop_reports_the_offline_search_cost() -> None:
    """The plan was not free; its search cost belongs in the compute column."""
    agent = AStarOpenLoopAgent(_plan())
    agent.reset()
    assert agent.diagnostics.frames_simulated == 123
    assert agent.diagnostics.nodes_expanded == 7
    assert agent.diagnostics.planning_calls == 1


# ---------------------------------------------------------------------------- diagnostics


def test_diagnostics_reset_clears_every_counter() -> None:
    """Stale counters would inflate the next episode's reported cost."""
    diagnostics = AgentDiagnostics(
        planning_calls=3, planning_seconds=1.5, nodes_expanded=50,
        frames_simulated=900, notes={"x": 1},
    )
    diagnostics.reset()
    assert diagnostics.planning_calls == 0
    assert diagnostics.planning_seconds == 0.0
    assert diagnostics.nodes_expanded == 0
    assert diagnostics.frames_simulated == 0
    assert diagnostics.notes == {}
