"""Tests for the side-by-side comparison session and the agent factory.

Headless throughout -- no Streamlit import, no window. The dashboard is a thin shell over
:class:`ComparisonSession`, so everything worth testing lives here.

The emulator-leak test matters more than it looks: each side owns a native emulator, and a
dashboard that rebuilds a session on every dropdown change without closing the old one will
exhaust memory during a demo.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.factory import (
    AGENT_NAMES, EVALUABLE_AGENTS, agent_availability, build_agent,
)
from src.config import load_config
from src.viz.comparison import SIDES, ComparisonSession


@pytest.fixture(scope="module")
def configs():
    """The demo and planner configs, loaded once."""
    return load_config("demo.yaml"), load_config("astar.yaml")


@pytest.fixture
def session(configs):
    """A cheap two-random-agent session, so tests stay fast."""
    config, astar_config = configs
    built = ComparisonSession(
        "random", "random", config, astar_config,
        sticky_p=0.0, seed=0, scale=1, frames_per_tick=3,
    )
    yield built
    built.close()


# ------------------------------------------------------------------------------- factory


def test_every_named_agent_is_constructible_on_1_1(configs) -> None:
    """The dashboard offers these in a dropdown; each must actually build."""
    config, astar_config = configs
    for name in AGENT_NAMES:
        available, reason = agent_availability(name, "1-1", config)
        assert available, f"{name} unavailable on 1-1: {reason}"
        agent, _ = build_agent(name, "1-1", config, astar_config)
        assert agent.name
        if hasattr(agent, "planner_env"):
            agent.planner_env.close()


def test_unavailable_agent_reports_a_reason_not_a_crash(configs) -> None:
    """A missing artefact must grey out a UI option, not raise mid-demo."""
    config, _ = configs
    available, reason = agent_availability("astar_open_loop", "1-2", config)
    assert not available
    assert reason, "an unavailable agent must explain itself"


def test_building_an_unavailable_agent_raises_clearly(configs) -> None:
    """Callers that skip the availability check still get a useful error."""
    config, astar_config = configs
    with pytest.raises(FileNotFoundError):
        build_agent("astar_open_loop", "1-2", config, astar_config)


def test_unknown_agent_is_rejected(configs) -> None:
    """A typo'd name must fail rather than silently produce nothing."""
    config, astar_config = configs
    with pytest.raises((ValueError, FileNotFoundError)):
        build_agent("astar_openloop", "1-1", config, astar_config)


def test_evaluable_agents_exclude_the_human(configs) -> None:
    """A human run cannot be collected unattended, so the harness must not offer it."""
    assert "human" not in EVALUABLE_AGENTS
    assert set(EVALUABLE_AGENTS) < set(AGENT_NAMES)


def test_open_loop_build_returns_its_plan(configs) -> None:
    """The caller needs the plan to draw it; reloading it separately would be wasteful."""
    config, astar_config = configs
    agent, plan = build_agent("astar_open_loop", "1-1", config, astar_config)
    assert plan is not None and len(plan) > 0
    assert build_agent("random", "1-1", config, astar_config)[1] is None


# ------------------------------------------------------------------------------- stepping


def test_tick_advances_both_sides_by_frames_per_tick(session) -> None:
    """Lockstep is the whole premise -- the panels must stay on the same frame count."""
    session.tick()
    assert all(session.runners[side].state.frame == 3 for side in SIDES)
    session.tick()
    assert all(session.runners[side].state.frame == 6 for side in SIDES)


def test_tick_returns_a_frame_and_stats_per_side(session) -> None:
    """What the UI renders."""
    out = session.tick()
    assert set(out) == set(SIDES)
    for side in SIDES:
        assert isinstance(out[side]["frame"], np.ndarray)
        assert out[side]["frame"].ndim == 3
        assert out[side]["stats"]["agent"] == "random"
        assert out[side]["agent"] == "random"


def test_both_sides_start_from_the_same_seed(session) -> None:
    """Any divergence must be the agents' doing, not a different starting point."""
    out = session.tick()
    assert out["left"]["stats"]["x"] == out["right"]["stats"]["x"]


def test_a_finished_side_stops_advancing_but_still_renders(session) -> None:
    """A dead agent should freeze on screen while the other keeps going."""
    session.runners["left"].state.episode_over = True
    before = session.runners["left"].state.frame
    out = session.tick()
    assert session.runners["left"].state.frame == before
    assert out["left"]["frame"] is not None
    assert session.runners["right"].state.frame > before


def test_reset_restarts_both_episodes(session) -> None:
    """One button must return the comparison to a known state."""
    for _ in range(4):
        session.tick()
    session.reset()
    assert session.ticks == 0
    assert all(session.runners[side].state.frame == 0 for side in SIDES)


def test_finished_is_true_only_when_both_sides_are_done(session) -> None:
    """Used to stop the UI's refresh loop."""
    assert not session.finished
    session.runners["left"].state.episode_over = True
    assert not session.finished
    session.runners["right"].state.episode_over = True
    assert session.finished


# ------------------------------------------------------------------------ sticky control


def test_set_sticky_reaches_both_samplers(session) -> None:
    """The live toggle is the demo's point; it must affect execution, not just the HUD."""
    session.set_sticky(0.25)
    assert all(session.runners[side].env.sampler.p == 0.25 for side in SIDES)
    for _ in range(40):
        session.tick()
    assert any(session.runners[side].env.sampler.n_overridden > 0 for side in SIDES)


def test_sticky_can_be_switched_off_again(session) -> None:
    """Toggling back must genuinely stop the perturbation."""
    session.set_sticky(0.25)
    session.set_sticky(0.0)
    assert all(session.runners[side].env.sampler.p == 0.0 for side in SIDES)
    overridden_before = {s: session.runners[s].env.sampler.n_overridden for s in SIDES}
    for _ in range(20):
        session.tick()
    assert all(
        session.runners[s].env.sampler.n_overridden == overridden_before[s] for s in SIDES
    )


def test_stats_report_the_realised_override_rate(session) -> None:
    """The two panels face the same process, not the same realisation, so each reports its own."""
    session.set_sticky(0.25)
    for _ in range(60):
        session.tick()
    out = session.tick()
    for side in SIDES:
        assert 0.0 <= out[side]["stats"]["sticky_realised"] <= 1.0
        assert out[side]["stats"]["sticky_p"] == 0.25


# ------------------------------------------------------------------------------- lifecycle


def test_close_releases_both_runners(configs) -> None:
    """Each side owns a native emulator; an unclosed session leaks one per side."""
    config, astar_config = configs
    built = ComparisonSession("random", "random", config, astar_config, scale=1)
    built.close()
    assert built.runners == {}


def test_close_is_idempotent(configs) -> None:
    """The UI may close a session on both re-selection and teardown."""
    config, astar_config = configs
    built = ComparisonSession("random", "random", config, astar_config, scale=1)
    built.close()
    built.close()


# ------------------------------------------------------------------------- runner stats


def test_planner_stats_appear_only_for_planning_agents(configs) -> None:
    """A reactive policy should omit the planning keys, not report zeros.

    Reporting 0 would conflate "does not plan" with "has not planned yet".
    """
    config, astar_config = configs
    built = ComparisonSession(
        "astar_closed_loop", "random", config, astar_config, scale=1, frames_per_tick=6
    )
    try:
        for _ in range(5):
            out = built.tick()
        assert "replans" in out["left"]["stats"]
        assert out["left"]["stats"]["replans"] > 0
        assert "nodes_expanded" in out["left"]["stats"]
        assert "replans" not in out["right"]["stats"]
    finally:
        built.close()


def test_stats_describe_the_configuration(session) -> None:
    """The panel shows the C-space quantities, not just a picture."""
    for _ in range(10):
        out = session.tick()
    stats = out["left"]["stats"]
    for key in ("x", "y", "vx", "vy", "on_ground", "frame", "action", "enemies"):
        assert key in stats
