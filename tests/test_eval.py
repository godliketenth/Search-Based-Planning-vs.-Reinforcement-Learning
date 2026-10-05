"""Tests for the metrics and the evaluation harness.

Death-cause classification gets the most attention here. Phase 2 established that a pit fall
sets ``y_viewport > 1`` several frames *before* ``player_state`` flips to the dying value, so
the obvious ordering of those two checks silently misattributes most pit deaths to enemies --
a bug that would produce a plausible, wrong table rather than an error.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.agents.random_agent import RandomAgent
from src.env import ram_map
from src.eval.harness import (
    CONDITION_DETERMINISTIC, condition_name, run_cell, run_episode,
    write_results_csv, write_summary_csv,
)
from src.eval.metrics import DeathCause, EpisodeResult, classify_death, summarise


def _ram(y_viewport: int = 1, player_state: int = 0x08) -> np.ndarray:
    ram = np.zeros(2048, dtype=np.uint8)
    ram[ram_map.ADDR_Y_VIEWPORT] = y_viewport
    ram[ram_map.ADDR_PLAYER_STATE] = player_state
    return ram


def _result(**overrides) -> EpisodeResult:
    base = dict(
        agent="a", level="1-1", condition="deterministic", episode=0, seed=0,
        completed=False, max_x=600, frames=500, death_cause=DeathCause.ENEMY,
    )
    base.update(overrides)
    return EpisodeResult(**base)


# ------------------------------------------------------------------- death classification


def test_completion_is_not_a_death() -> None:
    """Reaching the flag must never be counted as a failure of any kind."""
    assert classify_death(_ram(), {}, flag_get=True) == DeathCause.NONE


def test_pit_fall_is_detected_from_the_viewport() -> None:
    """``y_viewport > 1`` means Mario has fallen below the screen."""
    assert classify_death(_ram(y_viewport=3), {}, flag_get=False) == DeathCause.PIT


def test_pit_takes_priority_over_the_dying_flag() -> None:
    """The ordering that matters.

    During a pit fall the viewport flag fires first and ``player_state`` flips to dying a few
    frames later. Testing ``player_state`` first would file most pit deaths under 'enemy',
    producing a wrong table with no error.
    """
    ram = _ram(y_viewport=4, player_state=ram_map.PLAYER_STATE_DYING)
    assert classify_death(ram, {"is_dying": True}, flag_get=False) == DeathCause.PIT


def test_enemy_contact_is_detected_from_player_state() -> None:
    """Phase 0, Finding 6: ``player_state == 0x0B`` on the contact frame."""
    ram = _ram(player_state=ram_map.PLAYER_STATE_DYING)
    assert classify_death(ram, {}, flag_get=False) == DeathCause.ENEMY


def test_dead_state_also_counts_as_enemy_contact() -> None:
    """``0x06`` is the post-animation dead state."""
    ram = _ram(player_state=ram_map.PLAYER_STATE_DEAD)
    assert classify_death(ram, {}, flag_get=False) == DeathCause.ENEMY


def test_surviving_to_the_frame_cap_is_a_timeout() -> None:
    """A stalled agent -- like the random one at the first pipe -- is not a death."""
    assert classify_death(_ram(), {}, flag_get=False) == DeathCause.TIMEOUT


# ------------------------------------------------------------------------------ summaries


def test_summary_groups_by_agent_level_and_condition() -> None:
    """One row per cell of the experimental grid."""
    results = [
        _result(agent="x", condition="deterministic"),
        _result(agent="x", condition="sticky_0.25"),
        _result(agent="y", condition="deterministic"),
    ]
    rows = summarise(results)
    assert len(rows) == 3
    assert {row["agent"] for row in rows} == {"x", "y"}


def test_completion_rate_is_a_fraction_of_episodes() -> None:
    """The headline metric."""
    results = [_result(completed=True), _result(completed=False),
               _result(completed=True), _result(completed=True)]
    assert summarise(results)[0]["completion_rate"] == pytest.approx(0.75)


def test_frames_to_goal_averages_completed_episodes_only() -> None:
    """Averaging in a death's frame count would make a fast-dying agent look fast."""
    results = [
        _result(completed=True, frames=1000, death_cause=DeathCause.NONE),
        _result(completed=False, frames=100, death_cause=DeathCause.ENEMY),
    ]
    assert summarise(results)[0]["mean_frames_to_goal"] == pytest.approx(1000.0)


def test_frames_to_goal_is_none_when_nothing_completed() -> None:
    """Must not report 0 or crash for an agent that never finished."""
    assert summarise([_result(completed=False)])[0]["mean_frames_to_goal"] is None


def test_death_causes_are_counted_separately() -> None:
    """How an agent fails is a reported metric, not a footnote."""
    results = [
        _result(death_cause=DeathCause.ENEMY), _result(death_cause=DeathCause.PIT),
        _result(death_cause=DeathCause.PIT), _result(death_cause=DeathCause.TIMEOUT),
    ]
    row = summarise(results)[0]
    assert (row["deaths_enemy"], row["deaths_pit"], row["timeouts"]) == (1, 2, 1)


# ----------------------------------------------------------------------------- conditions


def test_condition_names_are_stable() -> None:
    """These strings are table keys and figure labels; they must not drift."""
    assert condition_name(0.0) == CONDITION_DETERMINISTIC
    assert condition_name(0.25) == "sticky_0.25"


# --------------------------------------------------------------------------------- CSV


def test_results_csv_has_one_row_per_episode(tmp_path) -> None:
    """The tidy per-episode table the report's analysis reads."""
    path = tmp_path / "episodes.csv"
    write_results_csv([_result(episode=0), _result(episode=1)], path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["agent"] == "a"
    assert "death_cause" in rows[0]


def test_notes_are_flattened_for_csv() -> None:
    """Nested dicts cannot go in a CSV cell; they are rendered as key=value pairs."""
    row = _result(notes={"failed_replans": 2, "plan_exhausted": True}).as_row()
    assert row["notes"] == "failed_replans=2;plan_exhausted=True"


def test_summary_csv_round_trips(tmp_path) -> None:
    """The aggregated table written by the script."""
    path = tmp_path / "summary.csv"
    write_summary_csv(summarise([_result(completed=True)]), path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["completion_rate"] == "1.0"


# ------------------------------------------------------------------------------- harness


def test_run_episode_produces_a_complete_result() -> None:
    """End to end on the real emulator, with the cheapest agent."""
    result = run_episode(
        RandomAgent(seed=0), level="1-1", sticky_p=0.0, seed=0, episode=0, max_frames=400
    )
    assert result.agent == "random"
    assert result.condition == CONDITION_DETERMINISTIC
    assert result.frames <= 400
    assert result.max_x > 0
    assert result.decisions > 0
    assert result.decision_ms_mean >= 0.0
    assert result.death_cause in vars(DeathCause).values()


def test_deterministic_condition_is_actually_deterministic() -> None:
    """The premise of the whole experimental design.

    If repeated runs under the deterministic condition differed, the second condition would
    not isolate the effect of the perturbation.
    """
    first = run_episode(RandomAgent(seed=3), "1-1", 0.0, seed=0, episode=0, max_frames=300)
    second = run_episode(RandomAgent(seed=3), "1-1", 0.0, seed=0, episode=1, max_frames=300)
    assert (first.max_x, first.frames, first.death_cause) == (
        second.max_x, second.frames, second.death_cause
    )


def test_sticky_condition_records_the_realised_rate() -> None:
    """The configured perturbation must be checkable against what actually happened."""
    result = run_episode(
        RandomAgent(seed=0), "1-1", 0.25, seed=0, episode=0, max_frames=400
    )
    assert result.sticky_realised is not None
    assert 0.15 < result.sticky_realised < 0.35


def test_deterministic_condition_records_no_sticky_rate() -> None:
    """p = 0 adds no wrapper at all, so there is nothing to report."""
    result = run_episode(RandomAgent(seed=0), "1-1", 0.0, seed=0, episode=0, max_frames=200)
    assert result.sticky_realised is None


def test_decision_every_reduces_the_decision_count() -> None:
    """PPO decides every 4 frames; the planner every frame. Both on the same clock."""
    every_frame = run_episode(
        RandomAgent(seed=0), "1-1", 0.0, seed=0, episode=0, max_frames=200, decision_every=1
    )
    every_fourth = run_episode(
        RandomAgent(seed=0), "1-1", 0.0, seed=0, episode=0, max_frames=200, decision_every=4
    )
    assert every_fourth.decisions == pytest.approx(every_frame.decisions / 4, abs=1)
    assert every_fourth.frames == every_frame.frames


def test_run_cell_builds_a_fresh_agent_per_episode() -> None:
    """A leaked plan queue or policy state would corrupt the next episode's result."""
    built = []

    def factory():
        agent = RandomAgent(seed=len(built))
        built.append(agent)
        return agent

    results = run_cell(factory, "1-1", 0.0, n_episodes=2, max_frames=150, verbose=False)
    assert len(results) == 2
    assert len(built) == 2
    assert built[0] is not built[1]
    assert [r.episode for r in results] == [0, 1]
