"""Tests for the OpenCV side-by-side window and the terminal demo launcher.

Headless throughout: :class:`ComparisonWindow` keeps its interface state in plain attributes
and composites with numpy, so everything except ``cv2.imshow`` is testable without a display.

The launcher is tested for one thing only -- that its menu offers every agent the factory can
build. A new agent added to :data:`src.agents.factory.AGENT_NAMES` and forgotten here would
not fail anywhere else; it would simply be missing on the day.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.agents.factory import AGENT_NAMES
from src.config import PROJECT_ROOT, load_config
from src.viz.comparison import ComparisonSession
from src.viz.comparison_window import (
    DIVIDER_WIDTH, STATS_HEIGHT, TITLE_HEIGHT, ComparisonWindow, compose,
)
from src.viz.live_demo import SPEED_STEPS

LABELS = {"left": "A* open-loop", "right": "A* closed-loop"}


@pytest.fixture(scope="module")
def configs():
    """The demo and planner configs, loaded once."""
    return load_config("demo.yaml"), load_config("astar.yaml")


@pytest.fixture
def session(configs):
    """Two random agents at scale 1, so the composite stays small and fast."""
    config, astar_config = configs
    built = ComparisonSession(
        "random", "random", config, astar_config,
        sticky_p=0.0, seed=0, scale=1, frames_per_tick=3,
    )
    yield built
    built.close()


@pytest.fixture
def window(session):
    """A window over that session, recording nowhere."""
    return ComparisonWindow(session, LABELS, sticky_p=0.25, goal_x=3161)


# ----------------------------------------------------------------------------- composite


def test_composite_is_two_captioned_panels_side_by_side(session) -> None:
    """The layout must be exactly both frames plus their strips and one divider."""
    payloads = session.tick()
    image = compose(payloads, LABELS, goal_x=3161)
    height, width = payloads["left"]["frame"].shape[:2]
    assert image.shape == (TITLE_HEIGHT + height + STATS_HEIGHT,
                           width * 2 + DIVIDER_WIDTH, 3)
    assert image.dtype == np.uint8


def test_composite_preserves_each_frame_unaltered(session) -> None:
    """A panel must show the game, not a resampled approximation of it."""
    payloads = session.tick()
    image = compose(payloads, LABELS, goal_x=3161)
    height, width = payloads["left"]["frame"].shape[:2]
    top = TITLE_HEIGHT
    assert np.array_equal(image[top:top + height, :width], payloads["left"]["frame"])
    assert np.array_equal(image[top:top + height, width + DIVIDER_WIDTH:],
                          payloads["right"]["frame"])


def test_composite_works_without_a_goal(session) -> None:
    """``goal_x`` is optional; a level with no configured goal must still render."""
    assert compose(session.tick(), LABELS, goal_x=None).size > 0


# ------------------------------------------------------------------------------ controls


def test_sticky_toggle_applies_to_both_panels(window, session) -> None:
    """The point of the key: one press perturbs both agents, in the same frame."""
    assert window.toggle_sticky() == pytest.approx(0.25)
    assert all(runner.env.sampler.p == pytest.approx(0.25)
               for runner in session.runners.values())
    assert window.toggle_sticky() == pytest.approx(0.0)
    assert all(runner.env.sampler.p == 0.0 for runner in session.runners.values())


def test_sticky_falls_back_to_the_design_probability(session) -> None:
    """Opened unperturbed, the toggle must still do something visible."""
    window = ComparisonWindow(session, LABELS, sticky_p=0.0)
    assert window.toggle_sticky() == pytest.approx(0.25)


def test_pause_blocks_advancing_and_step_releases_one_tick(window) -> None:
    """``space`` then ``n`` must yield exactly one tick, not a resumed run."""
    assert window.should_advance()
    window.handle_key(ord(" "))
    assert not window.should_advance()
    window.handle_key(ord("n"))
    assert window.should_advance()
    assert not window.should_advance()


def test_speed_keys_stay_inside_the_ladder(window) -> None:
    """Holding `[` or `]` must clamp rather than index off the end."""
    for _ in range(len(SPEED_STEPS) + 3):
        window.handle_key(ord("]"))
    assert window.speed == SPEED_STEPS[-1]
    for _ in range(len(SPEED_STEPS) + 3):
        window.handle_key(ord("["))
    assert window.speed == SPEED_STEPS[0]


def test_faster_playback_waits_less(window) -> None:
    """The frame delay must track the speed multiplier, and never reach zero."""
    slow = window.delay_ms()
    window.handle_key(ord("]"))
    assert 0 < window.delay_ms() < slow


def test_quit_stops_the_loop(window) -> None:
    """``q`` is the documented way out."""
    window.handle_key(ord("q"))
    assert not window.running


def test_unknown_and_absent_keys_are_ignored(window) -> None:
    """``cv2.waitKey`` returns -1 when nothing was pressed; it must not be a command."""
    window.handle_key(-1)
    window.handle_key(255)
    window.handle_key(ord("z"))
    assert window.running and not window.paused


def test_overlay_key_cycles_both_panels(window, session) -> None:
    """Panels must share an overlay preset, or they are not comparable."""
    window.handle_key(ord("o"))
    indices = {runner.state.overlay_index for runner in session.runners.values()}
    assert indices == {window.overlay_index}


# ------------------------------------------------------------------------------- running


def test_headless_run_advances_and_writes_an_mp4(session, tmp_path: Path) -> None:
    """The recording is the backup for presenting, so it must exist without a window."""
    out = tmp_path / "pair.mp4"
    window = ComparisonWindow(session, LABELS, goal_x=3161, record_path=out)
    assert window.run_headless(ticks=3) == 3
    assert session.ticks == 3
    assert out.exists() and out.stat().st_size > 0


def test_headless_run_stops_when_both_episodes_end(configs, tmp_path: Path) -> None:
    """A finished session must not spin; the loop ends on its own."""
    config, astar_config = configs
    built = ComparisonSession(
        "random", "random", config, astar_config,
        sticky_p=0.0, seed=0, scale=1, frames_per_tick=3,
    )
    try:
        for runner in built.runners.values():
            runner.state.episode_over = True
        window = ComparisonWindow(built, LABELS)
        assert window.run_headless() == 0
    finally:
        built.close()


# ------------------------------------------------------------------------------ launcher


@pytest.fixture(scope="module")
def showcase():
    """``scripts/showcase.py``, imported by path -- it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "showcase", PROJECT_ROOT / "scripts" / "showcase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_menu_offers_every_agent_the_factory_builds(showcase) -> None:
    """An agent missing from the menu is an agent nobody demonstrates."""
    assert set(showcase.AGENTS) == set(AGENT_NAMES)


def test_launcher_pairs_name_known_agents(showcase) -> None:
    """Every side-by-side entry must be buildable, not a typo'd name."""
    for left, right, _ in showcase.PAIRS:
        assert left in showcase.AGENTS and right in showcase.AGENTS


def test_launcher_commands_use_the_running_interpreter(showcase) -> None:
    """Children must inherit this venv, not whatever ``python`` resolves to."""
    for command in (showcase.demo_command("ppo", "1-1", 0.25),
                    showcase.pair_command("ppo", "random", "1-1", 0.25),
                    showcase.dashboard_command()):
        assert command[0] == sys.executable


def test_launcher_reads_its_table_from_the_results_csv(showcase, capsys) -> None:
    """The menu must quote the evaluation, not a number typed into the script."""
    showcase.print_table("1-1")
    printed = capsys.readouterr().out
    assert "phase5_summary_main.csv" in printed
    for agent in ("astar_open_loop", "astar_closed_loop", "ppo", "random"):
        assert agent in printed
