"""Tests for the live demo.

All headless: they exercise the runner, the interface state machine and the recorder without
ever opening a window, so the suite runs anywhere. The demo's value is that it must work on
the day, so the things tested are the ones that would ruin a presentation -- a window that
fails to open, a recording that is empty, a toggle that does nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.human_agent import KEY_TO_ACTION, HumanAgent
from src.agents.random_agent import RandomAgent
from src.viz.live_demo import (
    KEY_FASTER, KEY_OVERLAY, KEY_PAUSE, KEY_QUIT, KEY_RESET, KEY_SLOWER, KEY_STEP, KEY_STICKY,
    OVERLAY_MODES, SPEED_STEPS, DemoRunner, DemoState,
)


@pytest.fixture
def runner():
    """A headless runner over the cheapest agent."""
    demo = DemoRunner(
        RandomAgent(seed=0), level="1-1", sticky_p=0.0, scale=1,
        max_frames=400, record_path=None, headless=True,
    )
    demo.reset()
    yield demo
    demo.close()


# ----------------------------------------------------------------------------- rendering


def test_step_returns_a_rendered_frame(runner) -> None:
    """The loop's only output; if this is wrong there is nothing to show or record."""
    frame = runner.step()
    assert frame.shape == (240, 256, 3)
    assert frame.dtype == np.uint8
    assert frame.min() != frame.max(), "frame should not be blank"


def test_scale_upsamples_the_frame() -> None:
    """The NES frame is tiny; the demo renders it at 3x by default."""
    demo = DemoRunner(RandomAgent(seed=0), scale=3, headless=True, max_frames=20)
    try:
        demo.reset()
        assert demo.step().shape == (720, 768, 3)
    finally:
        demo.close()


def test_consecutive_frames_differ(runner) -> None:
    """Guards against a recorder capturing one frame repeatedly.

    ``env.render()`` returns a reused view of the emulator's screen buffer, so a missing copy
    silently produces a video of a single still image -- exactly what happened to the Phase 1
    GIF before it was caught.
    """
    frames = [runner.step() for _ in range(12)]
    assert len({frame.tobytes() for frame in frames}) > 1


def test_overlay_layers_change_what_is_drawn(runner) -> None:
    """The `o` key cycles presets; each must actually alter the image."""
    seen = []
    for _ in range(len(OVERLAY_MODES)):
        seen.append(runner.render().tobytes())
        runner.handle_key(KEY_OVERLAY)
    assert len(set(seen)) > 1


def test_overlay_key_cycles_back_to_the_start(runner) -> None:
    """Cycling must wrap rather than run off the end of the preset list."""
    for _ in range(len(OVERLAY_MODES)):
        runner.handle_key(KEY_OVERLAY)
    assert runner.state.overlay_index == 0


# ------------------------------------------------------------------------- sticky toggle


def test_sticky_toggle_switches_the_perturbation(runner) -> None:
    """The demo's most important control: it reproduces the headline result live."""
    assert runner.env.sampler.p == 0.0
    assert runner.toggle_sticky() == 0.25
    assert runner.env.sampler.p == 0.25
    assert runner.toggle_sticky() == 0.0
    assert runner.env.sampler.p == 0.0


def test_sticky_toggle_respects_a_configured_probability() -> None:
    """Launching with --sticky 0.5 must toggle to 0.5, not the 0.25 default."""
    demo = DemoRunner(RandomAgent(seed=0), sticky_p=0.5, headless=True, max_frames=20)
    try:
        assert demo.env.sampler.p == 0.5     # starts on, since p > 0
        assert demo.toggle_sticky() == 0.0   # off
        assert demo.toggle_sticky() == 0.5   # back on at the configured value
    finally:
        demo.close()


def test_sticky_toggle_actually_perturbs_execution(runner) -> None:
    """A toggle that changed only the HUD would be worse than none."""
    runner.toggle_sticky()
    for _ in range(200):
        runner.step()
    assert runner.env.sampler.n_overridden > 0


def test_sticky_key_is_wired_to_the_toggle(runner) -> None:
    """The `s` key, as documented in the CLI help."""
    runner.handle_key(KEY_STICKY)
    assert runner.state.sticky_on
    assert runner.env.sampler.p > 0


# --------------------------------------------------------------------- playback controls


def test_pause_halts_advancement(runner) -> None:
    """`should_advance` is what the loop consults; pausing must stop the world."""
    runner.handle_key(KEY_PAUSE)
    assert runner.state.paused
    assert not runner.should_advance()


def test_step_advances_exactly_one_frame_while_paused(runner) -> None:
    """Frame-stepping is how a still is captured for a slide."""
    runner.handle_key(KEY_PAUSE)
    runner.handle_key(KEY_STEP)
    assert runner.should_advance()
    assert not runner.should_advance(), "step must be consumed, not latched"


def test_speed_controls_clamp_at_both_ends(runner) -> None:
    """Running off the end of the speed table would raise mid-demo."""
    for _ in range(len(SPEED_STEPS) + 3):
        runner.handle_key(KEY_FASTER)
    assert runner.state.speed == SPEED_STEPS[-1]
    for _ in range(len(SPEED_STEPS) + 3):
        runner.handle_key(KEY_SLOWER)
    assert runner.state.speed == SPEED_STEPS[0]


def test_frame_delay_tracks_speed_and_stays_positive() -> None:
    """``cv2.waitKey(0)`` blocks forever, so the delay must never reach zero."""
    state = DemoState()
    slow = state.frame_delay_ms()
    state.speed_index = len(SPEED_STEPS) - 1
    assert state.frame_delay_ms() < slow
    assert state.frame_delay_ms() >= 1


def test_quit_key_stops_the_loop(runner) -> None:
    """`q` must end the run cleanly rather than requiring a kill."""
    runner.handle_key(KEY_QUIT)
    assert not runner.state.running


def test_reset_key_restarts_the_episode(runner) -> None:
    """`r` gives a second attempt without relaunching."""
    for _ in range(30):
        runner.step()
    assert runner.state.frame == 30
    runner.handle_key(KEY_RESET)
    assert runner.state.frame == 0
    assert not runner.state.episode_over


def test_unknown_and_empty_keys_are_ignored(runner) -> None:
    """``cv2.waitKey`` returns -1 constantly when no key is pressed."""
    before = (runner.state.paused, runner.state.speed_index, runner.state.overlay_index)
    for key in (-1, 255, ord("z")):
        runner.handle_key(key)
    assert (runner.state.paused, runner.state.speed_index, runner.state.overlay_index) == before


def test_episode_over_stops_advancement(runner) -> None:
    """The final frame is held so the outcome stays on screen."""
    runner.state.episode_over = True
    assert not runner.should_advance()


# ------------------------------------------------------------------------------ recording


def test_headless_run_writes_a_playable_mp4(tmp_path) -> None:
    """The mp4 is the viva backup, so it must exist and contain real frames."""
    import imageio.v2 as imageio

    path = tmp_path / "demo.mp4"
    demo = DemoRunner(RandomAgent(seed=0), scale=1, max_frames=5000,
                      record_path=path, headless=True)
    rendered = demo.run_headless(frames=40)
    assert rendered == 40
    assert path.exists() and path.stat().st_size > 0

    frames = [np.asarray(frame) for frame in imageio.get_reader(path)]
    assert len(frames) == 40
    assert len({frame.tobytes() for frame in frames}) > 1, "video must not be one still image"


def test_recording_can_be_disabled(tmp_path) -> None:
    """`--no-record` must not create a file."""
    demo = DemoRunner(RandomAgent(seed=0), scale=1, record_path=None, headless=True,
                      max_frames=100)
    assert demo.run_headless(frames=10) == 10
    assert not list(tmp_path.iterdir())


def test_headless_run_stops_at_episode_end() -> None:
    """A dead agent should not keep recording an inert screen for thousands of frames."""
    demo = DemoRunner(RandomAgent(seed=0), scale=1, max_frames=25,
                      record_path=None, headless=True)
    assert demo.run_headless(frames=10_000) <= 25


# ---------------------------------------------------------------------------- human agent


def test_human_keys_latch_an_action() -> None:
    """``cv2.waitKey`` has no key-up events, so a press persists until the next one."""
    agent = HumanAgent()
    agent.reset()
    assert agent.act(None) == 0
    agent.set_key(ord("d"))
    assert agent.act(None) == KEY_TO_ACTION[ord("d")]
    assert agent.act(None) == KEY_TO_ACTION[ord("d")], "action must persist"


def test_human_movement_keys_do_not_leak_into_ui_commands() -> None:
    """`a` means 'left' for a human player, not a UI shortcut."""
    demo = DemoRunner(HumanAgent(), scale=1, headless=True, max_frames=20)
    try:
        demo.reset()
        demo.handle_key(ord("a"))
        assert demo.agent.act(None) == KEY_TO_ACTION[ord("a")]
        assert demo.state.running, "a movement key must not be read as a UI command"
    finally:
        demo.close()


def test_ui_keys_still_work_for_the_human_agent() -> None:
    """Keys the agent does not claim must still reach the interface."""
    demo = DemoRunner(HumanAgent(), scale=1, headless=True, max_frames=20)
    try:
        demo.reset()
        demo.handle_key(KEY_QUIT)
        assert not demo.state.running
    finally:
        demo.close()


def test_human_reset_releases_the_latched_action() -> None:
    """A new episode must not inherit the previous one's held button."""
    agent = HumanAgent()
    agent.reset()
    agent.set_key(ord("f"))
    agent.reset()
    assert agent.act(None) == 0
