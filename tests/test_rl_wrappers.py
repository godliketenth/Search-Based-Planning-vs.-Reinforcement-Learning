"""Tests for the RL observation and timing wrappers.

Preprocessing bugs are quiet: a policy will happily train on a subtly wrong observation and
simply learn less, so these pin the shapes, the frame accounting and the episode outcome
fields the Phase 5 metrics depend on.
"""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
import pytest

from src.env.factories import make_rl_env
from src.rl.wrappers import EpisodeStatistics, FrameSkip, NativeGrayResize


@pytest.fixture
def base_env():
    """A wrapped RL env without any preprocessing."""
    env = make_rl_env("1-1", max_episode_steps=2000, seed=0)
    yield env
    env.close()


class _CountingEnv(gym.Env):
    """Counts how many times it was stepped, and can be told when to terminate."""

    action_space = gym.spaces.Discrete(7)
    observation_space = gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)

    def __init__(self, terminate_after: int | None = None) -> None:
        self.steps = 0
        self.terminate_after = terminate_after

    def reset(self, *, seed=None, options=None):  # noqa: D102
        self.steps = 0
        return np.zeros((2, 2, 3), np.uint8), {"x_pos": 0}

    def step(self, action):  # noqa: D102
        self.steps += 1
        terminated = (
            self.terminate_after is not None and self.steps >= self.terminate_after
        )
        return np.zeros((2, 2, 3), np.uint8), 1.0, terminated, False, {"x_pos": self.steps}


# ------------------------------------------------------------------------------ FrameSkip


def test_frame_skip_emulates_exactly_skip_frames() -> None:
    """One agent step must advance the world by ``skip`` frames, no more, no fewer."""
    inner = _CountingEnv()
    env = FrameSkip(inner, skip=4)
    env.reset()
    env.step(0)
    assert inner.steps == 4
    env.step(0)
    assert inner.steps == 8


def test_frame_skip_sums_reward_over_the_held_frames() -> None:
    """Dropping the intermediate rewards would silently scale the return by 1/skip."""
    env = FrameSkip(_CountingEnv(), skip=4)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward == 4.0


def test_frame_skip_stops_early_on_termination() -> None:
    """Stepping a finished episode would raise; the reward must cover only real frames."""
    inner = _CountingEnv(terminate_after=2)
    env = FrameSkip(inner, skip=4)
    env.reset()
    _, reward, terminated, _, _ = env.step(0)
    assert terminated
    assert inner.steps == 2
    assert reward == 2.0


@pytest.mark.parametrize("bad_skip", [0, -1])
def test_frame_skip_rejects_invalid_skip(bad_skip: int) -> None:
    """A skip of zero would emulate nothing and return a stale observation."""
    with pytest.raises(ValueError):
        FrameSkip(_CountingEnv(), skip=bad_skip)


def test_frame_skip_advances_the_real_emulator(base_env) -> None:
    """Against the real game: four skipped frames move Mario about four frames' worth."""
    base_env.reset(seed=0)
    for _ in range(20):
        base_env.step(3)
    unskipped_x = base_env.unwrapped._get_info()["x_pos"]

    env = FrameSkip(make_rl_env("1-1", max_episode_steps=2000, seed=0), skip=4)
    try:
        env.reset(seed=0)
        for _ in range(5):  # 5 agent steps x 4 frames = the same 20 frames
            _, _, _, _, info = env.step(3)
        assert info["x_pos"] == unskipped_x
    finally:
        env.close()


# ----------------------------------------------------------------------- NativeGrayResize


def test_gray_resize_shape_dtype_and_space(base_env) -> None:
    """NatureCNN expects 84x84 uint8; the declared space must match what is returned."""
    env = NativeGrayResize(base_env, size=84)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.shape == obs.shape
    assert env.observation_space.contains(obs)


def test_gray_resize_honours_a_custom_size(base_env) -> None:
    """Size is configurable, and the space must follow it."""
    env = NativeGrayResize(base_env, size=64)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 64, 1)


def test_native_grayscale_matches_an_rgb_conversion() -> None:
    """The native buffer must be the same image, not a different render path.

    Reading grayscale natively avoids an RGB->gray conversion in Python; this checks the
    shortcut is sound rather than merely fast. Needs ``render_mode='rgb_array'`` so the RGB
    frame is actually returned for comparison.
    """
    env = make_rl_env("1-1", max_episode_steps=2000, seed=0, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        for _ in range(30):
            env.step(3)
        native = env.unwrapped.observation(mode="grayscale")
        rgb = np.array(env.unwrapped.render(), copy=True)
        converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        assert native.shape == converted.shape
        # Luma coefficients differ between the emulator and OpenCV, so compare loosely.
        assert float(np.mean(np.abs(native.astype(int) - converted.astype(int)))) < 20
    finally:
        env.close()


def test_gray_resize_output_is_not_constant(base_env) -> None:
    """Guards against an all-zero observation, which trains happily and learns nothing."""
    env = NativeGrayResize(base_env)
    obs, _ = env.reset(seed=0)
    assert obs.min() != obs.max()


# ---------------------------------------------------------------------- EpisodeStatistics


def test_episode_statistics_reports_outcome_on_termination() -> None:
    """The Phase 5 table needs outcome fields, not just episode return."""
    env = EpisodeStatistics(_CountingEnv(terminate_after=3))
    env.reset()
    for _ in range(2):
        _, _, terminated, _, info = env.step(0)
        assert "episode_extra" not in info
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["episode_extra"]["max_x"] == 3


def test_episode_statistics_tracks_the_maximum_not_the_final_x() -> None:
    """``x_pos`` freezes during the death animation and can also go backwards.

    Reporting the final x would understate how far an agent actually got, so the wrapper
    keeps the maximum.
    """

    class _Retreating(_CountingEnv):
        def step(self, action):
            self.steps += 1
            x = {1: 100, 2: 250, 3: 120}[self.steps]
            return (
                np.zeros((2, 2, 3), np.uint8), 0.0, self.steps >= 3, False,
                {"x_pos": x, "is_dying": True},
            )

    env = EpisodeStatistics(_Retreating())
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(0)
    assert info["episode_extra"]["max_x"] == 250
    assert info["episode_extra"]["died"] is True


def test_episode_statistics_marks_a_flag_as_not_a_death() -> None:
    """Completion and death must not be conflated in the completion-rate metric."""

    class _Winning(_CountingEnv):
        def step(self, action):
            self.steps += 1
            return (
                np.zeros((2, 2, 3), np.uint8), 0.0, True, False,
                {"x_pos": 3161, "flag_get": True, "is_dying": False},
            )

    env = EpisodeStatistics(_Winning())
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["episode_extra"]["flag_get"] is True
    assert info["episode_extra"]["died"] is False
