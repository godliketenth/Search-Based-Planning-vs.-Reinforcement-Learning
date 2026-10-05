"""Tests for the vectorised training environment.

These use ``force_dummy`` so they run in-process: the point is the observation pipeline and
the wrapper ordering, not the parallelism, and spawning subprocesses in a test suite is slow
and competes with anything else running.
"""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.rl.vec_env import DEFAULT_N_STACK, make_env_fn, make_training_vec_env
from src.rl.wrappers import DEFAULT_FRAME_SIZE


@pytest.fixture(scope="module")
def venv():
    """A two-worker in-process vec env."""
    environment = make_training_vec_env(n_envs=2, force_dummy=True, max_episode_steps=500)
    yield environment
    environment.close()


def test_env_fn_builds_a_preprocessed_env() -> None:
    """The thunk must produce the 84x84 observation, not the raw RGB frame."""
    env = make_env_fn(level="1-1", max_episode_steps=500)()
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE, 1)
        assert obs.dtype == np.uint8
    finally:
        env.close()


def test_observation_is_channels_first_and_stacked(venv) -> None:
    """NatureCNN wants ``(C, H, W)``, and C is the frame stack."""
    obs = venv.reset()
    assert obs.shape == (2, DEFAULT_N_STACK, DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE)
    assert obs.dtype == np.uint8
    assert venv.observation_space.shape == (
        DEFAULT_N_STACK, DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE,
    )


def test_newest_frame_is_last_in_the_stack(venv) -> None:
    """Pins the stacking convention.

    On reset the buffer is zero-filled and the current frame is placed at the end, so the
    earlier channels are blank. If this ever reversed, the policy would be fed its history
    backwards -- which trains without error and simply learns worse.
    """
    obs = venv.reset()
    assert obs[0, -1].max() > 0, "the newest frame should hold the current image"
    assert obs[0, 0].max() == 0, "older slots should still be empty right after reset"


def test_stack_fills_as_steps_accumulate(venv) -> None:
    """After enough steps every slot holds a real frame."""
    venv.reset()
    for _ in range(DEFAULT_N_STACK + 1):
        obs, _, _, _ = venv.step(np.full(2, 3))
    assert all(obs[0, channel].max() > 0 for channel in range(DEFAULT_N_STACK))


def test_frames_differ_across_the_stack(venv) -> None:
    """Stacking exists so velocity is observable; identical frames would defeat it."""
    venv.reset()
    for _ in range(12):
        obs, _, _, _ = venv.step(np.full(2, 3))
    assert not np.array_equal(obs[0, 0], obs[0, -1])


def test_step_returns_one_entry_per_worker(venv) -> None:
    """Basic vec-env contract, relied on by PPO's rollout buffer."""
    venv.reset()
    obs, rewards, dones, infos = venv.step(np.full(2, 3))
    assert obs.shape[0] == 2
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    assert len(infos) == 2


def test_single_env_uses_dummy_vec_env() -> None:
    """A subprocess for one worker is pure IPC overhead with no parallelism to gain."""
    environment = make_training_vec_env(n_envs=1, max_episode_steps=200)
    try:
        assert isinstance(environment.venv.venv, DummyVecEnv)
    finally:
        environment.close()


def test_force_dummy_avoids_subprocesses() -> None:
    """Tests and the live demo both want a single process."""
    environment = make_training_vec_env(n_envs=2, force_dummy=True, max_episode_steps=200)
    try:
        assert isinstance(environment.venv.venv, DummyVecEnv)
        assert not isinstance(environment.venv.venv, SubprocVecEnv)
    finally:
        environment.close()


def test_sticky_actions_are_applied_when_requested() -> None:
    """The stochastic evaluation condition must reach the training env too."""
    environment = make_training_vec_env(
        n_envs=1, sticky_p=0.25, max_episode_steps=200, seed=0
    )
    try:
        environment.reset()
        for _ in range(20):
            environment.step(np.full(1, 3))
    finally:
        environment.close()


def test_workers_get_different_seeds() -> None:
    """Identical workers would collect identical, redundant rollouts."""
    first = make_env_fn(seed=0, rank=0, max_episode_steps=200)()
    second = make_env_fn(seed=0, rank=1, max_episode_steps=200)()
    try:
        assert first.unwrapped is not second.unwrapped
    finally:
        first.close()
        second.close()
