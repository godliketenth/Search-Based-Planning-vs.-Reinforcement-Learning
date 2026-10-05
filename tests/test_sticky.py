"""Tests for sticky actions -- the perturbation the project's central result depends on.

If sticky actions do not actually perturb execution, the open-loop vs. closed-loop
comparison silently measures nothing, and both conditions would report identical numbers.
These tests use a stub environment rather than the emulator so they stay fast.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from src.env.sticky import StickyActionSampler, StickyActionWrapper


class _EchoEnv(gym.Env):
    """Minimal env that reports the action it was given, for testing the wrapper."""

    action_space = gym.spaces.Discrete(7)
    observation_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):  # noqa: D102
        return 0, {}

    def step(self, action):  # noqa: D102
        return 0, 0.0, False, False, {"executed": int(action)}


def test_zero_probability_is_a_pass_through() -> None:
    """p = 0 is the deterministic condition and must never alter an action."""
    sampler = StickyActionSampler(0.0, seed=0)
    commanded = [1, 3, 2, 4, 0, 6, 5, 5, 1]
    assert [sampler.filter(a) for a in commanded] == commanded
    assert sampler.n_overridden == 0


def test_probability_one_locks_onto_the_first_action() -> None:
    """p = 1 repeats forever, which is the degenerate upper bound of the perturbation."""
    sampler = StickyActionSampler(1.0, seed=0)
    assert [sampler.filter(a) for a in [3, 1, 2, 4, 0]] == [3, 3, 3, 3, 3]


def test_first_action_of_an_episode_is_never_overridden() -> None:
    """There is no previous action to repeat at an episode boundary."""
    sampler = StickyActionSampler(1.0, seed=0)
    assert sampler.filter(6) == 6
    assert sampler.n_overridden == 0


def test_reset_clears_previous_action_and_counters() -> None:
    """Stale state must not leak across episodes."""
    sampler = StickyActionSampler(1.0, seed=0)
    sampler.filter(3)
    sampler.filter(1)
    assert sampler.n_overridden == 1
    sampler.reset()
    assert sampler.filter(5) == 5  # not stuck on 3
    assert sampler.n_commands == 1
    assert sampler.n_overridden == 0


def test_same_seed_gives_identical_perturbation() -> None:
    """Reproducibility: a Phase 5 run must be repeatable from its seed."""
    commanded = [1, 3, 2, 4, 0, 6, 5] * 20
    a = StickyActionSampler(0.25, seed=7)
    b = StickyActionSampler(0.25, seed=7)
    assert [a.filter(x) for x in commanded] == [b.filter(x) for x in commanded]


def test_different_seeds_give_different_perturbation() -> None:
    """Otherwise the 'random' perturbation would be a fixed pattern."""
    commanded = [1, 3, 2, 4, 0, 6, 5] * 20
    a = StickyActionSampler(0.25, seed=1)
    b = StickyActionSampler(0.25, seed=2)
    assert [a.filter(x) for x in commanded] != [b.filter(x) for x in commanded]


def test_realised_rate_approaches_configured_probability() -> None:
    """The realised override rate is reported in the metrics; it must be honest."""
    sampler = StickyActionSampler(0.25, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20_000):
        sampler.filter(int(rng.integers(7)))
    assert sampler.realised_rate == pytest.approx(0.25, abs=0.02)


def test_realised_rate_is_zero_before_any_command() -> None:
    """Guards against a divide-by-zero in the demo HUD on frame 0."""
    assert StickyActionSampler(0.25).realised_rate == 0.0


@pytest.mark.parametrize("bad_p", [-0.1, 1.5])
def test_invalid_probability_rejected(bad_p: float) -> None:
    """A mistyped sticky probability must fail at construction, not skew a result."""
    with pytest.raises(ValueError):
        StickyActionSampler(bad_p)


def test_wrapper_perturbs_the_action_the_env_receives() -> None:
    """The wrapper must change what reaches the env, not merely what is logged."""
    env = StickyActionWrapper(_EchoEnv(), p=1.0, seed=0)
    env.reset()
    assert env.step(3)[4]["executed"] == 3
    assert env.step(1)[4]["executed"] == 3  # repeated, not the commanded 1


def test_wrapper_reset_clears_sampler_state() -> None:
    """Episode boundaries through the Gymnasium API must reset the sampler too."""
    env = StickyActionWrapper(_EchoEnv(), p=1.0, seed=0)
    env.reset()
    env.step(3)
    env.step(1)
    env.reset()
    assert env.step(5)[4]["executed"] == 5
    assert env.sampler.n_overridden == 0
