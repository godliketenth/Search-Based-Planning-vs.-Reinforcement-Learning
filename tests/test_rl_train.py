"""Tests for training setup: device choice, checkpoint discovery and resume.

The resume path is the one worth protecting. A 5M-step run takes hours, so a resume that
silently restarts from zero -- or loads the wrong checkpoint -- costs a day and is easy not
to notice, because training "works" either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.rl.agent import PPOAgent
from src.rl.train import (
    CHECKPOINT_PREFIX,
    OutcomeCallback,
    TrainingPaths,
    latest_checkpoint,
    resolve_device,
)


class _RecordingLogger:
    """Captures what the callback logs. SB3 exposes `callback.logger` as a read-only
    property proxying `callback.model.logger`, so tests attach a stub model."""

    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


class _StubModel:
    """Minimal stand-in for the SB3 model a callback reads its logger from."""

    def __init__(self, logger: _RecordingLogger) -> None:
        self.logger = logger


def _attach_logger(callback: OutcomeCallback) -> _RecordingLogger:
    """Give ``callback`` a logger it can write to, and return it for assertions."""
    logger = _RecordingLogger()
    callback.model = _StubModel(logger)
    return logger


# ------------------------------------------------------------------------------- devices


def test_auto_prefers_an_accelerator_when_available() -> None:
    """Measured, MPS wins by 3.6x end-to-end here, so auto must not sit on cpu."""
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_unavailable_devices_fall_back_to_cpu() -> None:
    """A config written on one machine must still run on another."""
    if not torch.cuda.is_available():
        assert resolve_device("cuda") == "cpu"
    if not torch.backends.mps.is_available():
        assert resolve_device("mps") == "cpu"


def test_explicit_cpu_is_honoured() -> None:
    """The benchmark needs to force cpu even when an accelerator exists."""
    assert resolve_device("cpu") == "cpu"


# --------------------------------------------------------------------------- checkpoints


def test_latest_checkpoint_orders_numerically_not_lexicographically(tmp_path: Path) -> None:
    """``900000`` sorts after ``1000000`` as a string -- resume would go backwards."""
    for steps in (100_000, 900_000, 1_000_000):
        (tmp_path / f"{CHECKPOINT_PREFIX}_{steps}_steps.zip").touch()
    assert latest_checkpoint(tmp_path).name == f"{CHECKPOINT_PREFIX}_1000000_steps.zip"


def test_latest_checkpoint_is_none_when_there_are_none(tmp_path: Path) -> None:
    """``--resume`` on a fresh run must start training, not crash."""
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


def test_latest_checkpoint_ignores_unrelated_files(tmp_path: Path) -> None:
    """A final model or a stray file must not be mistaken for a checkpoint."""
    (tmp_path / "final_model.zip").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / f"{CHECKPOINT_PREFIX}_500_steps.zip").touch()
    assert latest_checkpoint(tmp_path).name == f"{CHECKPOINT_PREFIX}_500_steps.zip"


def test_training_paths_layout(tmp_path: Path) -> None:
    """Checkpoints, TensorBoard logs and the final model stay separate per run."""
    paths = TrainingPaths(tmp_path / "run")
    paths.ensure()
    assert paths.checkpoints.is_dir()
    assert paths.tensorboard.is_dir()
    assert paths.final_model.parent == paths.run_dir


# ------------------------------------------------------------------------------ callback


def test_outcome_callback_tracks_completion_and_distance() -> None:
    """Episode return hides the thing being measured; these are the Phase 5 metrics."""
    callback = OutcomeCallback(window=10)
    callback.locals = {
        "infos": [
            {"episode_extra": {"max_x": 600, "flag_get": False, "died": True, "time_left": 350}},
            {"episode_extra": {"max_x": 3161, "flag_get": True, "died": False, "time_left": 90}},
        ]
    }

    logger = _attach_logger(callback)
    assert callback._on_step() is True
    assert callback.best_x == 3161
    assert callback.flags == 1
    assert logger.records["mario/completion_rate"] == pytest.approx(0.5)
    assert logger.records["mario/mean_max_x"] == pytest.approx(1880.5)


def test_outcome_callback_ignores_steps_without_finished_episodes() -> None:
    """Most steps carry no ``episode_extra``; the callback must not invent data."""
    callback = OutcomeCallback()
    callback.locals = {"infos": [{}, {"x_pos": 100}]}

    logger = _attach_logger(callback)
    assert callback._on_step() is True
    assert logger.records == {}, "nothing should be logged without a finished episode"


def test_outcome_callback_window_is_bounded() -> None:
    """A rolling window, not an ever-growing list, over a multi-million-step run."""
    callback = OutcomeCallback(window=5)

    _attach_logger(callback)
    for index in range(20):
        callback.locals = {
            "infos": [
                {"episode_extra": {"max_x": index, "flag_get": False, "died": True, "time_left": 0}}
            ]
        }
        callback._on_step()
    assert len(callback.max_x_history) == 5
    assert callback.best_x == 19


# --------------------------------------------------------------------------------- agent


def test_ppo_agent_round_trips_through_a_saved_model(tmp_path: Path) -> None:
    """The harness loads a saved policy; it must act deterministically from one.

    Trains for a handful of steps only -- this checks the save/load/act path, not learning.
    """
    from stable_baselines3 import PPO

    from src.rl.vec_env import make_training_vec_env

    venv = make_training_vec_env(n_envs=1, max_episode_steps=100, force_dummy=True)
    try:
        model = PPO("CnnPolicy", venv, n_steps=64, batch_size=32, device="cpu", seed=0)
        model.learn(total_timesteps=64)
        path = tmp_path / "model.zip"
        model.save(path)
    finally:
        venv.close()

    agent = PPOAgent(path, device="cpu", deterministic=True)
    agent.reset()
    observation = np.zeros((4, 84, 84), dtype=np.uint8)
    action = agent.act(observation)
    assert isinstance(action, int)
    assert 0 <= action < 7
    assert agent.act(observation) == action, "deterministic mode must be repeatable"
