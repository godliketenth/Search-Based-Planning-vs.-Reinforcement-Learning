"""Reinforcement learning: PPO from pixels.

The model-free counterpart to :mod:`src.planning`. PPO sees 84x84 grayscale frames and no
model of the dynamics, and pays its cost in samples at training time rather than in search
at plan time.
"""

from .agent import PPOAgent
from .train import TrainingPaths, latest_checkpoint, resolve_device, train
from .vec_env import make_env_fn, make_training_vec_env
from .wrappers import EpisodeStatistics, FrameSkip, NativeGrayResize

__all__ = [
    "EpisodeStatistics",
    "FrameSkip",
    "NativeGrayResize",
    "PPOAgent",
    "TrainingPaths",
    "latest_checkpoint",
    "make_env_fn",
    "make_training_vec_env",
    "resolve_device",
    "train",
]
