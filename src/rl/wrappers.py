"""Observation and timing wrappers for the RL agent.

These exist for two reasons, one standard and one measured:

**Standard.** PPO learns from pixels, so the 240x256 RGB frame is reduced to the usual
84x84 grayscale NatureCNN input, and one action is held for several frames so that an
"agent step" is a decision rather than a single frame.

**Measured.** With ``SubprocVecEnv`` every observation is pickled through a pipe. A raw RGB
frame is 184 KB; at 84x84 grayscale it is 6.9 KB. Doing this reduction *inside the worker*
raised 8-worker throughput from 2,823 to 7,679 frames/s -- a 2.7x difference that comes
entirely from not shipping pixels we were about to throw away. See
``docs/phase4-findings.md``.

Gymnasium's own wrappers are not used here: ``FrameStackObservation`` raises at ``reset``
on this observation shape, and ``ResizeObservation`` returns ``(84, 84)`` while declaring
``(84, 84, 1)``. Frame stacking happens at the vector-env level instead (see
:mod:`src.rl.vec_env`), which is both the standard SB3 recipe and cheaper over the pipe.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import cv2
import gymnasium as gym
import numpy as np

#: NatureCNN's expected input size.
DEFAULT_FRAME_SIZE = 84

#: Frames per agent decision. 4 is the Atari convention and keeps the action-repeat
#: coarse enough that PPO is not forced to re-decide every 1/60 s.
DEFAULT_SKIP = 4


class FrameSkip(gym.Wrapper):
    """Hold one action for several frames, summing the reward.

    **Sticky actions interact with this deliberately.** ``StickyActionWrapper`` is applied
    *inside* this wrapper (it is part of the env this wraps), so each held frame is
    perturbed independently -- the perturbation is **frame-level**, matching the granularity
    the A* plan executor uses. That is a deliberate departure from the ALE convention of
    perturbing once per agent step, so that both agents face an identical process and the
    Phase 5 comparison stays like-for-like.
    """

    def __init__(self, env: gym.Env, skip: int = DEFAULT_SKIP) -> None:
        """Wrap ``env`` so each step emulates ``skip`` frames.

        Args:
            env: the environment to wrap.
            skip: frames per agent step. Must be at least 1.

        Raises:
            ValueError: if ``skip`` is below 1.
        """
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.skip = skip

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict]:
        """Apply ``action`` for ``skip`` frames, or until the episode ends.

        Returns:
            The usual 5-tuple, with reward summed over the frames actually emulated.
        """
        total_reward = 0.0
        terminated = truncated = False
        obs: Any = None
        info: dict = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class NativeGrayResize(gym.ObservationWrapper):
    """Reduce the frame to an ``(size, size, 1)`` uint8 grayscale image.

    Reads the emulator's **native** grayscale buffer rather than converting the RGB frame in
    Python (Phase 0, Finding 5), so the colour conversion never happens at all.
    """

    def __init__(self, env: gym.Env, size: int = DEFAULT_FRAME_SIZE) -> None:
        """Wrap ``env``.

        Args:
            env: the environment to wrap.
            size: output edge length in pixels.
        """
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )

    def observation(self, observation: Any) -> np.ndarray:
        """Return the current frame as ``(size, size, 1)`` uint8.

        The passed-in observation is ignored: the native grayscale buffer describes the same
        frame more cheaply. ``INTER_AREA`` is used because this is a downscale, where it
        avoids the aliasing that ``INTER_LINEAR`` would introduce on single-pixel sprites.
        """
        gray = self.env.unwrapped.observation(mode="grayscale")
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return small[:, :, None]


class EpisodeStatistics(gym.Wrapper):
    """Record per-episode outcome fields the Phase 5 metrics need.

    SB3's ``Monitor`` records return and length; this adds the task-specific outcome --
    how far Mario got, whether he reached the flag, and whether he died -- into ``info``
    under ``episode_extra`` when an episode ends.
    """

    def __init__(self, env: gym.Env) -> None:
        """Wrap ``env``."""
        super().__init__(env)
        self._max_x = 0

    def reset(self, **kwargs) -> tuple[Any, dict]:  # noqa: D102 - Gymnasium's contract
        obs, info = self.env.reset(**kwargs)
        self._max_x = int(info.get("x_pos", 0))
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict]:
        """Step, tracking the furthest x reached this episode."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._max_x = max(self._max_x, int(info.get("x_pos", 0)))
        if terminated or truncated:
            info["episode_extra"] = {
                "max_x": self._max_x,
                "flag_get": bool(info.get("flag_get", False)),
                # x_pos freezes during the death animation, so death is read from the
                # game's own flags, never inferred from position (Phase 0, Finding 6).
                "died": bool(info.get("is_dying", False) or info.get("is_dead", False)),
                "time_left": int(info.get("time", 0)),
            }
        return obs, reward, terminated, truncated, info
