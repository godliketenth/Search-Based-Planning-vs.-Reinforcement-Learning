"""PPO as an agent the evaluation harness can drive.

Both agents implement the same two-method interface so the Phase 5 harness is written once
and cannot accidentally favour either. The asymmetry stays in what they are *given*, not in
how they are *measured*: PPO sees an 84x84 grayscale stack, the planner sees RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    """Wraps a trained PPO policy behind ``reset()`` / ``act(obs)``.

    Attributes:
        deterministic: if ``True``, take the argmax action rather than sampling. Sampling is
            the training-time behaviour; evaluation normally wants the mode.
    """

    name = "ppo"

    def __init__(self, model_path: str | Path, device: str = "cpu",
                 deterministic: bool = True) -> None:
        """Load a saved model.

        Args:
            model_path: path to a ``.zip`` saved by SB3.
            device: torch device for inference.
            deterministic: whether to act greedily.
        """
        self.model = PPO.load(model_path, device=device)
        self.deterministic = deterministic
        self._state: Any = None

    def reset(self) -> None:
        """Clear any recurrent state. PPO with a CNN policy is stateless, so this is a no-op
        beyond keeping the interface uniform with the planner's executor."""
        self._state = None

    def act(self, observation: np.ndarray) -> int:
        """Choose an action for one observation.

        Args:
            observation: an ``(84, 84, 1)`` or stacked observation, as produced by the
                training wrappers.

        Returns:
            A ``SIMPLE_MOVEMENT`` action index.
        """
        action, self._state = self.model.predict(
            observation, state=self._state, deterministic=self.deterministic
        )
        return int(np.asarray(action).reshape(-1)[0])
