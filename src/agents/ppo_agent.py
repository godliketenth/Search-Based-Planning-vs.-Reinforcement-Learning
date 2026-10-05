"""The trained PPO policy, behind the shared agent interface.

Reads only ``observation.pixels`` -- the same 84x84 grayscale stack it was trained on. It
never touches ``observation.env``, which is what makes the asymmetry concrete: the planner
is *permitted* privileged state and chooses to use it; the policy is not and could not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.base import AgentDiagnostics, Observation
from src.rl.agent import PPOAgent as _PPOPolicy


class PPOEvalAgent:
    """Wraps a trained policy for the evaluation harness."""

    name = "ppo"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        training_frames: int = 0,
        training_seconds: float = 0.0,
    ) -> None:
        """Load the policy.

        Args:
            model_path: SB3 ``.zip``.
            device: torch device for inference.
            deterministic: act greedily rather than sampling.
            training_frames: emulated frames spent training. Carried into the results so the
                compute column compares like with like against the planner's search frames.
            training_seconds: wall-clock training time, likewise.
        """
        self._policy = _PPOPolicy(model_path, device=device, deterministic=deterministic)
        self._training_frames = training_frames
        self._training_seconds = training_seconds
        self.diagnostics = AgentDiagnostics()

    def reset(self) -> None:
        """Clear policy state and restate the training cost for this episode's record."""
        self._policy.reset()
        self.diagnostics.reset()
        # Training cost is paid once, before evaluation; it is reported as the agent's
        # compute cost so the comparison against A*'s search frames is explicit.
        self.diagnostics.frames_simulated = self._training_frames
        self.diagnostics.planning_seconds = self._training_seconds
        self.diagnostics.notes["cost_is_training_not_planning"] = True

    def act(self, observation: Observation) -> int:
        """Choose an action from pixels alone."""
        return self._policy.act(np.asarray(observation.pixels))
