"""A uniformly random policy: the floor every other agent has to beat.

Phase 1 established what this is worth: it stalls permanently at x ~ 594, the first pipe, on
every seed tested -- and does not even die there. Clearing a pipe needs a *committed*
multi-frame manoeuvre (jump while already moving right, and hold right through the whole
arc), and a controller that re-samples its button every frame cannot represent commitment.

That is the kinodynamic constraint stated as a baseline, and the direct motivation for
searching over motion primitives rather than over individual button presses.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import AgentDiagnostics, Observation
from src.env.actions import N_ACTIONS


class RandomAgent:
    """Samples actions uniformly, ignoring the observation entirely."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        """Seed the action sampler so a run is reproducible."""
        self._rng = np.random.default_rng(seed)
        self.diagnostics = AgentDiagnostics()

    def reset(self) -> None:
        """Nothing to carry between episodes."""
        self.diagnostics.reset()

    def act(self, observation: Observation) -> int:
        """Return a uniformly random action."""
        return int(self._rng.integers(N_ACTIONS))
