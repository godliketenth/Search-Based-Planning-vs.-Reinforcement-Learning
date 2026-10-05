"""Sticky actions: the perturbation that makes open-loop planning fail.

The NES emulator is fully deterministic, so an A* plan is just a memorised button sequence
that replays to the flag every single time. Reporting that as a result would say nothing
about planning. Following Machado et al. (2018), each frame the *previously executed*
action is repeated instead of the commanded one with probability ``p``, which:

* breaks open-loop execution (the plan desynchronises from the world), and
* forces **closed-loop receding-horizon replanning** -- the project's central measurement.

Motion-planning framing: sticky actions model **actuation uncertainty**. The planner's
transition model is exact, but the executed control differs from the commanded control, so
a plan that was feasible at plan time is no longer feasible at execution time.

The same logic is needed at two different levels of the stack, so it lives in one place:

* :class:`StickyActionSampler` works on plain ints and is what the Phase 3/5 plan executor
  uses, because the planner drives ``_frame_advance`` directly and has no Gymnasium
  wrappers to hang behaviour off (Phase 0, Finding 4).
* :class:`StickyActionWrapper` is a Gymnasium ``ActionWrapper`` around the sampler, for PPO.

Both agents therefore face an identical perturbation process, which the comparison relies on.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class StickyActionSampler:
    """Repeat the previously executed action with probability ``p``.

    Deliberately agnostic about whether the ints it is handed are discrete action indices
    (RL) or joypad bytes (planner) -- it only ever repeats a value it was already given.

    Attributes:
        p: probability of repeating the previous action.
        n_commands: how many actions have been filtered since the last :meth:`reset`.
        n_overridden: how many of those were replaced by the previous action. Reported in
            the demo HUD and the Phase 5 metrics as a check that the realised sticky rate
            matches the configured one.
    """

    def __init__(self, p: float, seed: int | None = None) -> None:
        """Initialise the sampler.

        Args:
            p: repeat probability in ``[0, 1]``. ``0.0`` makes this a pass-through.
            seed: seed for the internal RNG. The RNG is independent of the environment's,
                so the sticky noise is reproducible without perturbing env seeding.

        Raises:
            ValueError: if ``p`` is outside ``[0, 1]``.
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"sticky probability must be in [0, 1], got {p}")
        self.p = float(p)
        self._rng = np.random.default_rng(seed)
        self._prev: int | None = None
        self.n_commands = 0
        self.n_overridden = 0

    def reset(self) -> None:
        """Clear the previous action and the counters, at an episode boundary.

        The first action of an episode is never overridden: there is nothing to repeat.
        """
        self._prev = None
        self.n_commands = 0
        self.n_overridden = 0

    def filter(self, commanded: int) -> int:
        """Map a commanded action to the action actually executed.

        Args:
            commanded: the action the agent asked for.

        Returns:
            Either ``commanded`` or the previously executed action.
        """
        commanded = int(commanded)
        self.n_commands += 1
        if self._prev is not None and self.p > 0.0 and self._rng.random() < self.p:
            self.n_overridden += 1
            # NOTE: `_prev` tracks what was *executed*, not what was commanded, so a run of
            # overrides keeps repeating the same stuck action rather than drifting.
            return self._prev
        self._prev = commanded
        return commanded

    @property
    def realised_rate(self) -> float:
        """Fraction of commands that were overridden (``0.0`` before any command)."""
        return self.n_overridden / self.n_commands if self.n_commands else 0.0


class StickyActionWrapper(gym.ActionWrapper):
    """Gymnasium wrapper applying :class:`StickyActionSampler` to discrete actions.

    Used for the RL agent. The planner does not use this -- see the module docstring.
    """

    def __init__(self, env: gym.Env, p: float, seed: int | None = None) -> None:
        """Wrap ``env`` so that its actions become sticky with probability ``p``."""
        super().__init__(env)
        self.sampler = StickyActionSampler(p, seed=seed)

    def action(self, action: int) -> int:
        """Return the action actually executed for a commanded ``action``."""
        return self.sampler.filter(action)

    def reset(self, **kwargs):  # noqa: D102 - inherits Gymnasium's contract
        self.sampler.reset()
        return self.env.reset(**kwargs)
