"""The one interface both agents implement.

The evaluation harness is written against this protocol and nothing else, so it is
structurally incapable of treating the planner and the policy differently: same env, same
action space, same step accounting, same perturbation, same metrics.

What it does *not* equalise -- deliberately
-------------------------------------------
:class:`Observation` carries **both** an 84x84 pixel stack and a handle on the live
environment. PPO reads the pixels; the planner reads RAM and takes emulator snapshots
through the handle. That is the project's **assumption asymmetry**, and putting both fields
in one object makes it explicit at the type level rather than burying it in two different
call signatures.

Stating it plainly, because the report has to: A* is given privileged state and a perfect
model of the dynamics; PPO is given pixels and no model. They are not like-for-like agents.
The harness measures them identically; it does not pretend they are equally endowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything an agent may look at on one decision.

    Attributes:
        pixels: the stacked, preprocessed frames PPO was trained on. Shape ``(4, 84, 84)``.
        env: the live environment. Planner agents reach ``env.unwrapped`` for RAM reads and
            ``dump_state()``; PPO never touches it. **Agents must not step this env** -- the
            harness owns stepping, so that every agent's step accounting is identical.
        frame: frames elapsed in this episode, for agents that follow a timed plan.
        info: the env's ``info`` dict from the previous step.
    """

    pixels: np.ndarray
    env: Any
    frame: int = 0
    info: dict = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """An agent the harness can evaluate.

    Attributes:
        name: short identifier used as the row key in results tables.
    """

    name: str

    def reset(self) -> None:
        """Prepare for a new episode. Called after the env is reset, before the first act."""
        ...

    def act(self, observation: Observation) -> int:
        """Choose a ``SIMPLE_MOVEMENT`` action index for this step.

        Args:
            observation: what the agent is allowed to see.

        Returns:
            An action index in ``0..6``.
        """
        ...


@dataclass
class AgentDiagnostics:
    """Per-episode counters an agent may expose, beyond the action it returned.

    Reported alongside the standard metrics so that the *cost* of each approach is visible,
    not just its success rate.

    Attributes:
        planning_calls: how many searches were run during the episode (0 for a reactive
            policy, 1 for an open-loop plan executed from cache, many for closed-loop).
        planning_seconds: total time spent searching.
        nodes_expanded: total search nodes expanded.
        frames_simulated: emulator frames burned on planning -- the planner's true compute
            cost, directly comparable to PPO's training frames.
        notes: anything else worth carrying into the results table.
    """

    planning_calls: int = 0
    planning_seconds: float = 0.0
    nodes_expanded: int = 0
    frames_simulated: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Zero the counters at the start of an episode."""
        self.planning_calls = 0
        self.planning_seconds = 0.0
        self.nodes_expanded = 0
        self.frames_simulated = 0
        self.notes = {}
