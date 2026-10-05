"""The A* heuristic, and why it is admissible.

``h(s)`` estimates the remaining cost from ``s`` to the goal. Cost ``g`` is measured in
**emulated frames**, so the planner minimises time-to-goal, and ``h`` must also be in frames.

    h(s) = max(0, x_goal - s.x) / v_max

**Admissible** (never overestimates): the game hard-caps Mario's horizontal speed at
``v_max`` = 3 px/frame with B held -- measured directly in Phase 2, not assumed. No
trajectory can close horizontal distance faster than the cap, so no trajectory can reach
the goal in fewer frames than ``h`` says. A* with an admissible heuristic returns an
optimal path.

**Consistent** (monotone): ``h`` is 1-Lipschitz in ``x`` with constant ``1/v_max``, and any
edge of ``n`` frames moves ``x`` by at most ``n * v_max``. So
``h(s) <= cost(s, s') + h(s')`` for every edge. Consistency is the stronger property, and
it is what guarantees a node is never re-expanded after being closed -- which is exactly
what lets the planner release each node's 272 KB snapshot on expansion.

**What would break it:** adding any risk or clearance penalty to the edge cost makes ``h``
inadmissible unless ``h`` is rescaled to match, because ``h`` would then be measuring
frames while ``g`` measured frames-plus-penalty. Likewise setting ``x_goal`` beyond the
actual flag overestimates. Setting it short is safe but weakens the heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.env.ram_map import MAX_VX_PIXELS_PER_FRAME
from src.env.state import MarioState


@dataclass(frozen=True, slots=True)
class DistanceHeuristic:
    """Frames-to-goal lower bound, based on the hard horizontal speed cap.

    Attributes:
        goal_x: level x treated as the goal for estimation purposes. Only the *heuristic*
            uses this; the goal **test** is the emulator's own ``flag_get``, so a wrong
            value costs search efficiency, not correctness -- provided it is not an
            overestimate, which would cost admissibility.
        v_max: maximum horizontal speed in px/frame.
    """

    goal_x: int
    v_max: float = MAX_VX_PIXELS_PER_FRAME

    def __post_init__(self) -> None:
        if self.v_max <= 0:
            raise ValueError(f"v_max must be positive, got {self.v_max}")

    def __call__(self, state: MarioState) -> float:
        """Estimated remaining frames to the goal. Never negative."""
        return max(0.0, (self.goal_x - state.x) / self.v_max)
