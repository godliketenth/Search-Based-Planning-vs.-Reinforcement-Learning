"""Search-based planning: motion primitives, exact forward simulation and A*.

The planner treats the emulator as its transition function and its collision checker, so
every graph edge is dynamically feasible by construction and every collision check is exact.
"""

from .astar import SearchBudget, plan_level
from .heuristic import DistanceHeuristic
from .plan import Plan, SearchStats
from .primitives import DEFAULT_PRIMITIVES, MotionPrimitive, primitives_from_config
from .simulator import ForwardSimulator, SnapshotStore, Transition

__all__ = [
    "DEFAULT_PRIMITIVES",
    "DistanceHeuristic",
    "ForwardSimulator",
    "MotionPrimitive",
    "Plan",
    "SearchBudget",
    "SearchStats",
    "SnapshotStore",
    "Transition",
    "plan_level",
    "primitives_from_config",
]
