"""Environment layer: action encoding, env factories and action perturbation.

The planner and the RL agent drive the same emulator through deliberately different
interfaces; see :mod:`src.env.factories` for why.
"""

from .actions import ACTION_BYTES, ACTION_NAMES, N_ACTIONS, action_for_byte, byte_for_action
from .factories import make_planner_env, make_rl_env, parse_level
from .ram_map import Enemy, TileGrid, Velocity
from .state import MarioState, WorldState
from .sticky import StickyActionSampler, StickyActionWrapper

__all__ = [
    "ACTION_BYTES",
    "ACTION_NAMES",
    "Enemy",
    "MarioState",
    "N_ACTIONS",
    "StickyActionSampler",
    "StickyActionWrapper",
    "TileGrid",
    "Velocity",
    "WorldState",
    "action_for_byte",
    "byte_for_action",
    "make_planner_env",
    "make_rl_env",
    "parse_level",
]
