"""Mario's configuration space.

This module defines *what a state is* for the planner, which is the single most important
modelling decision in the project.

Why not pixels, and why not a grid cell
---------------------------------------
A naive formulation treats the problem as a grid: Mario occupies a tile, and the planner
searches a 4- or 8-connected grid of tiles. That is wrong here, and visibly so --
Phase 1's random agent demonstrated it by getting permanently stuck at the first pipe. The
reason is **kinodynamic constraints**: Mario carries momentum, gravity acts every frame, and
air control is limited, so he cannot move from a cell to an adjacent cell at will. Whether a
pipe is passable depends on how fast he is already moving and whether he is on the ground.

So the configuration is::

    q = (x, y, vx, vy, on_ground, powerup)

Position *and* velocity, i.e. a state in a kinodynamic C-space rather than a workspace
point. ``on_ground`` is included because it decides which controls are even available (a
jump cannot be initiated mid-air), and ``powerup`` because a big Mario has a taller
collision box and hence a different free space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import ram_map

#: Default lattice resolution for duplicate detection in Phase 3's A*.
#:
#: Continuous states are bucketed before being compared, because two states that differ by
#: one pixel are, for planning purposes, the same state -- without bucketing, A* would
#: expand a practically infinite number of near-identical nodes. The consequence is that the
#: planner is **resolution-complete**, not complete, and optimal only up to this resolution
#: and the primitive set. Coarser is faster but can merge genuinely different states (and so
#: miss a solution); finer is slower.
DEFAULT_X_RES = 8      # pixels -- half a tile
DEFAULT_Y_RES = 8      # pixels
DEFAULT_VX_RES = 8     # subpixels/frame, i.e. 0.5 px/frame


@dataclass(frozen=True, slots=True)
class MarioState:
    """A point in Mario's configuration space.

    Frozen and slotted: states are used as dictionary keys and created in large numbers
    inside the A* loop, so they must be hashable and cheap.

    Attributes:
        x: level-absolute horizontal position in pixels.
        y: level-absolute vertical position in pixels, increasing **downwards**, continuous
            across a fall out of the viewport.
        vx: horizontal velocity in 1/16 px per frame (see :class:`ram_map.Velocity`).
        vy: vertical velocity in whole px per frame, positive downwards.
        on_ground: whether Mario is standing on something.
        powerup: 0 small, 1 big, 2 fire.
    """

    x: int
    y: int
    vx: int
    vy: int
    on_ground: bool
    powerup: int

    @classmethod
    def from_ram(cls, ram: np.ndarray) -> MarioState:
        """Extract the configuration from a RAM view.

        Args:
            ram: the zero-copy ``env.ram`` array.

        Returns:
            The current :class:`MarioState`.
        """
        velocity = ram_map.read_velocity(ram)
        return cls(
            x=ram_map.read_x(ram),
            y=ram_map.read_y_level(ram),
            vx=velocity.vx_subpixels,
            vy=velocity.vy_pixels,
            on_ground=ram_map.read_on_ground(ram),
            powerup=ram_map.read_powerup(ram),
        )

    @classmethod
    def from_env(cls, env: Any) -> MarioState:
        """Extract the configuration from a planner env (convenience wrapper)."""
        return cls.from_ram(env.ram)

    @property
    def vx_pixels(self) -> float:
        """Horizontal velocity in pixels per frame."""
        return self.vx / ram_map.VX_SUBPIXELS_PER_PIXEL

    def lattice_key(
        self,
        x_res: int = DEFAULT_X_RES,
        y_res: int = DEFAULT_Y_RES,
        vx_res: int = DEFAULT_VX_RES,
    ) -> tuple[int, int, int, bool, int]:
        """Bucket this state onto the search lattice, for duplicate detection.

        ``vy`` is deliberately **excluded**: it is almost determined by ``y`` and
        ``on_ground`` under constant gravity, so including it would split nodes that are
        equivalent for planning and inflate the open list.

        Args:
            x_res: horizontal bucket width in pixels.
            y_res: vertical bucket width in pixels.
            vx_res: horizontal velocity bucket width in subpixels/frame.

        Returns:
            A hashable key. Two states sharing a key are treated as the same node.
        """
        return (
            self.x // x_res,
            self.y // y_res,
            self.vx // vx_res,
            self.on_ground,
            self.powerup,
        )


@dataclass(frozen=True, slots=True)
class WorldState:
    """Everything the planner and the debug overlay need about one instant.

    :class:`MarioState` is the configuration being searched over; the rest is the
    environment around it -- the obstacle map and the moving obstacles.

    Attributes:
        mario: Mario's configuration.
        tiles: the local obstacle map (about two screens wide).
        enemies: active enemies, as level-absolute positions.
        is_dying: whether this trajectory has already failed.
        flag_get: whether the goal has been reached.
        time: the in-game timer.
        screen_left: level x of the left edge of the viewport, for drawing overlays.
        mario_y_screen: Mario's y in screen pixels. Kept here rather than on
            :class:`MarioState` because it is a rendering detail, not a configuration
            coordinate -- the C-space uses the level-absolute ``y``.
    """

    mario: MarioState
    tiles: ram_map.TileGrid
    enemies: tuple[ram_map.Enemy, ...]
    is_dying: bool
    flag_get: bool
    time: int
    screen_left: int
    mario_y_screen: int

    @classmethod
    def from_env(cls, env: Any, flag_get: bool = False, time: int = 0) -> WorldState:
        """Read a full world state from a planner env.

        Args:
            env: a raw planner env.
            flag_get: goal flag, which lives outside the addresses mapped here and so is
                passed in from ``info`` by the caller.
            time: the in-game timer, likewise.

        Returns:
            The current :class:`WorldState`.
        """
        ram = env.ram
        return cls(
            mario=MarioState.from_ram(ram),
            tiles=ram_map.read_tile_grid(ram),
            enemies=ram_map.read_enemies(ram),
            is_dying=ram_map.read_is_dying(ram),
            flag_get=flag_get,
            time=time,
            screen_left=ram_map.read_screen_left(ram),
            mario_y_screen=ram_map.read_y_screen(ram),
        )
