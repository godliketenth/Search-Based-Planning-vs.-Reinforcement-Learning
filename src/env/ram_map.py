"""Reading Mario's configuration from NES RAM.

``_get_info()`` already returns 40+ parsed fields, but three things the planner needs are
missing, and all three are what turn a *pixel* problem into a *motion planning* problem:

* **velocity** ``(vx, vy)`` -- without it the state is not a configuration in a kinodynamic
  C-space, only a position. Two Marios at the same ``x`` with different ``vx`` have
  completely different reachable sets.
* the **tile grid** -- the obstacle map, i.e. C-obstacle in discretised form.
* **enemy positions** -- ``info`` exposes enemy *types* only, not where they are.

Every address here was verified empirically against observed behaviour (see
``docs/phase2-findings.md``); none is taken on trust from a published RAM map.

Performance
-----------
These reads sit in the A* inner loop, so they use numpy fancy-indexing over the zero-copy
``env.ram`` view with precomputed index arrays (~0.2 us per read). Note this **contradicts**
a Phase 0 suggestion: ``ram_values()`` measured ~14 us, roughly 80x slower, because of
per-call native marshalling. Either is negligible beside a 660 us ``frame_advance``, but the
cheap one is also the simpler one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------- addresses

#: Mario's horizontal position: page * 256 + offset. Matches ``info['x_pos']``.
ADDR_X_PAGE = 0x6D
ADDR_X_SCREEN = 0x86

#: Level x of the left edge of the viewport, used to align the tile grid to the screen.
ADDR_SCREEN_EDGE_X = 0x071C

#: Horizontal velocity, signed, in **1/16 pixel per frame**. Max running speed is 0x30 (48),
#: i.e. 3 px/frame; max walking speed is 0x18 (24).
ADDR_VX = 0x0057

#: Vertical velocity, signed, in **whole pixels per frame**, positive downwards. Terminal
#: fall speed is +5; a jump starts at about -4.
ADDR_VY = 0x009F

#: Mario's "float state": 0 on the ground, 1 airborne, 3 on the flagpole. This is the
#: ``on_ground`` predicate -- a genuine part of the configuration, because whether a jump
#: can be initiated at all depends on it.
ADDR_FLOAT_STATE = 0x001D

#: Vertical viewport page (1 = on screen, >1 = below it, i.e. falling down a pit).
ADDR_Y_VIEWPORT = 0x00B5

#: Mario's y in screen pixels, measured downwards from the top of the screen.
ADDR_Y_PIXEL = 0x00CE

#: Player state. ``0x08`` normal, ``0x0B`` dying, ``0x06`` dead, ``0x04`` entering a pipe.
ADDR_PLAYER_STATE = 0x000E

#: Powerup level: 0 small, 1 big, 2 fire. Part of the configuration, because a big Mario
#: has a taller collision box and therefore a different free space.
ADDR_POWERUP = 0x0756

#: Player states meaning "this trajectory has failed".
PLAYER_STATE_DYING = 0x0B
PLAYER_STATE_DEAD = 0x06

#: Tile buffer: two screens of 13 rows x 16 columns, as a ring buffer over level pages.
TILE_BUFFER_START = 0x0500
TILE_ROWS = 13
TILE_COLS = 16
TILE_PAGE_BYTES = TILE_ROWS * TILE_COLS  # 208
TILE_SIZE = 16

#: Screen y of the top row of the tile buffer. Rows below this are the status bar.
TILE_TOP_Y = 32

#: Enemy slots. Each is a parallel array of five bytes.
ENEMY_SLOTS = 5
ADDR_ENEMY_ACTIVE = 0x000F
ADDR_ENEMY_TYPE = 0x0016
ADDR_ENEMY_X_PAGE = 0x006E
ADDR_ENEMY_X_SCREEN = 0x0087
ADDR_ENEMY_Y = 0x00CF

#: Velocity unit conversion: ``ADDR_VX`` counts sixteenths of a pixel per frame.
VX_SUBPIXELS_PER_PIXEL = 16

#: Maximum horizontal speed, in pixels/frame, with B held. This is the hard cap the game
#: imposes, and it is exactly what makes the Phase 3 heuristic ``(x_goal - x) / v_max``
#: admissible: no trajectory can cover ground faster than this.
MAX_VX_PIXELS_PER_FRAME = 3.0

#: Tile bytes that are *not* solid despite being non-zero. Coins occupy a tile slot but can
#: be walked through. Everything else non-zero (ground 0x54, brick 0x51/0x52, question
#: block 0xC0/0xC1, pipe 0x12-0x15, underground scenery 0x5C/0x5D/0x61) blocks movement.
NON_SOLID_TILES = frozenset({0x00, 0xC2})

# Precomputed index arrays, so a per-frame read is a single numpy gather.
_ENEMY_ACTIVE_IDX = np.arange(ADDR_ENEMY_ACTIVE, ADDR_ENEMY_ACTIVE + ENEMY_SLOTS)
_ENEMY_TYPE_IDX = np.arange(ADDR_ENEMY_TYPE, ADDR_ENEMY_TYPE + ENEMY_SLOTS)
_ENEMY_X_PAGE_IDX = np.arange(ADDR_ENEMY_X_PAGE, ADDR_ENEMY_X_PAGE + ENEMY_SLOTS)
_ENEMY_X_SCREEN_IDX = np.arange(ADDR_ENEMY_X_SCREEN, ADDR_ENEMY_X_SCREEN + ENEMY_SLOTS)
_ENEMY_Y_IDX = np.arange(ADDR_ENEMY_Y, ADDR_ENEMY_Y + ENEMY_SLOTS)

_ROW_OFFSETS = (np.arange(TILE_ROWS) * TILE_COLS).reshape(TILE_ROWS, 1)


def _signed(value: int) -> int:
    """Interpret an unsigned byte as a two's-complement signed byte."""
    value = int(value)
    return value - 256 if value > 127 else value


class Velocity(NamedTuple):
    """Mario's velocity, in the units the hardware stores.

    The two components use *different* units, which is a real hardware quirk and a classic
    source of silent bugs, so they are named rather than packed into a bare tuple.

    Attributes:
        vx_subpixels: horizontal velocity in 1/16 px per frame, positive rightwards.
        vy_pixels: vertical velocity in whole px per frame, **positive downwards**.
    """

    vx_subpixels: int
    vy_pixels: int

    @property
    def vx_pixels(self) -> float:
        """Horizontal velocity in pixels per frame."""
        return self.vx_subpixels / VX_SUBPIXELS_PER_PIXEL


class Enemy(NamedTuple):
    """One active enemy -- a moving obstacle.

    Attributes:
        slot: which of the five hardware slots this came from.
        kind: enemy type byte (``0x06`` is a Goomba).
        x: level-absolute horizontal position, directly comparable to ``info['x_pos']``.
        y: vertical position in screen pixels, measured downwards.
    """

    slot: int
    kind: int
    x: int
    y: int


class TileGrid(NamedTuple):
    """The visible obstacle map, aligned to the level's 16-pixel tile lattice.

    Attributes:
        tiles: ``(13, 16)`` array of tile bytes. ``tiles[row, col]``.
        col0: level tile column of ``col == 0``, so the level-absolute x of a column is
            ``(col0 + col) * 16``.
    """

    tiles: np.ndarray
    col0: int

    def level_x(self, col: int) -> int:
        """Level-absolute x (left edge, in pixels) of a grid column."""
        return (self.col0 + col) * TILE_SIZE

    def screen_y(self, row: int) -> int:
        """Screen y (top edge, in pixels) of a grid row."""
        return TILE_TOP_Y + row * TILE_SIZE

    def solid(self) -> np.ndarray:
        """Boolean ``(13, 16)`` occupancy mask: ``True`` where movement is blocked."""
        return is_solid(self.tiles)


def read_x(ram: np.ndarray) -> int:
    """Return Mario's level-absolute horizontal position (matches ``info['x_pos']``)."""
    return int(ram[ADDR_X_PAGE]) * 256 + int(ram[ADDR_X_SCREEN])


def read_screen_left(ram: np.ndarray) -> int:
    """Return the level x of the left edge of the viewport."""
    left_offset = (int(ram[ADDR_X_SCREEN]) - int(ram[ADDR_SCREEN_EDGE_X])) % 256
    return read_x(ram) - left_offset


def read_velocity(ram: np.ndarray) -> Velocity:
    """Read Mario's velocity. See :class:`Velocity` for the (differing) units."""
    return Velocity(_signed(ram[ADDR_VX]), _signed(ram[ADDR_VY]))


def read_on_ground(ram: np.ndarray) -> bool:
    """Whether Mario is standing on something.

    Part of the configuration, not a derived convenience: a jump primitive is only
    applicable when this is ``True``, which is precisely a kinodynamic constraint on the
    available controls.
    """
    return int(ram[ADDR_FLOAT_STATE]) == 0


def read_y_screen(ram: np.ndarray) -> int:
    """Mario's y in screen pixels, measured downwards from the top of the screen."""
    return int(ram[ADDR_Y_PIXEL])


def read_y_level(ram: np.ndarray) -> int:
    """Mario's level-absolute y, continuous across a fall off the bottom of the screen.

    ``info['y_pos']`` is screen-relative *and* inverted (``255 - y_pixel``), so it is not
    usable as a C-space coordinate: it wraps when Mario falls into a pit. This composes the
    vertical viewport page with the pixel offset instead, giving a value that increases
    monotonically downwards and never wraps.

    The overworld levels do not scroll vertically, so within a level the viewport page is 1
    except while falling into a pit.
    """
    return int(ram[ADDR_Y_VIEWPORT]) * 256 + int(ram[ADDR_Y_PIXEL])


def read_powerup(ram: np.ndarray) -> int:
    """Powerup level: 0 small, 1 big, 2 fire."""
    return int(ram[ADDR_POWERUP])


def read_player_state(ram: np.ndarray) -> int:
    """Raw player-state byte. ``0x08`` is normal play."""
    return int(ram[ADDR_PLAYER_STATE])


def read_is_dying(ram: np.ndarray) -> bool:
    """The planner's edge-invalidity signal: has this trajectory killed Mario?

    Two distinct failures, and missing the second one is an easy mistake:

    * ``player_state == 0x0B`` -- the death animation, set on the frame of contact with an
      enemy or hazard (Phase 0, Finding 6).
    * ``y_viewport > 1`` -- Mario has fallen below the screen, i.e. into a pit. This fires
      *before* the death animation, so a collision checker that only tests ``player_state``
      keeps expanding nodes for a Mario who is already doomed.

    Note that ``x_pos`` freezes once this is True, so position must never be used to infer
    death -- that produced a false-positive test during the Phase 0 spike.
    """
    return (
        int(ram[ADDR_PLAYER_STATE]) == PLAYER_STATE_DYING
        or int(ram[ADDR_Y_VIEWPORT]) > 1
    )


def read_enemies(ram: np.ndarray) -> tuple[Enemy, ...]:
    """Read every currently active enemy.

    Returns:
        A tuple of :class:`Enemy`, one per occupied slot. Empty when nothing is on screen.
    """
    active = ram[_ENEMY_ACTIVE_IDX]
    if not active.any():
        return ()
    kinds = ram[_ENEMY_TYPE_IDX]
    pages = ram[_ENEMY_X_PAGE_IDX]
    screens = ram[_ENEMY_X_SCREEN_IDX]
    ys = ram[_ENEMY_Y_IDX]
    return tuple(
        Enemy(slot, int(kinds[slot]), int(pages[slot]) * 256 + int(screens[slot]), int(ys[slot]))
        for slot in range(ENEMY_SLOTS)
        if active[slot]
    )


def read_tile_grid(ram: np.ndarray) -> TileGrid:
    """Read the visible obstacle map out of the scrolling tile buffer.

    The hardware keeps **two screens** of tiles in a ring buffer indexed by level page, so
    ``TILE_BUFFER_START + (page % 2) * 208`` selects the half holding a given column. This
    function resolves that wrap and returns a grid aligned to the viewport.

    Important limitation for Phase 3: the buffer only ever holds about two screens, so this
    is a **local** obstacle map, not a global one -- the planner cannot see tiles far ahead.
    That is acceptable because collision checking is done by exact forward simulation, not
    by consulting this grid; the grid is for state features, the heuristic and visualisation.

    Returns:
        A :class:`TileGrid` whose ``col0`` records which level tile column it starts at.
    """
    col0 = read_screen_left(ram) // TILE_SIZE
    level_cols = col0 + np.arange(TILE_COLS)
    page = (level_cols // TILE_COLS) % 2
    col_in_page = level_cols % TILE_COLS
    # (13, 16) gather: row offset down the page, plus the page-resolved column.
    indices = TILE_BUFFER_START + page * TILE_PAGE_BYTES + col_in_page + _ROW_OFFSETS
    return TileGrid(tiles=np.asarray(ram[indices]), col0=int(col0))


def tile_at(ram: np.ndarray, x_level: int, y_screen: int) -> int:
    """Read the single tile byte covering a level-absolute point.

    Args:
        x_level: level-absolute x in pixels.
        y_screen: screen y in pixels, measured downwards.

    Returns:
        The tile byte, or ``0`` for points above the tile area or below the level.
    """
    row = (int(y_screen) - TILE_TOP_Y) // TILE_SIZE
    if not 0 <= row < TILE_ROWS:
        return 0
    level_col = int(x_level) // TILE_SIZE
    page = (level_col // TILE_COLS) % 2
    index = TILE_BUFFER_START + page * TILE_PAGE_BYTES + row * TILE_COLS + level_col % TILE_COLS
    return int(ram[index])


def is_solid(tiles: np.ndarray | int) -> np.ndarray | bool:
    """Whether tile bytes block movement.

    Non-zero means "a tile is drawn here"; the only drawn-but-passable tile in the levels
    this project uses is the coin (``0xC2``).

    Args:
        tiles: a tile byte or an array of them.

    Returns:
        A bool, or a boolean array of the same shape.
    """
    if np.isscalar(tiles):
        return int(tiles) not in NON_SOLID_TILES
    tiles = np.asarray(tiles)
    blocked = tiles != 0
    for passable in NON_SOLID_TILES:
        if passable:
            blocked &= tiles != passable
    return blocked
