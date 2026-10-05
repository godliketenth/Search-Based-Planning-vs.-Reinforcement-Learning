"""Drawing the planner's view of the world onto the game frame.

This module exists twice over:

* **Phase 2** -- as the verification tool. A tile grid extracted from RAM is impossible to
  check by reading numbers; drawn on top of the actual rendered frame, a one-column
  misalignment is obvious at a glance.
* **Phase 5b** -- as the live demo's renderer. It therefore already accepts a planned path
  and a replan flash, which Phase 3 will supply. Drawing the plan is what makes planning
  *observable* instead of an invisible internal detail, which is the point of the demo.

Colours are RGB tuples and frames stay in RGB throughout; no ``cvtColor`` is applied, so
what is drawn is what the array means. Callers rendering through OpenCV windows convert
once, at the window boundary.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from src.env import ram_map
from src.env.state import WorldState

#: Native NES frame size.
FRAME_HEIGHT = 240
FRAME_WIDTH = 256

# RGB palette. Chosen to stay legible against Mario's blue sky and brown ground.
COLOR_SOLID = (255, 64, 64)       # obstacle cells -- C-obstacle
COLOR_MARIO = (64, 255, 64)       # the configuration being planned for
COLOR_ENEMY = (255, 160, 0)       # moving obstacles
COLOR_PATH = (80, 200, 255)       # the planned trajectory
COLOR_REPLAN = (255, 255, 0)      # flash when the plan is recomputed
COLOR_VELOCITY = (255, 255, 255)  # velocity vector
COLOR_HUD = (255, 255, 255)
COLOR_HUD_SHADOW = (0, 0, 0)

#: Pixels of velocity arrow per pixel/frame of speed, so the arrow is visible at all.
VELOCITY_ARROW_GAIN = 8.0

#: Mario's stored y (``0x00CE``) is the top of the *big* Mario bounding box and does not
#: change with powerup level, so his feet are always at ``y + 32``. Small Mario is drawn in
#: the lower half of that box. Verified visually for small Mario against the ground surface
#: at screen y 208; see docs/phase2-findings.md.
MARIO_BOX_BOTTOM_OFFSET = 32

#: Top-left corner and width of the HUD panel, placed below the game's own status bar.
HUD_ORIGIN = (3, 48)
HUD_WIDTH = 104
HUD_LINE_HEIGHT = 10


def _to_screen(x_level: int, y_screen: int, screen_left: int) -> tuple[int, int]:
    """Convert a level-absolute x and a screen y into frame pixel coordinates."""
    return int(x_level - screen_left), int(y_screen)


def draw_tiles(frame: np.ndarray, world: WorldState, alpha: float = 0.35) -> None:
    """Outline every solid tile, in place.

    This is the **obstacle map** -- the discretised C-obstacle region. If these rectangles
    do not land exactly on the bricks and pipes in the frame underneath, the tile-buffer
    ring arithmetic is wrong.

    Args:
        frame: RGB frame, modified in place.
        world: the state to draw.
        alpha: fill opacity for solid cells; outlines are drawn fully opaque.
    """
    solid = world.tiles.solid()
    overlay = frame.copy()
    for row in range(ram_map.TILE_ROWS):
        for col in range(ram_map.TILE_COLS):
            if not solid[row, col]:
                continue
            x0, y0 = _to_screen(
                world.tiles.level_x(col), world.tiles.screen_y(row), world.screen_left
            )
            x1, y1 = x0 + ram_map.TILE_SIZE, y0 + ram_map.TILE_SIZE
            cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_SOLID, thickness=-1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_SOLID, thickness=1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_enemies(frame: np.ndarray, world: WorldState) -> None:
    """Box every active enemy -- the dynamic obstacles -- in place.

    Enemy avoidance is never planned explicitly: because the emulator simulates enemies too,
    a primitive that would collide simply dies in forward simulation and the edge is
    discarded. Drawing them is for the viewer's benefit, not the planner's.
    """
    for enemy in world.enemies:
        x, y = _to_screen(enemy.x, enemy.y, world.screen_left)
        cv2.rectangle(frame, (x, y), (x + 16, y + 16), COLOR_ENEMY, thickness=1)
        cv2.putText(
            frame, f"{enemy.kind:02x}", (x, y - 2),
            cv2.FONT_HERSHEY_PLAIN, 0.5, COLOR_ENEMY, 1, cv2.LINE_AA,
        )


def draw_mario(frame: np.ndarray, world: WorldState) -> None:
    """Box Mario and draw his velocity vector, in place.

    The velocity arrow is the visible part of the configuration that a position-only
    formulation would throw away.
    """
    mario = world.mario
    height = 16 if mario.powerup == 0 else 32
    x, y_box_top = _to_screen(mario.x, world.mario_y_screen, world.screen_left)
    # Anchor the box at Mario's feet, not at the stored y -- see MARIO_BOX_BOTTOM_OFFSET.
    y_bottom = y_box_top + MARIO_BOX_BOTTOM_OFFSET
    y = y_bottom - height
    cv2.rectangle(frame, (x, y), (x + 16, y_bottom), COLOR_MARIO, thickness=1)

    centre = (x + 8, y + height // 2)
    tip = (
        int(centre[0] + mario.vx_pixels * VELOCITY_ARROW_GAIN),
        int(centre[1] + mario.vy * VELOCITY_ARROW_GAIN / 2),
    )
    if tip != centre:
        cv2.arrowedLine(frame, centre, tip, COLOR_VELOCITY, 1, cv2.LINE_AA, tipLength=0.3)


def draw_path(
    frame: np.ndarray,
    world: WorldState,
    path: Sequence[tuple[int, int]],
    dotted: bool = True,
) -> None:
    """Draw a planned trajectory ahead of Mario, in place.

    Args:
        frame: RGB frame, modified in place.
        world: current state, for the viewport offset.
        path: planned waypoints as ``(x_level, y_screen)`` pairs. Phase 3 produces these
            by forward-simulating the plan and recording positions.
        dotted: draw as dots rather than a solid line, so the path does not obscure the
            level geometry underneath it.
    """
    points = [_to_screen(x, y, world.screen_left) for x, y in path]
    visible = [(x, y) for x, y in points if -16 <= x <= FRAME_WIDTH + 16]
    if len(visible) < 2:
        return
    if dotted:
        for x, y in visible[::2]:
            cv2.circle(frame, (x, y), 1, COLOR_PATH, thickness=-1)
    else:
        cv2.polylines(frame, [np.array(visible, np.int32)], False, COLOR_PATH, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: Iterable[str], replan_flash: bool = False) -> None:
    """Write HUD text down the left edge, in place.

    Args:
        frame: RGB frame, modified in place.
        lines: text lines, already formatted by the caller.
        replan_flash: draw a border flash, used by the demo to mark the frame on which the
            plan was recomputed.
    """
    if replan_flash:
        cv2.rectangle(
            frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), COLOR_REPLAN, thickness=2
        )
    lines = list(lines)
    if not lines:
        return

    # Darken a panel behind the text; white-on-sky is otherwise unreadable, and the game's
    # own status bar occupies the top 40 rows.
    x0, y0 = HUD_ORIGIN
    panel = (x0 - 2, y0 - 9), (x0 + HUD_WIDTH, y0 + HUD_LINE_HEIGHT * len(lines) - 4)
    region = frame[panel[0][1]:panel[1][1], panel[0][0]:panel[1][0]]
    if region.size:
        region[:] = (region * 0.35).astype(np.uint8)

    for index, line in enumerate(lines):
        origin = (x0, y0 + index * HUD_LINE_HEIGHT)
        cv2.putText(frame, line, origin,
                    cv2.FONT_HERSHEY_PLAIN, 0.6, COLOR_HUD, 1, cv2.LINE_AA)


def default_hud_lines(world: WorldState, extra: dict[str, object] | None = None) -> list[str]:
    """Build the standard HUD: the configuration, plus whatever the agent wants to add.

    Args:
        world: the state to describe.
        extra: additional ``label -> value`` rows, e.g. nodes expanded or replan count.

    Returns:
        Formatted lines, ready for :func:`draw_hud`.
    """
    mario = world.mario
    lines = [
        f"x {mario.x:4d}  y {mario.y:4d}",
        f"vx {mario.vx_pixels:+5.2f}  vy {mario.vy:+3d}",
        f"ground {int(mario.on_ground)}  pow {mario.powerup}",
        f"enemies {len(world.enemies)}",
    ]
    if world.is_dying:
        lines.append("DYING")
    lines.extend(f"{key} {value}" for key, value in (extra or {}).items())
    return lines


def render_overlay(
    frame: np.ndarray,
    world: WorldState,
    *,
    path: Sequence[tuple[int, int]] | None = None,
    hud: dict[str, object] | None = None,
    replan_flash: bool = False,
    show_tiles: bool = True,
    show_enemies: bool = True,
    show_hud: bool = True,
    scale: int = 1,
) -> np.ndarray:
    """Draw the full debug overlay onto a copy of ``frame``.

    The one entry point used by both the Phase 2 inspector and the Phase 5b demo.

    Args:
        frame: an RGB frame from ``env.render()``. **Copied** before drawing -- the
            emulator hands back a reusable view of its screen buffer, so drawing in place
            would corrupt the emulator's own frame.
        world: the state to visualise.
        path: optional planned trajectory as ``(x_level, y_screen)`` waypoints.
        hud: extra HUD rows.
        replan_flash: flash the border, to mark a replan.
        show_tiles: draw the obstacle map.
        show_enemies: draw dynamic obstacles.
        show_hud: draw the text HUD.
        scale: integer upscaling factor applied last, with nearest-neighbour interpolation
            so the pixel art stays crisp.

    Returns:
        A new RGB frame with the overlay drawn.
    """
    out = np.array(frame, dtype=np.uint8, copy=True)
    if show_tiles:
        draw_tiles(out, world)
    if path:
        draw_path(out, world, path)
    if show_enemies:
        draw_enemies(out, world)
    draw_mario(out, world)
    if show_hud:
        draw_hud(out, default_hud_lines(world, hud), replan_flash=replan_flash)
    if scale > 1:
        out = cv2.resize(
            out, (out.shape[1] * scale, out.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
    return out
