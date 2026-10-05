"""Tests for the debug overlay.

Mostly guarding one dangerous property: the overlay must never draw into the frame the
emulator handed it. ``render()`` returns a reusable view of the emulator's own screen
buffer, so drawing in place would corrupt the emulator's output and, in Phase 5b, the
recorded mp4.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env import ram_map
from src.env.actions import RIGHT_B
from src.env.factories import make_planner_env
from src.env.state import WorldState
from src.viz import debug_overlay


@pytest.fixture(scope="module")
def scene():
    """A world state and its frame, taken near the first pipe."""
    env = make_planner_env("1-1", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        while ram_map.read_x(env.ram) < 560:
            env._frame_advance(RIGHT_B)
        info = env._get_info()
        world = WorldState.from_env(env, info["flag_get"], info["time"])
        frame = np.array(env.render(), copy=True)
        yield frame, world
    finally:
        env.close()


def test_render_overlay_does_not_mutate_the_input_frame(scene) -> None:
    """The emulator reuses its screen buffer; drawing in place would corrupt it."""
    frame, world = scene
    original = frame.copy()
    debug_overlay.render_overlay(frame, world)
    assert np.array_equal(frame, original)


def test_render_overlay_actually_draws_something(scene) -> None:
    """A silently no-op overlay would be worse than none at all."""
    frame, world = scene
    out = debug_overlay.render_overlay(frame, world)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_scale_upsamples_by_an_integer_factor(scene) -> None:
    """The demo renders at 2-3x; nearest-neighbour keeps the pixel art crisp."""
    frame, world = scene
    out = debug_overlay.render_overlay(frame, world, scale=3)
    assert out.shape == (frame.shape[0] * 3, frame.shape[1] * 3, 3)


def test_layers_can_be_disabled_independently(scene) -> None:
    """The demo toggles these at runtime."""
    frame, world = scene
    bare = debug_overlay.render_overlay(
        frame, world, show_tiles=False, show_enemies=False, show_hud=False
    )
    full = debug_overlay.render_overlay(frame, world)
    assert not np.array_equal(bare, full)
    # Mario's box is always drawn, so even the bare version differs from the raw frame.
    assert not np.array_equal(bare, frame)


def test_path_is_drawn_when_supplied(scene) -> None:
    """Phase 3 supplies this; drawing the plan is the point of the demo."""
    frame, world = scene
    path = [(x, 200) for x in range(world.mario.x, world.mario.x + 160, 4)]
    without = debug_overlay.render_overlay(frame, world, show_hud=False)
    with_path = debug_overlay.render_overlay(frame, world, path=path, show_hud=False)
    assert not np.array_equal(without, with_path)


def test_offscreen_path_is_clipped_not_crashing(scene) -> None:
    """A receding-horizon plan routinely extends beyond the right edge of the screen."""
    frame, world = scene
    far = [(world.mario.x + 5000 + i * 4, 200) for i in range(40)]
    debug_overlay.render_overlay(frame, world, path=far)  # must not raise


def test_replan_flash_changes_the_border(scene) -> None:
    """The flash marks the frame on which the plan was recomputed."""
    frame, world = scene
    quiet = debug_overlay.render_overlay(frame, world)
    flashed = debug_overlay.render_overlay(frame, world, replan_flash=True)
    assert not np.array_equal(quiet[0, :, :], flashed[0, :, :])


def test_hud_lines_describe_the_configuration(scene) -> None:
    """The HUD is the viva's live explanation of what the planner is reasoning about."""
    _, world = scene
    lines = debug_overlay.default_hud_lines(world, {"nodes": 42})
    joined = " ".join(lines)
    # The HUD pads numbers to keep the columns steady as values change width.
    assert str(world.mario.x) in joined
    assert joined.startswith("x ")
    assert "vx" in joined and "vy" in joined
    assert "ground" in joined
    assert "nodes 42" in joined


def test_empty_hud_does_not_crash(scene) -> None:
    """Guards the panel-sizing arithmetic against an empty line list."""
    frame, _ = scene
    debug_overlay.draw_hud(frame.copy(), [])
