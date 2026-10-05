"""Tests for RAM extraction.

The planner's correctness rests entirely on these reads. A wrong tile address does not
crash -- it produces an obstacle map that is plausible but shifted, and A* then plans
confidently through walls. So the tests check each quantity against something observable
and independent of the address being tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env import ram_map
from src.env.actions import A as JUMP
from src.env.actions import NOOP, RIGHT_A_B, RIGHT_B
from src.env.factories import make_planner_env


@pytest.fixture(scope="module")
def env():
    """A reset raw emulator shared across tests."""
    environment = make_planner_env("1-1")
    environment.reset(seed=0)
    yield environment
    environment.close()


def _advance(env, byte: int, frames: int) -> None:
    for _ in range(frames):
        env._frame_advance(byte)


# ------------------------------------------------------------------------------- helpers


def test_signed_interprets_twos_complement() -> None:
    """Velocities are signed bytes; reading them unsigned silently inverts leftward motion."""
    assert ram_map._signed(0) == 0
    assert ram_map._signed(48) == 48
    assert ram_map._signed(127) == 127
    assert ram_map._signed(128) == -128
    assert ram_map._signed(255) == -1


def test_is_solid_scalar_and_array() -> None:
    """Empty space and coins are passable; ground, brick, pipe and blocks are not."""
    assert not ram_map.is_solid(0x00)
    assert not ram_map.is_solid(0xC2), "coins occupy a tile but can be walked through"
    for solid in (0x54, 0x51, 0x52, 0xC0, 0xC1, 0x12, 0x15, 0x61):
        assert ram_map.is_solid(solid), f"{solid:#04x} should be solid"

    array = np.array([[0x00, 0x54], [0xC2, 0xC0]], dtype=np.uint8)
    assert ram_map.is_solid(array).tolist() == [[False, True], [False, True]]


# ------------------------------------------------------------------------------ position


def test_read_x_matches_info(env) -> None:
    """Our composed x must agree with the package's own, independently derived value."""
    env.reset(seed=0)
    for _ in range(3):
        _advance(env, RIGHT_B, 30)
        assert ram_map.read_x(env.ram) == env._get_info()["x_pos"]


def test_screen_left_brackets_mario(env) -> None:
    """The viewport must contain Mario, or every overlay coordinate is wrong."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 120)
    screen_left = ram_map.read_screen_left(env.ram)
    assert screen_left <= ram_map.read_x(env.ram) <= screen_left + 256


def test_y_level_is_composed_from_viewport_and_pixel(env) -> None:
    """``info['y_pos']`` is inverted and screen-relative, so we compose our own."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 40)
    ram = env.ram
    assert ram_map.read_y_level(ram) == int(ram[ram_map.ADDR_Y_VIEWPORT]) * 256 + int(
        ram[ram_map.ADDR_Y_PIXEL]
    )
    # Our y increases downwards; the package's y_pos increases upwards. They must disagree
    # in sign of change, which is exactly why y_pos is unusable as a C-space coordinate.
    assert ram_map.read_y_screen(ram) == 255 - env._get_info()["y_pos"]


# ------------------------------------------------------------------------------ velocity


def test_velocity_is_zero_at_rest(env) -> None:
    """A standing Mario has no velocity in either axis."""
    env.reset(seed=0)
    _advance(env, NOOP, 40)
    velocity = ram_map.read_velocity(env.ram)
    assert velocity.vx_subpixels == 0
    assert velocity.vy_pixels == 0


def test_running_right_reaches_the_speed_cap(env) -> None:
    """vx must be positive, and capped at the value the admissible heuristic assumes."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 60)
    velocity = ram_map.read_velocity(env.ram)
    assert velocity.vx_subpixels > 0
    assert velocity.vx_pixels == pytest.approx(ram_map.MAX_VX_PIXELS_PER_FRAME)


def test_velocity_predicts_displacement(env) -> None:
    """vx and vy must actually describe the motion, not merely correlate with it."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 40)
    errors = []
    previous_x = ram_map.read_x(env.ram)
    previous_y = ram_map.read_y_screen(env.ram)
    for _ in range(50):
        velocity = ram_map.read_velocity(env.ram)
        env._frame_advance(RIGHT_A_B)
        x, y = ram_map.read_x(env.ram), ram_map.read_y_screen(env.ram)
        errors.append((abs((x - previous_x) - velocity.vx_pixels),
                       abs((y - previous_y) - velocity.vy_pixels)))
        previous_x, previous_y = x, y
    mean_x_error = float(np.mean([e[0] for e in errors]))
    mean_y_error = float(np.mean([e[1] for e in errors]))
    # Not exact: the hardware accumulates sub-pixel remainders between frames.
    assert mean_x_error < 1.0
    assert mean_y_error < 1.0


def test_on_ground_flips_during_a_jump(env) -> None:
    """``on_ground`` gates which controls are available, so it must track reality.

    Jumps straight up rather than running, so the whole arc completes well before the
    first Goomba at x ~ 315 -- otherwise Mario dies mid-test and never "lands".
    """
    env.reset(seed=0)
    _advance(env, NOOP, 20)
    assert ram_map.read_on_ground(env.ram)

    _advance(env, JUMP, 12)
    assert not ram_map.read_on_ground(env.ram), "should be airborne mid-jump"

    for _ in range(90):
        env._frame_advance(NOOP)
        if ram_map.read_on_ground(env.ram):
            break
    else:
        pytest.fail("never landed")
    assert not ram_map.read_is_dying(env.ram), "died mid-test; the landing check is vacuous"


# -------------------------------------------------------------------------------- death


def test_read_is_dying_agrees_with_the_package(env) -> None:
    """Our RAM-only death check must match ``info['is_dying']`` on every frame.

    The planner uses this as its edge-invalidity signal, so a disagreement would mean A*
    either discards good edges or keeps fatal ones.
    """
    env.reset(seed=0)
    for _ in range(200):
        env._frame_advance(RIGHT_B)
        assert ram_map.read_is_dying(env.ram) == env._get_info()["is_dying"]


def test_death_on_the_first_goomba_is_detected(env) -> None:
    """Phase 0, Finding 6: the known collision at x ~ 315."""
    env.reset(seed=0)
    for _ in range(200):
        env._frame_advance(RIGHT_B)
        if ram_map.read_is_dying(env.ram):
            break
    else:
        pytest.fail("expected a death on the first Goomba")
    assert ram_map.read_player_state(env.ram) == ram_map.PLAYER_STATE_DYING
    assert 250 < ram_map.read_x(env.ram) < 400


# ------------------------------------------------------------------------------ enemies


def test_first_goomba_is_found_and_approaches(env) -> None:
    """Enemy positions must be level-absolute and must converge on the collision point."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 60)
    enemies = ram_map.read_enemies(env.ram)
    assert enemies, "the first Goomba should be active by now"
    goomba = enemies[0]
    assert goomba.kind == 0x06, "0x06 is a Goomba"

    first_gap = abs(goomba.x - ram_map.read_x(env.ram))
    _advance(env, RIGHT_B, 40)
    later_gap = abs(ram_map.read_enemies(env.ram)[0].x - ram_map.read_x(env.ram))
    assert later_gap < first_gap, "Mario and the Goomba should be closing"


def test_enemy_types_match_the_package(env) -> None:
    """Cross-check against the one enemy field ``info`` does expose."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 60)
    ours = {enemy.kind for enemy in ram_map.read_enemies(env.ram)}
    theirs = {kind for kind in env._get_info()["enemy_types"] if kind}
    assert ours == theirs


# --------------------------------------------------------------------------- tile grid


def test_tile_grid_shape_and_alignment(env) -> None:
    """The grid must be screen-sized and start on a tile boundary at the viewport edge."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 120)
    grid = ram_map.read_tile_grid(env.ram)
    assert grid.tiles.shape == (ram_map.TILE_ROWS, ram_map.TILE_COLS)
    screen_left = ram_map.read_screen_left(env.ram)
    assert grid.level_x(0) <= screen_left < grid.level_x(0) + ram_map.TILE_SIZE


def test_ground_rows_are_solid(env) -> None:
    """1-1 opens with unbroken ground, which must appear as solid bottom rows."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 60)
    solid = ram_map.read_tile_grid(env.ram).solid()
    assert solid[11].all(), "row 11 is the ground surface"
    assert solid[12].all(), "row 12 is below the ground surface"
    assert not solid[:7].any(), "the sky should be empty"


def test_tile_under_a_grounded_mario_is_solid(env) -> None:
    """The decisive alignment test: a shifted grid fails this immediately."""
    env.reset(seed=0)
    for _ in range(150):
        env._frame_advance(RIGHT_B)
        if ram_map.read_is_dying(env.ram):
            break
        if not ram_map.read_on_ground(env.ram):
            continue
        feet_y = ram_map.read_y_screen(env.ram) + ram_map.TILE_SIZE * 2
        tile = ram_map.tile_at(env.ram, ram_map.read_x(env.ram) + 8, feet_y)
        assert ram_map.is_solid(tile), f"nothing solid under a grounded Mario (tile {tile:#04x})"


def test_tile_at_agrees_with_the_grid(env) -> None:
    """The point query and the bulk read must index the ring buffer identically."""
    env.reset(seed=0)
    _advance(env, RIGHT_B, 120)
    grid = ram_map.read_tile_grid(env.ram)
    for row in range(ram_map.TILE_ROWS):
        for col in range(ram_map.TILE_COLS):
            point = ram_map.tile_at(env.ram, grid.level_x(col), grid.screen_y(row))
            assert point == grid.tiles[row, col], f"mismatch at row {row}, col {col}"


def test_tile_at_outside_the_grid_is_empty(env) -> None:
    """Above the tile area (the status bar) and below the level there is nothing."""
    assert ram_map.tile_at(env.ram, 100, 0) == 0
    assert ram_map.tile_at(env.ram, 100, 1000) == 0


def test_the_first_pipe_appears_as_solid_tiles(env) -> None:
    """Ground truth from Phase 1: the pipe at x ~ 594 is what stopped the random agent.

    If the obstacle map is real, solid tiles must stand *above* ground level there.
    """
    env.reset(seed=0)
    while ram_map.read_x(env.ram) < 560:
        env._frame_advance(RIGHT_B)
    grid = ram_map.read_tile_grid(env.ram)
    solid = grid.solid()
    pipe_columns = [
        col for col in range(ram_map.TILE_COLS)
        if 580 <= grid.level_x(col) <= 620 and solid[10, col]
    ]
    assert pipe_columns, "expected solid tiles above ground level at the first pipe"
