"""Tests for the configuration-space representation.

The lattice tests matter for Phase 3: the lattice is what makes A* terminate at all, and
its resolution is exactly what the planner's completeness claim is qualified by.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.env import ram_map
from src.env.actions import NOOP, RIGHT_B
from src.env.factories import make_planner_env
from src.env.state import DEFAULT_VX_RES, DEFAULT_X_RES, MarioState, WorldState


@pytest.fixture(scope="module")
def env():
    """A reset raw emulator shared across tests."""
    environment = make_planner_env("1-1")
    environment.reset(seed=0)
    yield environment
    environment.close()


def _state(x=100, y=432, vx=0, vy=0, on_ground=True, powerup=0) -> MarioState:
    return MarioState(x=x, y=y, vx=vx, vy=vy, on_ground=on_ground, powerup=powerup)


# ------------------------------------------------------------------------- construction


def test_from_env_matches_the_underlying_readers(env) -> None:
    """The dataclass must not quietly transform anything it reads."""
    env.reset(seed=0)
    for _ in range(50):
        env._frame_advance(RIGHT_B)
    ram = env.ram
    state = MarioState.from_env(env)
    velocity = ram_map.read_velocity(ram)
    assert state.x == ram_map.read_x(ram)
    assert state.y == ram_map.read_y_level(ram)
    assert state.vx == velocity.vx_subpixels
    assert state.vy == velocity.vy_pixels
    assert state.on_ground == ram_map.read_on_ground(ram)
    assert state.powerup == ram_map.read_powerup(ram)


def test_state_is_frozen_and_hashable() -> None:
    """States are dictionary keys in A*'s closed set."""
    state = _state()
    assert hash(state) == hash(_state())
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.x = 5  # type: ignore[misc]


def test_vx_pixels_converts_from_subpixels() -> None:
    """The two velocity axes use different units; the conversion must be explicit."""
    assert _state(vx=48).vx_pixels == pytest.approx(3.0)
    assert _state(vx=-16).vx_pixels == pytest.approx(-1.0)


# ------------------------------------------------------------------------------ lattice


def test_nearby_states_share_a_lattice_key() -> None:
    """Without this, A* expands practically infinitely many near-identical nodes.

    Uses bucket-aligned bases: buckets are ``x // res``, so two states merge only when they
    fall inside the *same* bucket, not merely within ``res`` of each other.
    """
    base = 96  # a multiple of DEFAULT_X_RES
    assert _state(x=base).lattice_key() == _state(x=base + DEFAULT_X_RES - 1).lattice_key()
    assert _state(vx=0).lattice_key() == _state(vx=DEFAULT_VX_RES - 1).lattice_key()


def test_distant_states_have_different_lattice_keys() -> None:
    """Bucketing must not merge genuinely different configurations."""
    assert _state(x=100).lattice_key() != _state(x=100 + DEFAULT_X_RES).lattice_key()
    assert _state(y=400).lattice_key() != _state(y=500).lattice_key()


def test_velocity_is_part_of_the_key() -> None:
    """The point of a kinodynamic C-space: same position, different reachable set."""
    assert _state(vx=0).lattice_key() != _state(vx=48).lattice_key()


def test_on_ground_and_powerup_are_part_of_the_key() -> None:
    """Both change which controls are available or how big the collision box is."""
    assert _state(on_ground=True).lattice_key() != _state(on_ground=False).lattice_key()
    assert _state(powerup=0).lattice_key() != _state(powerup=1).lattice_key()


def test_vy_is_deliberately_excluded_from_the_key() -> None:
    """vy is near-determined by y and on_ground under constant gravity."""
    assert _state(vy=0).lattice_key() == _state(vy=4).lattice_key()


def test_coarser_resolution_merges_more_states() -> None:
    """Resolution is the knob that trades completeness for speed."""
    a, b = _state(x=128), _state(x=160)
    assert a.lattice_key(x_res=8) != b.lattice_key(x_res=8)
    assert a.lattice_key(x_res=64) == b.lattice_key(x_res=64)


# --------------------------------------------------------------------------- world state


def test_world_state_from_env_is_populated(env) -> None:
    """The overlay and the planner both read this; every field must be real."""
    env.reset(seed=0)
    for _ in range(60):
        env._frame_advance(RIGHT_B)
    info = env._get_info()
    world = WorldState.from_env(env, info["flag_get"], info["time"])

    assert world.mario.x == info["x_pos"]
    assert world.tiles.tiles.shape == (ram_map.TILE_ROWS, ram_map.TILE_COLS)
    assert world.is_dying == info["is_dying"]
    assert world.time == info["time"]
    assert world.screen_left <= world.mario.x <= world.screen_left + 256
    assert world.mario_y_screen == ram_map.read_y_screen(env.ram)
    assert all(isinstance(enemy, ram_map.Enemy) for enemy in world.enemies)


def test_world_state_sees_the_goomba(env) -> None:
    """Dynamic obstacles must reach the state object, not just the RAM reader."""
    env.reset(seed=0)
    for _ in range(60):
        env._frame_advance(RIGHT_B)
    world = WorldState.from_env(env)
    assert world.enemies, "the first Goomba should be active"
    assert any(enemy.kind == 0x06 for enemy in world.enemies)
