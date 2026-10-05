"""Tests for motion primitives.

A primitive is an edge of the search graph, so a malformed one does not crash -- it
produces a plan that cannot be executed.
"""

from __future__ import annotations

import pytest

from src.env.actions import LEFT, NOOP, RIGHT, RIGHT_A_B, RIGHT_B
from src.planning.primitives import (
    DEFAULT_PRIMITIVES,
    MotionPrimitive,
    hold,
    jump,
    primitives_from_config,
)


def test_hold_repeats_one_button() -> None:
    """The simplest primitive body."""
    assert hold(RIGHT_B, 4) == (RIGHT_B,) * 4
    assert hold(NOOP, 0) == ()


def test_jump_holds_a_then_keeps_moving() -> None:
    """Jump height is set by how long A is held; the rest preserves momentum."""
    body = jump(hold_frames=3, total_frames=8)
    assert len(body) == 8
    assert body[:3] == (RIGHT_A_B,) * 3
    assert body[3:] == (RIGHT_B,) * 5


def test_jump_rejects_a_hold_longer_than_its_duration() -> None:
    """Catches a config typo that would otherwise silently shorten the primitive."""
    with pytest.raises(ValueError):
        jump(hold_frames=10, total_frames=4)


def test_frames_equals_byte_count() -> None:
    """Edge cost is duration, so these must not drift apart."""
    for primitive in DEFAULT_PRIMITIVES:
        assert primitive.frames == len(primitive.bytes_) == len(primitive)


def test_default_set_is_non_empty_and_named() -> None:
    """Names appear in plans and in the demo HUD."""
    assert len(DEFAULT_PRIMITIVES) >= 5
    names = [primitive.name for primitive in DEFAULT_PRIMITIVES]
    assert len(set(names)) == len(names), "primitive names must be unique"
    for primitive in DEFAULT_PRIMITIVES:
        assert primitive.frames > 0


def test_default_set_spans_several_jump_heights() -> None:
    """The point of the set: different A-hold durations are different manoeuvres."""
    hold_lengths = {
        sum(1 for byte in primitive.bytes_ if byte == RIGHT_A_B)
        for primitive in DEFAULT_PRIMITIVES
    }
    assert len(hold_lengths - {0}) >= 3, "need several distinct jump heights"


def test_primitives_are_hashable_and_frozen() -> None:
    """They are stored in tuples and compared during plan reconstruction."""
    primitive = MotionPrimitive("x", (RIGHT_B, RIGHT_B))
    assert hash(primitive) == hash(MotionPrimitive("x", (RIGHT_B, RIGHT_B)))


def test_config_none_gives_the_default_set() -> None:
    """An omitted `primitives:` key must not silently produce an empty edge set."""
    assert primitives_from_config(None) == DEFAULT_PRIMITIVES
    assert primitives_from_config([]) == DEFAULT_PRIMITIVES


def test_config_builds_held_and_jump_primitives() -> None:
    """Both YAML forms must round-trip into the right bytes."""
    built = primitives_from_config(
        [
            {"name": "dash", "byte": "right_b", "frames": 3},
            {"name": "leap", "jump": {"hold": 2, "total": 5}},
            {"name": "retreat", "byte": "left", "frames": 2},
            {"name": "idle", "byte": "noop", "frames": 1},
            {"name": "stroll", "byte": "right", "frames": 2},
        ]
    )
    assert built[0].bytes_ == (RIGHT_B,) * 3
    assert built[1].bytes_ == (RIGHT_A_B, RIGHT_A_B, RIGHT_B, RIGHT_B, RIGHT_B)
    assert built[2].bytes_ == (LEFT, LEFT)
    assert built[3].bytes_ == (NOOP,)
    assert built[4].bytes_ == (RIGHT, RIGHT)


def test_config_rejects_unknown_buttons_and_malformed_entries() -> None:
    """A typo in the config must fail loudly, not produce a NOOP primitive."""
    with pytest.raises(ValueError, match="unknown button"):
        primitives_from_config([{"name": "bad", "byte": "rihgt", "frames": 2}])
    with pytest.raises(ValueError, match="needs either"):
        primitives_from_config([{"name": "bad"}])
