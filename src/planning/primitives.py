"""Motion primitives: the edges of the search graph.

Why primitives instead of grid moves
------------------------------------
A grid search would treat "move one cell right" and "move one cell up" as edges, which
assumes Mario can move between adjacent cells at will. He cannot: gravity acts every frame,
momentum carries him, and once airborne his control authority is sharply limited. A path
through a 4- or 8-connected grid is therefore not something Mario can execute -- it is
**dynamically infeasible**.

A motion primitive is instead a short, fixed sequence of button presses. Its effect is
whatever the physics produces, so every edge is **feasible by construction**. The planner
never has to check "can Mario do this?", only "where does this leave him, and did it kill
him?" -- and both questions are answered by running the actual game.

Jump height is set by *how long A is held*, which is why the set below is mostly jumps of
different hold durations. That is the control input of this system; holding A for 4 frames
and for 28 frames are genuinely different manoeuvres with different reachable sets.

Phase 1 gave the concrete motivation: a random policy, which samples a fresh button every
frame, could never clear the first pipe, because clearing it requires *committing* to right
+ A for the whole arc. A primitive is exactly that commitment, made atomic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.env.actions import LEFT, NOOP, RIGHT, RIGHT_A_B, RIGHT_B


@dataclass(frozen=True, slots=True)
class MotionPrimitive:
    """One edge of the search graph: a named, fixed sequence of joypad bytes.

    Attributes:
        name: human-readable label, used in plans, logs and the demo HUD.
        bytes_: the joypad byte to apply on each frame, in order.
    """

    name: str
    bytes_: tuple[int, ...]

    def __len__(self) -> int:
        """Duration of the primitive in frames, which is also its edge cost."""
        return len(self.bytes_)

    @property
    def frames(self) -> int:
        """Duration in frames. ``g`` is measured in frames, so this is the edge cost."""
        return len(self.bytes_)


def hold(byte: int, frames: int) -> tuple[int, ...]:
    """Build a primitive body that holds one button combination for ``frames`` frames."""
    return (byte,) * frames


def jump(hold_frames: int, total_frames: int, moving_byte: int = RIGHT_B) -> tuple[int, ...]:
    """Build a rightward jump that holds A for part of its duration.

    Args:
        hold_frames: frames with A held. Longer means higher -- this is the control that
            sets jump height, and the reason several jump primitives exist.
        total_frames: total duration. The remainder is spent running right without A, so
            Mario keeps his horizontal momentum through the arc.
        moving_byte: what to hold once A is released.

    Returns:
        The byte sequence.
    """
    if hold_frames > total_frames:
        raise ValueError("hold_frames cannot exceed total_frames")
    return hold(RIGHT_A_B, hold_frames) + hold(moving_byte, total_frames - hold_frames)


#: The default primitive set.
#:
#: Deliberately small. Branching factor multiplies through the search depth, and crossing
#: 1-1 needs roughly 90 primitives, so each extra edge is expensive. These seven cover:
#: accelerate/cruise, four jump heights, and a way to stall or retreat.
DEFAULT_PRIMITIVES: tuple[MotionPrimitive, ...] = (
    MotionPrimitive("run", hold(RIGHT_B, 12)),
    MotionPrimitive("walk", hold(RIGHT, 12)),
    MotionPrimitive("hop", jump(hold_frames=4, total_frames=20)),
    MotionPrimitive("jump", jump(hold_frames=12, total_frames=28)),
    MotionPrimitive("long_jump", jump(hold_frames=24, total_frames=40)),
    MotionPrimitive("wait", hold(NOOP, 8)),
    MotionPrimitive("back_off", hold(LEFT, 8)),
)


def primitives_from_config(spec: Sequence[dict] | None) -> tuple[MotionPrimitive, ...]:
    """Build a primitive set from YAML.

    Each entry is either ``{name, byte, frames}`` for a held button, or
    ``{name, jump: {hold, total}}`` for a jump.

    Args:
        spec: the ``primitives`` list from a config, or ``None`` for
            :data:`DEFAULT_PRIMITIVES`.

    Returns:
        The primitive set.

    Raises:
        ValueError: if an entry is malformed.
    """
    if not spec:
        return DEFAULT_PRIMITIVES

    byte_names = {
        "noop": NOOP, "right": RIGHT, "right_b": RIGHT_B,
        "right_a_b": RIGHT_A_B, "left": LEFT,
    }
    built: list[MotionPrimitive] = []
    for entry in spec:
        name = entry["name"]
        if "jump" in entry:
            built.append(
                MotionPrimitive(name, jump(entry["jump"]["hold"], entry["jump"]["total"]))
            )
        elif "byte" in entry and "frames" in entry:
            key = str(entry["byte"]).lower()
            if key not in byte_names:
                raise ValueError(f"unknown button {entry['byte']!r} in primitive {name!r}")
            built.append(MotionPrimitive(name, hold(byte_names[key], entry["frames"])))
        else:
            raise ValueError(f"primitive {name!r} needs either 'jump' or 'byte'+'frames'")
    return tuple(built)
