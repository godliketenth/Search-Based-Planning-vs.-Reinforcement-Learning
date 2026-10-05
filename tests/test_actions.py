"""Tests for the action-byte table.

These matter more than their size suggests: the planner works in joypad bytes and the RL
agent works in discrete indices, and a mismatch between the two would make every Phase 5
comparison meaningless while still producing plausible-looking numbers.
"""

from __future__ import annotations

import pytest

from src.env import actions


def test_byte_table_matches_phase0_findings() -> None:
    """The table must equal the one verified in docs/phase0-findings.md."""
    assert actions.ACTION_BYTES == (0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40)
    assert actions.N_ACTIONS == 7


def test_action_names() -> None:
    """Names are used in HUDs and logs, so pin their exact spelling."""
    assert actions.ACTION_NAMES == (
        "NOOP", "right", "right A", "right B", "right A B", "A", "left",
    )
    assert actions.name_for_action(3) == "right B"


def test_named_constants_match_indices() -> None:
    """The constants Phase 3 primitives are written in must match the table."""
    assert actions.NOOP == actions.ACTION_BYTES[0] == 0x00
    assert actions.RIGHT == actions.ACTION_BYTES[1] == 0x80
    assert actions.RIGHT_B == actions.ACTION_BYTES[3] == 0x82
    assert actions.LEFT == actions.ACTION_BYTES[6] == 0x40


def test_byte_action_round_trip() -> None:
    """byte_for_action and action_for_byte must be exact inverses."""
    for action in range(actions.N_ACTIONS):
        assert actions.action_for_byte(actions.byte_for_action(action)) == action


def test_encode_buttons_is_a_bitmask_or() -> None:
    """Button bytes OR together; this is what makes 'right + B' == 0x82."""
    assert actions.encode_buttons(["NOOP"]) == 0x00
    assert actions.encode_buttons(["right"]) == 0x80
    assert actions.encode_buttons(["right", "B"]) == 0x82
    assert actions.encode_buttons(["right", "A", "B"]) == 0x83
    # OR is order-independent.
    assert actions.encode_buttons(["B", "right"]) == actions.encode_buttons(["right", "B"])


@pytest.mark.parametrize("bad_action", [-1, 7, 99])
def test_byte_for_action_rejects_out_of_range(bad_action: int) -> None:
    """An out-of-range index must fail loudly, not silently wrap."""
    with pytest.raises(IndexError):
        actions.byte_for_action(bad_action)


@pytest.mark.parametrize("bad_byte", [0x02, 0x10, 0xFF])
def test_action_for_byte_rejects_unreachable_bytes(bad_byte: int) -> None:
    """Bytes outside SIMPLE_MOVEMENT (e.g. bare B, or 'up') have no action index."""
    with pytest.raises(KeyError):
        actions.action_for_byte(bad_byte)


def test_unknown_button_name_raises() -> None:
    """A typo'd button name must not silently encode to 0x00 (a NOOP)."""
    with pytest.raises(KeyError):
        actions.encode_buttons(["rigth"])
