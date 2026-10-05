"""NES joypad button bytes for the ``SIMPLE_MOVEMENT`` action set.

Two agents drive the same emulator through two different interfaces, and this module is the
single place where those interfaces are reconciled:

* The **RL agent** (Phase 4) sees a Gymnasium ``Discrete(7)`` action space, courtesy of
  :class:`nes_py.wrappers.JoypadSpace`, and never touches a button byte.
* The **planner** (Phase 3) bypasses the wrapper stack entirely and calls
  ``env.unwrapped._frame_advance(byte)``, so it needs the raw byte encoding. Motion
  primitives are defined as sequences of these bytes.

Keeping one table means a plan produced by the planner and an action index logged by the RL
agent refer to provably the same button press, which is what makes the Phase 5 comparison
like-for-like at the action level.

The byte encoding is a bitmask over the NES controller::

    right 0x80   left 0x40   down 0x20   up 0x10
    start 0x08   select 0x04   B 0x02      A 0x01

so ``right + B`` (run right) is ``0x80 | 0x02 = 0x82``.
"""

from __future__ import annotations

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

#: Button name -> bitmask, taken from ``JoypadSpace`` so this module cannot drift from the
#: encoding the RL wrapper actually applies.
BUTTON_BITS: dict[str, int] = dict(JoypadSpace._button_map)

#: The table recorded in ``docs/phase0-findings.md``. Asserted below rather than used, so a
#: package upgrade that reorders ``SIMPLE_MOVEMENT`` fails loudly at import instead of
#: silently invalidating every cached plan and every logged action.
_PHASE0_BYTES: tuple[int, ...] = (0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40)


def encode_buttons(buttons: list[str] | tuple[str, ...]) -> int:
    """OR together the bitmasks of a list of button names.

    Args:
        buttons: button names, e.g. ``["right", "B"]``. ``"NOOP"`` contributes nothing.

    Returns:
        The joypad byte to hand to ``_frame_advance``.

    Raises:
        KeyError: if a button name is not a real NES button.
    """
    byte = 0
    for button in buttons:
        byte |= BUTTON_BITS[button]
    return byte


#: Discrete action index -> joypad byte. Index order is ``SIMPLE_MOVEMENT``'s.
ACTION_BYTES: tuple[int, ...] = tuple(encode_buttons(b) for b in SIMPLE_MOVEMENT)

#: Discrete action index -> human-readable name, e.g. ``"right A B"``.
ACTION_NAMES: tuple[str, ...] = tuple(" ".join(b) for b in SIMPLE_MOVEMENT)

#: Number of actions in the action set.
N_ACTIONS: int = len(ACTION_BYTES)

if ACTION_BYTES != _PHASE0_BYTES:  # pragma: no cover - guards against a dependency change
    raise RuntimeError(
        "SIMPLE_MOVEMENT no longer matches the byte table verified in Phase 0.\n"
        f"  expected: {[hex(b) for b in _PHASE0_BYTES]}\n"
        f"  actual:   {[hex(b) for b in ACTION_BYTES]}\n"
        "Cached A* plans and logged RL actions are action *indices*, so a reordering here "
        "silently corrupts them. Re-verify docs/phase0-findings.md before proceeding."
    )

# Named bytes, for readable motion primitives in Phase 3.
NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A, LEFT = ACTION_BYTES

_BYTE_TO_ACTION: dict[int, int] = {byte: idx for idx, byte in enumerate(ACTION_BYTES)}


def byte_for_action(action: int) -> int:
    """Translate a discrete action index into a joypad byte.

    Args:
        action: index into the ``SIMPLE_MOVEMENT`` action set.

    Returns:
        The corresponding joypad byte.

    Raises:
        IndexError: if ``action`` is outside the action set.
    """
    if not 0 <= action < N_ACTIONS:
        raise IndexError(f"action {action} outside SIMPLE_MOVEMENT (0..{N_ACTIONS - 1})")
    return ACTION_BYTES[action]


def action_for_byte(byte: int) -> int:
    """Translate a joypad byte back into a discrete action index.

    The inverse of :func:`byte_for_action`. Used when a plan expressed in bytes has to be
    logged or replayed through the wrapped RL-style env (Phase 5b caches plans as action
    sequences, because emulator snapshots are not picklable).

    Args:
        byte: a joypad byte.

    Returns:
        The discrete action index.

    Raises:
        KeyError: if the byte is not reachable in this action set.
    """
    try:
        return _BYTE_TO_ACTION[byte]
    except KeyError:
        raise KeyError(
            f"byte {byte:#04x} is not in SIMPLE_MOVEMENT; "
            f"valid bytes are {[hex(b) for b in ACTION_BYTES]}"
        ) from None


def name_for_action(action: int) -> str:
    """Return the human-readable button list for an action index (for HUDs and logs)."""
    if not 0 <= action < N_ACTIONS:
        raise IndexError(f"action {action} outside SIMPLE_MOVEMENT (0..{N_ACTIONS - 1})")
    return ACTION_NAMES[action]
