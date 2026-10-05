"""A keyboard-driven agent, for a human baseline through the same harness.

**Known limitation, stated up front.** ``cv2.waitKey`` reports key *presses*, not key-down
and key-up events, so there is no way to detect a button being *held* -- which is exactly
what Mario requires. This agent therefore **latches**: a keypress sets the action and it
persists until another key is pressed. Playing well that way is awkward, and a human score
collected this way should be read as a rough floor, not as a fair measure of human skill.

Doing it properly would need a different input backend (pyglet, pygame) and is out of scope
for a one-week project. The agent exists so a human run is *possible* through the same
evaluation path as the automated agents, which is what makes it comparable at all.
"""

from __future__ import annotations

from src.agents.base import AgentDiagnostics, Observation
from src.env.actions import ACTION_NAMES

#: Keyboard mapping. Chosen for one-handed play on the right of the keyboard.
KEY_TO_ACTION = {
    ord("d"): 1,   # right
    ord("w"): 2,   # right + A  (jump right)
    ord("f"): 3,   # right + B  (run right)
    ord("e"): 4,   # right + A + B (run-jump right)
    ord("j"): 5,   # A (jump in place)
    ord("a"): 6,   # left
    ord("x"): 0,   # NOOP -- release
}


class HumanAgent:
    """Replays whatever action the viewer last selected.

    The demo loop feeds keypresses in via :meth:`set_key`; the agent itself is passive.
    """

    name = "human"

    def __init__(self) -> None:
        """Start idle."""
        self._action = 0
        self.diagnostics = AgentDiagnostics()

    def reset(self) -> None:
        """Release the latched action at the start of an episode."""
        self._action = 0
        self.diagnostics.reset()
        self.diagnostics.notes["input"] = "latched keys (cv2 has no key-up events)"

    def set_key(self, key: int) -> bool:
        """Offer a keypress to the agent.

        Args:
            key: a ``cv2.waitKey`` code.

        Returns:
            ``True`` if the key was a movement key and was consumed, so the demo loop knows
            not to also treat it as a UI command.
        """
        if key in KEY_TO_ACTION:
            self._action = KEY_TO_ACTION[key]
            return True
        return False

    @property
    def current_action_name(self) -> str:
        """Readable name of the latched action, for the HUD."""
        return ACTION_NAMES[self._action]

    def act(self, observation: Observation) -> int:
        """Return the latched action."""
        return self._action
