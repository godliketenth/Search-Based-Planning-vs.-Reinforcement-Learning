"""Two agents side by side, in one OpenCV window.

:mod:`src.viz.comparison` is the engine -- it advances two :class:`~src.viz.live_demo.DemoRunner`
instances in lockstep and hands back frames, and it deliberately contains no UI code so it can
be tested headlessly. The Streamlit dashboard is one shell over it; this module is the other,
for presenting without a browser.

Worth having both: the browser version is nicer to point a room at, but it depends on a web
server, a websocket and a frontend that has already broken once. A single OpenCV window is the
fallback that cannot fail that way, and it is also the only place the comparison can be driven
from the keyboard -- press ``s`` and the perturbation appears on both panels in the same frame.

The composite is laid out as two captioned panels with a divider: a title strip naming the
agent and colour-coded by outcome, the game frame with its overlay, and two lines of the stats
an examiner asks about (position, velocity, realised sticky rate, replans, nodes expanded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.viz.comparison import SIDES, ComparisonSession
from src.viz.live_demo import (
    KEY_FASTER,
    KEY_OVERLAY,
    KEY_PAUSE,
    KEY_QUIT,
    KEY_RESET,
    KEY_SLOWER,
    KEY_STEP,
    KEY_STICKY,
    NATIVE_FPS,
    OVERLAY_MODES,
    SPEED_STEPS,
)

#: Height in pixels of the strip above each panel that names the agent.
TITLE_HEIGHT = 30

#: Height of the strip below each panel carrying two lines of statistics.
STATS_HEIGHT = 44

#: Width of the gap between the two panels.
DIVIDER_WIDTH = 4

#: Outcome colours, RGB. The composite is assembled in RGB and converted once at the window
#: boundary, matching the convention in :mod:`src.viz.live_demo`.
WHITE = (235, 235, 235)
GREY = (170, 170, 170)
GREEN = (90, 220, 120)
RED = (235, 90, 90)


def _outcome(stats: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """Describe how this side's episode stands, as ``(text, colour)``."""
    if stats.get("flag_get"):
        return "REACHED THE FLAG", GREEN
    if stats.get("episode_over"):
        return "EPISODE OVER", RED
    return "running", GREY


def _stat_lines(stats: dict[str, Any], goal_x: int | None) -> tuple[str, str]:
    """Two lines of readout: where the agent is, and what its planner is costing."""
    position = f"x {stats['x']}" + (f"/{goal_x}" if goal_x else "")
    first = (
        f"frame {stats['frame']:>5}  {position}  vx {stats['vx']:+.2f}  "
        f"sticky {stats['sticky_p']:.2f} (realised {stats['sticky_realised']:.2f})"
    )
    if "replans" in stats:
        # Only planning agents report these; a learned policy omits them rather than
        # reporting zeros, so "no planning" stays distinguishable from "planned nothing".
        second = (
            f"replans {stats['replans']}  nodes {stats['nodes_expanded']}  "
            f"search {stats['planning_seconds']:.1f}s  action {stats['action']}"
        )
    else:
        second = f"no planning -- reactive policy  action {stats['action']}"
    return first, second


def _panel(frame: np.ndarray, label: str, stats: dict[str, Any], goal_x: int | None) -> np.ndarray:
    """Caption one agent's frame above and below."""
    import cv2

    height, width = frame.shape[:2]
    canvas = np.zeros((TITLE_HEIGHT + height + STATS_HEIGHT, width, 3), dtype=np.uint8)
    canvas[TITLE_HEIGHT:TITLE_HEIGHT + height] = frame

    outcome, colour = _outcome(stats)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label, (8, 21), font, 0.52, WHITE, 1, cv2.LINE_AA)
    (text_width, _), _ = cv2.getTextSize(outcome, font, 0.44, 1)
    cv2.putText(canvas, outcome, (width - text_width - 8, 21), font, 0.44, colour, 1, cv2.LINE_AA)

    for index, line in enumerate(_stat_lines(stats, goal_x)):
        baseline = TITLE_HEIGHT + height + 17 + index * 18
        cv2.putText(canvas, line, (8, baseline), font, 0.38, GREY, 1, cv2.LINE_AA)
    return canvas


def compose(
    payloads: dict[str, dict[str, Any]],
    labels: dict[str, str],
    goal_x: int | None = None,
) -> np.ndarray:
    """Build the full side-by-side image from one :meth:`ComparisonSession.tick` result.

    Args:
        payloads: ``{"left": {"frame", "stats", "agent"}, "right": {...}}``.
        labels: readable name per side, for the title strips.
        goal_x: the level's goal position, shown alongside each agent's x.

    Returns:
        An RGB image: left panel, divider, right panel.
    """
    panels = [
        _panel(payloads[side]["frame"], labels[side], payloads[side]["stats"], goal_x)
        for side in SIDES
    ]
    height = max(panel.shape[0] for panel in panels)
    divider = np.zeros((height, DIVIDER_WIDTH, 3), dtype=np.uint8)
    divider[:] = (60, 60, 60)
    return np.hstack([panels[0], divider, panels[1]])


class ComparisonWindow:
    """An interactive OpenCV window over a :class:`ComparisonSession`.

    Mirrors :class:`~src.viz.live_demo.DemoRunner`'s controls, applied to both sides at once:

    ===========  =====================================================================
    ``space``    pause / resume
    ``n``        advance one tick while paused
    ``[`` ``]``  slower / faster
    ``s``        toggle sticky actions on both panels -- the headline result, on demand
    ``o``        cycle overlay layers
    ``r``        restart both episodes from the same seed
    ``q``        quit
    ===========  =====================================================================
    """

    def __init__(
        self,
        session: ComparisonSession,
        labels: dict[str, str],
        *,
        sticky_p: float = 0.25,
        goal_x: int | None = None,
        record_path: Path | None = None,
        window_name: str = "planning vs learning",
    ) -> None:
        """Wrap a built session.

        Args:
            session: the engine. The window does not own it and will not close it.
            labels: readable agent name per side.
            sticky_p: probability applied when the ``s`` toggle is on. Falls back to the
                design's 0.25 if the session was opened unperturbed, so the key always does
                something visible.
            goal_x: the level's goal position, for the readout.
            record_path: mp4 of the composite, always written unless ``None`` -- the backup
                for presenting if a window cannot be opened on the day.
            window_name: OpenCV window title.
        """
        self.session = session
        self.labels = labels
        self.sticky_p = sticky_p or 0.25
        self.goal_x = goal_x
        self.record_path = record_path
        self.window_name = window_name

        self.running = True
        self.paused = False
        self.step_once = False
        self.sticky_on = False
        self.speed_index = SPEED_STEPS.index(1.0)
        self.overlay_index = 0
        self._writer: Any = None

    # -------------------------------------------------------------------------- controls

    @property
    def speed(self) -> float:
        """Current playback multiplier."""
        return SPEED_STEPS[self.speed_index]

    def delay_ms(self) -> int:
        """Milliseconds to wait between ticks, so playback tracks the NES clock."""
        per_tick = self.session.frames_per_tick / (NATIVE_FPS * self.speed)
        return max(1, int(per_tick * 1000))

    def toggle_sticky(self) -> float:
        """Switch the perturbation on or off on both panels. Returns the probability now set."""
        self.sticky_on = not self.sticky_on
        probability = self.sticky_p if self.sticky_on else 0.0
        self.session.set_sticky(probability)
        return probability

    def handle_key(self, key: int) -> None:
        """Apply one ``cv2.waitKey`` code. ``-1`` and ``255`` mean no key."""
        if key in (-1, 255):
            return
        if key == KEY_QUIT:
            self.running = False
        elif key == KEY_PAUSE:
            self.paused = not self.paused
        elif key == KEY_STEP:
            self.step_once = True
        elif key == KEY_SLOWER:
            self.speed_index = max(0, self.speed_index - 1)
        elif key == KEY_FASTER:
            self.speed_index = min(len(SPEED_STEPS) - 1, self.speed_index + 1)
        elif key == KEY_STICKY:
            self.toggle_sticky()
        elif key == KEY_OVERLAY:
            self.overlay_index = (self.overlay_index + 1) % len(OVERLAY_MODES)
            self.session.set_overlay(self.overlay_index)
        elif key == KEY_RESET:
            self.session.reset()

    def should_advance(self) -> bool:
        """Whether the next iteration should emulate a tick."""
        if self.session.finished:
            return False
        if not self.paused:
            return True
        if self.step_once:
            self.step_once = False
            return True
        return False

    # ---------------------------------------------------------------------------- output

    def _open_writer(self) -> None:
        """Open the mp4 writer, if recording was requested."""
        if self.record_path is None:
            return
        import imageio.v2 as imageio

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        # One mp4 frame per tick, so the recording plays at the same rate as the window.
        fps = NATIVE_FPS / self.session.frames_per_tick
        self._writer = imageio.get_writer(self.record_path, fps=fps, macro_block_size=1)

    def _record(self, image: np.ndarray) -> None:
        """Append one composite to the mp4."""
        if self._writer is not None:
            self._writer.append_data(np.ascontiguousarray(image))

    def _close_writer(self) -> None:
        """Finish the mp4."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------------- main loops

    def run(self) -> int:
        """Show the window until ``q`` or both episodes end plus a pause for the final frame.

        Falls back to :meth:`run_headless` if no window can be created, so the mp4 is produced
        either way.

        Returns:
            The number of ticks advanced.
        """
        try:
            import cv2

            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as error:  # pragma: no cover - depends on the display stack
            print(f"could not open a window ({error}); falling back to headless recording")
            return self.run_headless()

        self._open_writer()
        ticks = 0
        try:
            image = self._tick_image()
            while self.running:
                if self.should_advance():
                    image = self._tick_image()
                    ticks += 1
                cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.handle_key(cv2.waitKey(self.delay_ms()) & 0xFF)
        finally:
            self._close_writer()
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:  # pragma: no cover - window already gone
                pass
        return ticks

    def run_headless(self, ticks: int | None = None) -> int:
        """Advance with no window, writing the mp4. Returns the number of ticks advanced."""
        self._open_writer()
        advanced = 0
        try:
            while not self.session.finished and (ticks is None or advanced < ticks):
                self._tick_image()
                advanced += 1
        finally:
            self._close_writer()
        return advanced

    def _tick_image(self) -> np.ndarray:
        """Advance both sides one tick and return the recorded composite."""
        image = compose(self.session.tick(), self.labels, self.goal_x)
        self._record(image)
        return image
