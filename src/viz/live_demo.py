"""The live demo: planning made visible.

A planner's output is normally an invisible internal detail -- a list of numbers that
produces motion. This module draws the plan *on the game itself*: a dotted trajectory ahead
of Mario, flashing whenever the planner recomputes it. An examiner can then see the
difference between an agent that is following a plan and one that is reacting.

The single most useful control is the **live sticky-action toggle** (`s`). Pressing it mid-run
reproduces the project's headline result on demand: an open-loop plan desynchronises and
Mario dies, while the closed-loop planner visibly re-draws its path and recovers. That is the
difference between 0.15 and 0.80 completion, shown rather than tabulated.

Design notes
------------
* All drawing goes through Phase 2's :mod:`src.viz.debug_overlay`, which was built with this
  demo in mind -- it already accepts a planned path and a replan flash.
* ``env.render()`` returns a *reused view* of the emulator's screen buffer, so frames are
  copied before anything is drawn on them. Forgetting this silently collapsed the Phase 1 GIF
  to a single image.
* The mp4 is always recorded, and ``headless=True`` produces it with no window at all, so the
  deliverable never depends on a GUI working on the day.
* Frames are RGB throughout; conversion to BGR happens only at the OpenCV window boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.env.actions import name_for_action
from src.env.factories import make_rl_env
from src.env.state import WorldState
from src.env.sticky import StickyActionWrapper
from src.eval.harness import _PixelStack
from src.viz.debug_overlay import render_overlay

#: NES frame rate; the demo's nominal playback speed.
NATIVE_FPS = 60

#: Playback speed multipliers reachable with the `[` and `]` keys.
SPEED_STEPS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

#: Key codes returned by ``cv2.waitKey``.
KEY_QUIT = ord("q")
KEY_PAUSE = ord(" ")
KEY_STEP = ord("n")
KEY_SLOWER = ord("[")
KEY_FASTER = ord("]")
KEY_STICKY = ord("s")
KEY_OVERLAY = ord("o")
KEY_RESET = ord("r")

#: Overlay layer presets cycled by the `o` key: (tiles, enemies, path).
OVERLAY_MODES: tuple[tuple[bool, bool, bool], ...] = (
    (True, True, True),     # everything
    (False, True, True),    # plan and enemies, no obstacle map
    (False, False, True),   # plan only -- cleanest for a slide or a photo
    (False, False, False),  # raw game
)


@dataclass
class DemoState:
    """Mutable interface state, separated from the loop so it can be tested headlessly.

    Attributes:
        paused: whether playback is halted.
        step_once: consume one frame while paused, then halt again.
        speed_index: index into :data:`SPEED_STEPS`.
        overlay_index: index into :data:`OVERLAY_MODES`.
        sticky_on: whether the perturbation is currently applied.
        running: cleared by the quit key.
        frame: frames elapsed in the current episode.
        episode_over: set when the env terminates, so the loop can hold the final frame.
    """

    paused: bool = False
    step_once: bool = False
    speed_index: int = SPEED_STEPS.index(1.0)
    overlay_index: int = 0
    sticky_on: bool = False
    running: bool = True
    frame: int = 0
    episode_over: bool = False

    @property
    def speed(self) -> float:
        """Current playback speed multiplier."""
        return SPEED_STEPS[self.speed_index]

    @property
    def overlay_mode(self) -> tuple[bool, bool, bool]:
        """Current ``(tiles, enemies, path)`` layer visibility."""
        return OVERLAY_MODES[self.overlay_index]

    def frame_delay_ms(self) -> int:
        """Milliseconds to wait between frames at the current speed, at least 1."""
        return max(1, int(1000.0 / (NATIVE_FPS * self.speed)))


class DemoRunner:
    """Drives one agent in one environment, drawing the plan and recording an mp4."""

    def __init__(
        self,
        agent: Any,
        level: str = "1-1",
        sticky_p: float = 0.0,
        seed: int = 0,
        scale: int = 3,
        decision_every: int = 1,
        lookahead: int = 90,
        max_frames: int = 12_000,
        record_path: str | Path | None = None,
        headless: bool = False,
        window_name: str = "A* vs PPO -- Super Mario Bros",
    ) -> None:
        """Build the demo.

        Args:
            agent: anything satisfying the :class:`~src.agents.base.Agent` protocol. If it
                exposes ``planned_path()``, that path is drawn.
            level: ``"world-stage"``.
            sticky_p: the probability used **when the toggle is on**. The wrapper is always
                attached so the toggle can work, but starts at this value only if non-zero.
            seed: reset seed.
            scale: integer upscaling of the 256x240 frame for display.
            decision_every: frames between agent decisions (4 for PPO, 1 for the planner).
            lookahead: frames of planned trajectory to draw ahead of Mario.
            max_frames: hard cap per episode.
            record_path: mp4 destination. ``None`` disables recording.
            headless: run with no window. The mp4 is still written.
            window_name: OpenCV window title.
        """
        self.agent = agent
        self.level = level
        self.seed = seed
        self.scale = scale
        self.decision_every = decision_every
        self.lookahead = lookahead
        self.max_frames = max_frames
        self.headless = headless
        self.window_name = window_name
        self.record_path = Path(record_path) if record_path else None

        # The sticky wrapper is attached unconditionally so `s` can toggle it at runtime;
        # `make_rl_env` would omit it entirely at p = 0, leaving nothing to switch on.
        base = make_rl_env(level=level, sticky_p=0.0, render_mode="rgb_array", seed=seed)
        self.sticky_p = sticky_p
        self.env = StickyActionWrapper(base, p=sticky_p, seed=seed)

        self.state = DemoState(sticky_on=sticky_p > 0.0)
        self.env.sampler.p = sticky_p if self.state.sticky_on else 0.0

        self._stack = _PixelStack(self.env)
        self._pixels: np.ndarray | None = None
        self._info: dict = {}
        self._action = 0
        self._writer = None

    # ------------------------------------------------------------------ episode lifecycle

    def reset(self) -> None:
        """Start a fresh episode."""
        _, self._info = self.env.reset(seed=self.seed)
        self._pixels = self._stack.reset()
        self.agent.reset()
        self.state.frame = 0
        self.state.episode_over = False
        self._action = 0

    def step(self) -> np.ndarray:
        """Advance one emulator frame and return the rendered overlay frame (RGB).

        Returns:
            The frame to display, already upscaled.
        """
        from src.agents.base import Observation

        if self.state.frame % self.decision_every == 0 and not self.state.episode_over:
            self._action = self.agent.act(
                Observation(pixels=self._pixels, env=self.env,
                            frame=self.state.frame, info=self._info)
            )

        if not self.state.episode_over:
            _, _, terminated, truncated, self._info = self.env.step(self._action)
            self.state.frame += 1
            if self.state.frame % self.decision_every == 0:
                self._pixels = self._stack.push()
            if terminated or truncated or self.state.frame >= self.max_frames:
                self.state.episode_over = True

        return self.render()

    def render(self) -> np.ndarray:
        """Draw the current state, plan and HUD onto a copy of the emulator frame."""
        show_tiles, show_enemies, show_path = self.state.overlay_mode
        # `.unwrapped`: WorldState reads RAM directly, and Gymnasium 1.x wrappers do not
        # forward arbitrary attributes.
        world = WorldState.from_env(
            self.env.unwrapped, self._info.get("flag_get", False), self._info.get("time", 0)
        )
        path = self._planned_path() if show_path else None
        frame = self.env.render()
        return render_overlay(
            frame, world,
            path=path,
            hud=self._hud(),
            replan_flash=getattr(self.agent, "replanned_this_frame", False),
            show_tiles=show_tiles,
            show_enemies=show_enemies,
            scale=self.scale,
        )

    def _planned_path(self) -> list[tuple[int, int]] | None:
        """The agent's planned trajectory, if it has one to show."""
        planner = getattr(self.agent, "planned_path", None)
        return planner(self.lookahead) if callable(planner) else None

    def _hud(self) -> dict[str, Any]:
        """Extra HUD rows: who is playing, what it is doing, and what it cost."""
        diagnostics = getattr(self.agent, "diagnostics", None)
        rows: dict[str, Any] = {
            "agent": self.agent.name,
            "action": name_for_action(self._action),
            "sticky": f"{self.env.sampler.p:.2f}" if self.state.sticky_on else "off",
            "speed": f"{self.state.speed:g}x" + (" PAUSED" if self.state.paused else ""),
        }
        if diagnostics is not None and diagnostics.planning_calls:
            rows["replans"] = diagnostics.planning_calls
            rows["nodes"] = diagnostics.nodes_expanded
        if self.state.episode_over:
            rows["EPISODE"] = "OVER - r resets"
        return rows

    # ------------------------------------------------------------------------- interaction

    def handle_key(self, key: int) -> None:
        """Apply one keypress to the interface state.

        Args:
            key: a code from ``cv2.waitKey``. ``-1`` (no key) is ignored.
        """
        if key in (-1, 255):
            return
        # A human agent claims its movement keys first, so they are not also read as UI
        # commands (`a` would otherwise be ambiguous).
        setter = getattr(self.agent, "set_key", None)
        if callable(setter) and setter(key):
            return
        if key == KEY_QUIT:
            self.state.running = False
        elif key == KEY_PAUSE:
            self.state.paused = not self.state.paused
        elif key == KEY_STEP:
            self.state.step_once = True
        elif key == KEY_SLOWER:
            self.state.speed_index = max(0, self.state.speed_index - 1)
        elif key == KEY_FASTER:
            self.state.speed_index = min(len(SPEED_STEPS) - 1, self.state.speed_index + 1)
        elif key == KEY_STICKY:
            self.toggle_sticky()
        elif key == KEY_OVERLAY:
            self.state.overlay_index = (self.state.overlay_index + 1) % len(OVERLAY_MODES)
        elif key == KEY_RESET:
            self.reset()

    def toggle_sticky(self) -> float:
        """Switch the perturbation on or off mid-run.

        The demo's most important control: it turns the Phase 5 result into something that
        happens in front of the viewer. An open-loop plan starts failing within a second or
        two; the closed-loop planner redraws its path and keeps going.

        Returns:
            The sticky probability now in effect.
        """
        self.state.sticky_on = not self.state.sticky_on
        # Fall back to the design's p = 0.25 if the demo was launched unperturbed, so the
        # toggle always does something visible.
        self.env.sampler.p = (self.sticky_p or 0.25) if self.state.sticky_on else 0.0
        return self.env.sampler.p

    def should_advance(self) -> bool:
        """Whether the next loop iteration should emulate a frame."""
        if self.state.episode_over:
            return False
        if not self.state.paused:
            return True
        if self.state.step_once:
            self.state.step_once = False
            return True
        return False

    # ----------------------------------------------------------------------------- output

    def _open_writer(self) -> None:
        """Open the mp4 writer, if recording was requested."""
        if self.record_path is None:
            return
        import imageio.v2 as imageio

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.record_path, fps=NATIVE_FPS, macro_block_size=1)

    def _record(self, frame: np.ndarray) -> None:
        """Append one frame to the mp4."""
        if self._writer is not None:
            self._writer.append_data(np.ascontiguousarray(frame))

    def close(self) -> None:
        """Release the env, the writer and any window."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        try:
            self.env.close()
        except Exception:  # pragma: no cover - env may already be closed
            pass
        if not self.headless:
            try:
                import cv2

                cv2.destroyWindow(self.window_name)
            except Exception:  # pragma: no cover - no window was ever created
                pass

    # ------------------------------------------------------------------------- main loops

    def run_headless(self, frames: int | None = None) -> int:
        """Run with no window, recording to mp4. Returns the number of frames rendered."""
        self._open_writer()
        self.reset()
        limit = frames if frames is not None else self.max_frames
        rendered = 0
        try:
            while rendered < limit and not self.state.episode_over:
                self._record(self.step())
                rendered += 1
        finally:
            self.close()
        return rendered

    def run(self) -> int:
        """Run the interactive window, recording to mp4 alongside it.

        Falls back to :meth:`run_headless` if a window cannot be created, so the recording is
        produced either way.

        Returns:
            The number of frames rendered.
        """
        if self.headless:
            return self.run_headless()

        try:
            import cv2

            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as error:  # pragma: no cover - depends on the display stack
            print(f"could not open a window ({error}); falling back to headless recording")
            self.headless = True
            return self.run_headless()

        self._open_writer()
        self.reset()
        rendered = 0
        try:
            frame = self.render()
            while self.state.running:
                if self.should_advance():
                    frame = self.step()
                    self._record(frame)
                    rendered += 1
                cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.handle_key(cv2.waitKey(self.state.frame_delay_ms()) & 0xFF)
        finally:
            self.close()
        return rendered
