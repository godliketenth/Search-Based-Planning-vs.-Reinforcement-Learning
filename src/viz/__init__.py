"""Visualisation: the debug overlay and the live demo.

:mod:`src.viz.debug_overlay` was built in Phase 2 as a verification tool -- a tile grid
extracted from RAM cannot be checked by reading numbers, but a one-column misalignment drawn
over the real frame is obvious. It was designed from the start to accept a planned path and a
replan flash, so :mod:`src.viz.live_demo` reuses it unchanged.
"""

from .debug_overlay import default_hud_lines, render_overlay
from .live_demo import DemoRunner, DemoState

__all__ = ["DemoRunner", "DemoState", "default_hud_lines", "render_overlay"]
