#!/usr/bin/env python
"""Live comparison dashboard: watch two agents face the same perturbation.

Run with::

    .venv/bin/streamlit run scripts/dashboard.py

Why this exists
---------------
The project's headline result is a *behavioural* difference: under 25% sticky actions, A*
executing a fixed plan completes 15% of episodes while the same planner replanning every 15
frames completes 80%. In a report that is two numbers in a table. Here it is two Marios side by
side, and a slider the examiner can drag.

**The demonstration:** start both A* agents with sticky at 0.00 -- both reach the flag. Reset,
then raise sticky to 0.25 mid-run. The open-loop plan desynchronises and Mario dies; the
closed-loop planner visibly redraws its path and keeps going.

All numbers on the Results tab are read from the Phase 5 CSVs, never recomputed, so what the
examiner sees matches the report exactly.
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.factory import AGENT_NAMES, agent_availability  # noqa: E402
from src.config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR, load_config  # noqa: E402
from src.viz.comparison import SIDES, ComparisonSession  # noqa: E402
from src.viz.live_demo import OVERLAY_MODES  # noqa: E402

#: Seconds between UI ticks. ~20 Hz; the emulator advances `frames_per_tick` frames per tick.
TICK_SECONDS = 0.05

#: Largest displayed panel width in pixels. The panel is laid out as a fixed-aspect box that
#: shrinks with the column, so it never forces a horizontal scrollbar on a narrow window, and
#: never reflows between redraws -- a resizing image reads as flicker even when correct.
#: The NES frame is 256 px wide and the runner renders it at scale 2.
PANEL_MAX_WIDTH = 512

#: PNG compression for the live frames. Level 3 costs ~1 ms on a 512x480 frame and gets flat
#: pixel art down to a few KB; level 0 is 720 KB per frame, which the websocket feels at 20 Hz.
PNG_COMPRESS_LEVEL = 3

#: Readable names for the dropdowns.
LABELS = {
    "astar_open_loop": "A* open-loop (fixed plan)",
    "astar_closed_loop": "A* closed-loop (replans)",
    "ppo": "PPO (pixels)",
    "random": "Random",
    "human": "Human (keyboard — OpenCV demo only)",
}
OVERLAY_LABELS = ["Everything", "Plan + enemies", "Plan only", "Raw game"]


@st.cache_data
def load_configs() -> tuple[dict, dict]:
    """Load the demo and planner configs once per server process."""
    return load_config("demo.yaml"), load_config("astar.yaml")


@st.cache_data
def load_figure(path_str: str, mtime: float) -> bytes:
    """Read a figure once and cache it.

    Keyed on mtime as well as path, so regenerating a figure refreshes it rather than
    serving a stale cache.
    """
    return Path(path_str).read_bytes()


@st.cache_data
def load_summary():
    """Read the Phase 5 summary table. Never recomputed — the viva must match the report."""
    import pandas as pd

    path = TABLES_DIR / "phase5_summary_main.csv"
    return pd.read_csv(path) if path.exists() else None


def close_session() -> None:
    """Release the current session's emulators, if any.

    Called before building a new session. Each side owns a native emulator, so skipping this
    leaks one per dropdown change.
    """
    session = st.session_state.get("session")
    if session is not None:
        session.close()
        st.session_state["session"] = None


def sidebar(config: dict) -> dict:
    """Draw the controls and return the chosen settings."""
    st.sidebar.title("Comparison")

    level = st.sidebar.selectbox("Level", ["1-1", "1-2"], index=0)
    selectable = [name for name in AGENT_NAMES if name != "human"]

    def options(default: str) -> list[str]:
        return [n for n in selectable if agent_availability(n, level, config)[0]] or [default]

    available = options("random")
    left = st.sidebar.selectbox(
        "Left agent", available, index=available.index("astar_open_loop")
        if "astar_open_loop" in available else 0, format_func=lambda n: LABELS.get(n, n),
    )
    right = st.sidebar.selectbox(
        "Right agent", available, index=available.index("astar_closed_loop")
        if "astar_closed_loop" in available else 0, format_func=lambda n: LABELS.get(n, n),
    )

    unavailable = [
        (n, agent_availability(n, level, config)[1]) for n in selectable
        if not agent_availability(n, level, config)[0]
    ]
    for name, reason in unavailable:
        st.sidebar.caption(f"{LABELS.get(name, name)} unavailable — {reason}")

    st.sidebar.divider()
    sticky = st.sidebar.slider(
        "Sticky actions  p", 0.0, 0.5, 0.0, 0.05,
        help="Probability that the previous control repeats instead of the commanded one. "
             "Drag this up mid-run — that is the demonstration.",
    )
    seed = st.sidebar.number_input("Seed", 0, 999, 0, 1)
    frames_per_tick = st.sidebar.slider(
        "Speed  (emulator frames per tick)", 1, 6, 2,
        help="The NES runs at 60 fps and the browser ticks at ~20 Hz, so 3 is about real time. "
             "Lower it if the picture stutters.",
    )
    overlay = st.sidebar.selectbox("Overlay", range(len(OVERLAY_MODES)), index=0,
                                   format_func=lambda i: OVERLAY_LABELS[i])

    st.sidebar.divider()
    running = st.sidebar.toggle("Run", value=False)
    if st.sidebar.button("Reset episode", width='stretch'):
        session = st.session_state.get("session")
        if session is not None:
            session.reset()

    st.sidebar.divider()
    recorded = st.sidebar.toggle(
        "Play recorded video instead", value=False,
        help="Fallback for presenting: plays the mp4s in media/ with no emulator running.",
    )
    return {
        "level": level, "left": left, "right": right, "sticky": sticky, "seed": seed,
        "frames_per_tick": frames_per_tick, "overlay": overlay, "running": running,
        "recorded": recorded,
    }


def ensure_session(settings: dict, config: dict, astar_config: dict) -> ComparisonSession:
    """Build the session, rebuilding only when a structural setting changed."""
    key = (settings["left"], settings["right"], settings["level"], settings["seed"])
    if st.session_state.get("session_key") != key or st.session_state.get("session") is None:
        close_session()
        st.session_state["session"] = ComparisonSession(
            left_name=settings["left"], right_name=settings["right"],
            config=config, astar_config=astar_config, level=settings["level"],
            sticky_p=settings["sticky"], seed=settings["seed"], scale=2,
            frames_per_tick=settings["frames_per_tick"],
        )
        st.session_state["session_key"] = key

    session = st.session_state["session"]
    # These are cheap to apply every rerun, and must track the sliders live.
    session.frames_per_tick = settings["frames_per_tick"]
    session.set_sticky(settings["sticky"])
    session.set_overlay(settings["overlay"])
    return session


def frame_to_data_uri(frame: np.ndarray) -> str:
    """Encode one rendered frame as an inline ``data:`` URI.

    ``st.image`` does not survive a 20 Hz fragment: it writes each frame into Streamlit's
    media-file manager under a fresh URL and drops the previous one on the next rerun, so a
    browser request issued 50 ms late arrives after its file has been collected and the panel
    shows a broken image. Inlining the bytes removes the second round trip altogether -- the
    frame arrives in the same websocket message that lays out the panel, so it cannot miss.
    """
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_panel(column, side: str, payload: dict) -> None:
    """Render one agent's frame and its configuration readout."""
    stats = payload["stats"]
    frame = payload["frame"]
    height, width = frame.shape[:2]
    with column:
        st.subheader(LABELS.get(payload["agent"], payload["agent"]))
        # The wrapper reserves the box before the image paints: `aspect-ratio` plus a capped
        # percentage width means the surrounding text never moves as frames swap, and two
        # panels always fit the window instead of overflowing it.
        st.html(
            f'<img src="{frame_to_data_uri(frame)}" alt="game frame" style="width:100%;'
            f'max-width:{PANEL_MAX_WIDTH}px;aspect-ratio:{width}/{height};display:block;'
            'image-rendering:pixelated;border-radius:6px;background:#000;">'
        )

        if stats["flag_get"]:
            st.success("Reached the flag")
        elif stats["episode_over"]:
            st.error("Episode over")

        a, b, c, d = st.columns(4)
        a.metric("x", stats["x"])
        b.metric("vx", f"{stats['vx']:+.2f}")
        c.metric("on ground", "yes" if stats["on_ground"] else "no")
        d.metric("frame", stats["frame"])
        st.caption(
            f"action `{stats['action']}` · enemies {stats['enemies']} · "
            f"sticky realised {stats['sticky_realised']:.2f}"
        )


def render_panels(session: ComparisonSession, running: bool) -> None:
    """Draw both panels, advancing the episode first when running.

    This is the only part of the page that updates during a run. It is wrapped in
    ``st.fragment`` by :func:`live_tab`, so a rerun of *this function* does not re-execute the
    script: the sidebar, title, tab bar and the Results tab all stay mounted.

    That distinction is the whole fix. Driving the loop with ``st.rerun()`` re-ran the entire
    script ~20x/s, and because Streamlit renders every tab body rather than only the visible
    one, each tick also re-sent the Results tab's figures. Every element got a new identity, so
    the browser replaced DOM nodes instead of patching them -- which is what the flicker was.

    **Must not call** ``st.rerun()``: the fragment's own timer paces it.
    """
    tick_ms: float | None = None
    if running and not session.finished:
        started = time.perf_counter()
        payload = session.tick()
        tick_ms = (time.perf_counter() - started) * 1e3
    else:
        payload = {
            side: {
                "frame": session.runners[side].render(),
                "stats": session.runners[side].stats(),
                "agent": session.names[side],
            }
            for side in SIDES
        }

    columns = st.columns(2)
    for column, side in zip(columns, SIDES):
        draw_panel(column, side, payload[side])

    if tick_ms is not None:
        st.caption(
            f"{tick_ms:.0f} ms per tick · {session.frames_per_tick} frames per tick. "
            "Pauses on a closed-loop panel are the planner searching, not a stall."
        )
    elif session.finished:
        st.info("Both episodes finished — press Reset to run again.")
    else:
        st.caption("Switch on **Run** in the sidebar to start.")


def live_tab(settings: dict, config: dict, astar_config: dict) -> None:
    """The side-by-side comparison."""
    if settings["recorded"]:
        st.info("Playing recorded runs — no emulator is running.")
        columns = st.columns(2)
        for column, side in zip(columns, SIDES):
            name = settings[side]
            path = PROJECT_ROOT / "media" / f"demo_{name}_{settings['level']}.mp4"
            with column:
                st.subheader(LABELS.get(name, name))
                if path.exists():
                    st.video(str(path))
                else:
                    st.warning(f"No recording at {path.relative_to(PROJECT_ROOT)}")
        return

    session = ensure_session(settings, config, astar_config)

    # `run_every=None` makes this a plain one-shot render, so pausing genuinely stops the
    # timer rather than leaving it spinning on a session that is not advancing.
    should_tick = settings["running"] and not session.finished
    fragment = st.fragment(run_every=TICK_SECONDS if should_tick else None)
    fragment(render_panels)(session, should_tick)

    st.caption(
        "Both panels run the same level from the same seed and face the same perturbation "
        "*process*. They cannot face the same noise *sequence*: sticky actions repeat the "
        "previously executed action, so once the agents act differently their override "
        "sequences necessarily differ. Each panel reports its own realised rate."
    )


def planner_tab(settings: dict) -> None:
    """What the search is doing, for whichever side plans."""
    session = st.session_state.get("session")
    if session is None or settings["recorded"]:
        st.info("Start a live run to see planner internals.")
        return

    planning = False
    columns = st.columns(2)
    for column, side in zip(columns, SIDES):
        stats = session.runners[side].stats()
        with column:
            st.subheader(LABELS.get(session.names[side], session.names[side]))
            if "replans" not in stats:
                st.caption(
                    "This agent does not plan. PPO maps pixels to a button with no lookahead; "
                    "Random has no model at all."
                )
                continue
            planning = True
            a, b = st.columns(2)
            a.metric("replans", stats["replans"])
            b.metric("nodes expanded", stats["nodes_expanded"])
            c, d = st.columns(2)
            c.metric("time searching", f"{stats['planning_seconds']:.1f} s")
            d.metric("failed replans", stats.get("failed_replans", 0))

    if planning:
        st.caption(
            "A failed replan means no safe trajectory was found inside the budget; the agent "
            "keeps its previous plan rather than stalling. On level 1-2 almost every replan "
            "fails — the heuristic measures horizontal distance only, which carries little "
            "information on a vertically structured level."
        )
        st.caption("The dotted path on each frame is the trajectory that side is committed to.")


def results_tab() -> None:
    """The measured Phase 5 results, read from disk."""
    st.subheader("Measured results")
    figures = [
        ("comparison_1_robustness.png", "Completion before and after perturbation"),
        ("comparison_2_cost_vs_robustness.png", "What each approach paid, and what it bought"),
        ("comparison_3_failure_modes.png", "How each agent's episodes ended"),
        ("comparison_4_transfer.png", "On a level neither agent prepared for"),
    ]
    for name, caption in figures:
        path = FIGURES_DIR / name
        if path.exists():
            # Cached on (path, mtime) so a full rerun does not re-read 444 KB from disk,
            # while a regenerated figure still refreshes.
            st.image(load_figure(str(path), path.stat().st_mtime), caption=caption,
                     width='stretch')

    summary = load_summary()
    if summary is not None:
        st.subheader("Phase 5 summary")
        st.dataframe(summary, width='stretch', hide_index=True)
        st.caption(f"Read from {(TABLES_DIR / 'phase5_summary_main.csv').relative_to(PROJECT_ROOT)} "
                   "— these are the report's numbers, not recomputed.")
    else:
        st.warning("No summary CSV yet — run scripts/phase5_eval.py")


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="A* vs PPO — Super Mario Bros", layout="wide")
    config, astar_config = load_configs()
    settings = sidebar(config)

    st.title("Search-based planning vs. reinforcement learning")
    st.caption(
        "Two agents, one level, one perturbation. Raise **sticky actions** in the sidebar "
        "while they run: the fixed plan desynchronises, the replanner recovers."
    )

    live, planner, results = st.tabs(["Live comparison", "Planner internals", "Results"])
    with live:
        live_tab(settings, config, astar_config)
    with planner:
        planner_tab(settings)
    with results:
        results_tab()


if __name__ == "__main__":
    main()
