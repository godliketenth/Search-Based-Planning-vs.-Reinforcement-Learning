#!/usr/bin/env python
"""Demo launcher: everything this project can show, from one terminal menu.

The project has accumulated several ways to be demonstrated -- five agents in the live
OpenCV window, two agents side by side, a Streamlit dashboard, a table and four figures.
Remembering the flags for each of those while someone is watching is not the moment to be
reading ``--help``, so this script puts them behind a numbered menu, prints the exact
command it is about to run, and runs it.

Printing the command matters: by the end of a session the commands are familiar, and the
menu stops being load-bearing. It is a convenience over the existing entry points, not a
replacement for them -- nothing here can be done only from the menu.

Each demo runs as a child process, so a crash in one returns to the menu instead of taking
it down, and each emulator is definitely released when its window closes.

Usage
-----
    .venv/bin/python scripts/showcase.py                      # the menu
    .venv/bin/python scripts/showcase.py --pair astar_open_loop,astar_closed_loop
    .venv/bin/python scripts/showcase.py --pair ppo,astar_closed_loop --sticky 0.25
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR, load_config  # noqa: E402
from src.agents.factory import agent_availability  # noqa: E402

#: Readable name and one-line pitch per agent, in the order they are offered.
AGENTS: dict[str, tuple[str, str]] = {
    "astar_closed_loop": ("A* closed-loop (replans)", "receding horizon, every 15 frames"),
    "astar_open_loop": ("A* open-loop (fixed plan)", "one plan, replayed blind"),
    "ppo": ("PPO (from pixels)", "5.0M steps, 74 min of training"),
    "random": ("Random", "the floor every method beats"),
    "human": ("Human (you play)", "same harness, your keyboard"),
}

#: The side-by-side pairings worth having a menu entry. The first is the project's result.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("astar_open_loop", "astar_closed_loop", "the headline result"),
    ("astar_closed_loop", "ppo", "planning vs. learning"),
)

#: Panel upscaling for the side-by-side window. The single-agent demo uses 3 from
#: ``configs/demo.yaml``, but two panels at that scale are 1540 px wide and do not fit a
#: laptop screen; 2 matches the dashboard. ``--scale`` still overrides it.
PAIR_SCALE = 2

#: Levels the project evaluated. 1-2 is the transfer level -- no cached plan exists for it,
#: so the open-loop agent reports itself unavailable there rather than failing mid-demo.
LEVELS: tuple[str, ...] = ("1-1", "1-2")

#: Columns of the Phase 5 summary worth reading aloud, as (CSV key, header, width). Widths
#: include the gap to the next column, so the longest value ("deterministic") still separates.
TABLE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("agent", "agent", 20),
    ("condition", "condition", 16),
    ("completion_rate", "completion", 12),
    ("deaths_enemy", "enemy", 7),
    ("deaths_pit", "pit", 5),
    ("decision_ms_mean", "ms/decision", 11),
)

#: The one sentence worth saying about each level's numbers once they are on screen.
TABLE_NOTES: dict[str, tuple[str, ...]] = {
    "1-1": (
        "A* open-loop drops 1.00 -> 0.15 under sticky actions; the same planner",
        "replanning every 15 frames holds 0.80. That gap is the project.",
    ),
    "1-2": (
        "Neither method transfers: the planner reaches 950 px before its node budget",
        "runs out, and PPO stalls at 354 -- barely past random's 355.",
    ),
}


# --------------------------------------------------------------------------- launching

def run(command: list[str]) -> None:
    """Print a command, run it to completion, and report a non-zero exit."""
    print("\n$ " + " ".join(command) + "\n")
    try:
        code = subprocess.call(command, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return
    # ctrl-c is how the dashboard and the human demo are meant to end, so the signal exit
    # codes it produces are not worth reporting as failures.
    if code not in (0, 130, -2):
        print(f"\n[exited with code {code}]")
    input("\npress enter to return to the menu ")


def python() -> str:
    """The interpreter running this script, so children use the same venv."""
    return sys.executable


def demo_command(agent: str, level: str, sticky: float) -> list[str]:
    """The ``scripts/demo.py`` invocation for one agent."""
    return [python(), "scripts/demo.py", "--agent", agent,
            "--level", level, "--sticky", f"{sticky}"]


def pair_command(left: str, right: str, level: str, sticky: float) -> list[str]:
    """This script's own non-interactive side-by-side mode."""
    return [python(), "scripts/showcase.py", "--pair", f"{left},{right}",
            "--level", level, "--sticky", f"{sticky}"]


def dashboard_command() -> list[str]:
    """Serve the Streamlit dashboard. Stays in the foreground; ctrl-c returns here."""
    return [python(), "-m", "streamlit", "run", "scripts/dashboard.py"]


def open_paths(paths: list[Path], what: str) -> None:
    """Hand files to the desktop's file opener, or list them if there is none."""
    existing = [path for path in paths if path.exists()]
    if not existing:
        print(f"\nno {what} on disk yet.")
        return
    opener = {"Darwin": "open", "Linux": "xdg-open", "Windows": "start"}.get(platform.system())
    if opener is None:
        print(f"\n{what}:")
        for path in existing:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
        return
    print(f"\nopening {len(existing)} {what}...")
    subprocess.call([opener, *[str(path) for path in existing]])
    input("\npress enter to return to the menu ")


# ------------------------------------------------------------------------------ numbers

def print_table(level: str) -> None:
    """Print the measured Phase 5 completion table for one level.

    Read straight from the CSV the report and the slides quote, so the menu cannot show a
    number that disagrees with the written-up result.
    """
    path = TABLES_DIR / "phase5_summary_main.csv"
    if not path.exists():
        print("\nno summary CSV yet -- run scripts/phase5_eval.py")
        return

    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["level"] == level]

    print(f"\n  Completion on {level}, from {path.relative_to(PROJECT_ROOT)}\n")
    print("  " + "".join(header.ljust(width) for _, header, width in TABLE_COLUMNS))
    print("  " + "-" * sum(width for _, _, width in TABLE_COLUMNS))
    for row in rows:
        cells = []
        for key, _, width in TABLE_COLUMNS:
            value = row[key]
            if key == "completion_rate":
                value = f"{float(value):.2f}"
            elif key == "decision_ms_mean":
                value = f"{float(value):.3f}"
            cells.append(value.ljust(width))
        print("  " + "".join(cells))
    print()
    for line in TABLE_NOTES.get(level, ()):
        print(f"  {line}")


# --------------------------------------------------------------------------------- menu

def availability(level: str, config: dict) -> dict[str, tuple[bool, str]]:
    """Check every agent once, so the menu can mark entries before they are chosen."""
    return {name: agent_availability(name, level, config) for name in AGENTS}


def draw_menu(level: str, sticky: float, status: dict[str, tuple[bool, str]]) -> None:
    """Print the menu."""
    rule = "=" * 74
    print(f"\n{rule}")
    print("  Search-Based Planning vs. Reinforcement Learning -- demo launcher")
    print(f"  level {level}    sticky-when-toggled {sticky:.2f}")
    print(rule)

    print("\n  ONE AGENT  (window + mp4; s toggles sticky mid-run)")
    for index, (name, (label, pitch)) in enumerate(AGENTS.items(), start=1):
        available, reason = status[name]
        note = pitch if available else f"unavailable -- {reason}"
        print(f"    {index}  {label:<26} {note}")

    print("\n  TWO AGENTS, SIDE BY SIDE  (one window, same seed, same perturbation)")
    for offset, (left, right, pitch) in enumerate(PAIRS):
        pairing = f"{AGENTS[left][0]}  vs  {AGENTS[right][0]}"
        blocked = [name for name in (left, right) if not status[name][0]]
        note = pitch if not blocked else f"unavailable -- {status[blocked[0]][1]}"
        print(f"    {len(AGENTS) + offset + 1}  {pairing}")
        print(f"       {note}")

    print("\n  OTHER")
    print("    d  Streamlit dashboard in the browser (live, planner internals, results)")
    print("    t  print the measured completion table")
    print("    f  open the four comparison figures")
    print("    m  open the recorded mp4s")
    print("\n  SETTINGS")
    print(f"    p  sticky probability        (now {sticky:.2f})")
    print(f"    l  level                     (now {level})")
    print("    q  quit")


def ask_sticky(current: float) -> float:
    """Read a new sticky probability, keeping the old one on bad or empty input."""
    raw = input(f"\nsticky probability applied when you press 's' [{current:.2f}]: ").strip()
    if not raw:
        return current
    try:
        value = float(raw)
    except ValueError:
        print("not a number; unchanged")
        return current
    if not 0.0 <= value <= 1.0:
        print("must be between 0 and 1; unchanged")
        return current
    return value


def ask_level(current: str) -> str:
    """Read a new level from the evaluated set."""
    raw = input(f"\nlevel {'/'.join(LEVELS)} [{current}]: ").strip()
    if not raw:
        return current
    if raw not in LEVELS:
        print(f"not one of {', '.join(LEVELS)}; unchanged")
        return current
    return raw


def menu(config: dict) -> None:
    """The interactive loop."""
    level = config["level"]
    sticky = float(config["sticky_p"]) or 0.25
    agents = list(AGENTS)

    while True:
        status = availability(level, config)
        draw_menu(level, sticky, status)
        try:
            choice = input("\nchoice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ("q", "quit", "exit"):
            return
        if choice == "p":
            sticky = ask_sticky(sticky)
        elif choice == "l":
            level = ask_level(level)
        elif choice == "d":
            print("\nctrl-c in this terminal stops the server and returns to the menu.")
            run(dashboard_command())
        elif choice == "t":
            print_table(level)
            input("\npress enter to return to the menu ")
        elif choice == "f":
            open_paths([FIGURES_DIR / f"comparison_{index}_{name}.png" for index, name in
                        ((1, "robustness"), (2, "cost_vs_robustness"),
                         (3, "failure_modes"), (4, "transfer"))], "figures")
        elif choice.isdigit() and 1 <= int(choice) <= len(agents):
            name = agents[int(choice) - 1]
            available, reason = status[name]
            if not available:
                print(f"\n{AGENTS[name][0]} is unavailable -- {reason}")
                continue
            run(demo_command(name, level, sticky))
        elif choice.isdigit() and len(agents) < int(choice) <= len(agents) + len(PAIRS):
            left, right, _ = PAIRS[int(choice) - len(agents) - 1]
            blocked = [name for name in (left, right) if not status[name][0]]
            if blocked:
                print(f"\n{AGENTS[blocked[0]][0]} is unavailable -- {status[blocked[0]][1]}")
                continue
            run(pair_command(left, right, level, sticky))
        elif choice == "m":
            open_paths(sorted((PROJECT_ROOT / config["record_dir"]).glob("*.mp4")), "recordings")
        else:
            print("\nnot a choice")


# ------------------------------------------------------------------------- side by side

def run_pair(left: str, right: str, config: dict, astar_config: dict, sticky: float,
             scale: int) -> None:
    """Open the two-panel window. The non-interactive half of this script."""
    from src.viz.comparison import ComparisonSession
    from src.viz.comparison_window import ComparisonWindow

    level = config["level"]
    for name in (left, right):
        available, reason = agent_availability(name, level, config)
        if not available:
            raise SystemExit(f"{name} is unavailable on {level}: {reason}")

    print(f"building both agents on {level}...")
    session = ComparisonSession(
        left_name=left,
        right_name=right,
        config=config,
        astar_config=astar_config,
        level=level,
        sticky_p=0.0,
        seed=config["seed"],
        scale=scale,
    )
    record_path = PROJECT_ROOT / config["record_dir"] / f"pair_{left}_vs_{right}_{level}.mp4"
    window = ComparisonWindow(
        session,
        labels={"left": AGENTS[left][0], "right": AGENTS[right][0]},
        sticky_p=sticky,
        goal_x=config["goal_x"].get(level),
        record_path=record_path,
        window_name=f"{left} vs {right} -- {level}",
    )

    print("=" * 68)
    print(f"  {AGENTS[left][0]}   vs   {AGENTS[right][0]}")
    print("=" * 68)
    print(f"  both from seed {config['seed']}, so any divergence is the agents', not luck")
    print(f"  press 's' to apply sticky actions at p = {sticky:.2f} to both panels")
    print("  space pause | n step | [ ] speed | o overlay | r restart | q quit")
    print(f"  recording  : {record_path.relative_to(PROJECT_ROOT)}\n")

    try:
        ticks = window.run()
    finally:
        session.close()
    print(f"\nadvanced {ticks} ticks ({ticks * session.frames_per_tick} emulator frames)")
    print(f"mp4: {record_path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------------- entry

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="demo.yaml")
    parser.add_argument("--astar-config", default="astar.yaml")
    parser.add_argument("--pair", help="run two agents side by side: 'left,right'")
    parser.add_argument("--level", help="override: 'world-stage'")
    parser.add_argument("--sticky", type=float, dest="sticky_p",
                        help="override: probability used when the sticky toggle is on")
    parser.add_argument("--scale", type=int, help="override: panel upscaling")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(
        args.config,
        {"level": args.level, "sticky_p": args.sticky_p, "scale": args.scale},
    )

    if args.pair:
        names = [name.strip() for name in args.pair.split(",")]
        if len(names) != 2 or any(name not in AGENTS for name in names):
            raise SystemExit(f"--pair wants two of {', '.join(AGENTS)} separated by a comma")
        astar_config = load_config(args.astar_config)
        run_pair(names[0], names[1], config, astar_config,
                 float(config["sticky_p"]) or 0.25, args.scale or PAIR_SCALE)
        return

    menu(config)


if __name__ == "__main__":
    main()
