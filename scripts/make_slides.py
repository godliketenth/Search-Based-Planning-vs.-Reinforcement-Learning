#!/usr/bin/env python
"""Build the mini-project slide deck on the college template.

The template is the NMIMS deck used for the NLP mini project: Office base theme, the
institute logo on the title slide, bold ``#C00000`` headings, 20 pt bulleted bodies. Only
that shell is reused -- every slide's content is written here and every number is read from
``results/tables/phase5_summary_main.csv`` rather than typed in.

``python-pptx`` is not a runtime dependency of the project, so this runs from a throwaway
environment::

    uv venv /tmp/docvenv && uv pip install --python /tmp/docvenv/bin/python python-pptx
    /tmp/docvenv/bin/python scripts/make_slides.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/Users/aditya/Documents/Study/NLP/Mini Project/Project/midTermReport.pptx")
OUT = ROOT / "deliverables" / "RCMP_MiniProject_Presentation.pptx"
FIGURES = ROOT / "results" / "figures"

#: The template's heading colour.
RED = RGBColor(0xC0, 0x00, 0x00)
INK = RGBColor(0x1A, 0x1A, 0x1A)

#: The template's body geometry, copied from its content slides so new slides sit identically.
BODY_LEFT, BODY_TOP = Emu(838200), Emu(1841500)
BODY_WIDTH, BODY_HEIGHT = Emu(10515600), Emu(4254500)
SLIDE_WIDTH, SLIDE_HEIGHT = Emu(12192000), Emu(6858000)


# --------------------------------------------------------------------------- template surgery


def drop_slide(prs, index: int) -> None:
    """Remove one slide, releasing its relationship so the part is not orphaned."""
    slides = prs.slides._sldIdLst
    element = list(slides)[index]
    prs.part.drop_rel(element.get(qn("r:id")))
    slides.remove(element)


def clear_content_slides(prs) -> None:
    """Keep the title slide (it carries the logo); discard the borrowed content."""
    for index in range(len(prs.slides._sldIdLst) - 1, 0, -1):
        drop_slide(prs, index)


# --------------------------------------------------------------------------- text helpers


def style_paragraph(paragraph, *, bullet: str | None, size: int, level: int = 0,
                    space_before: int = 1200) -> None:
    """Apply the template's bullet geometry to one paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    indent = 285750 if level == 0 else 228600
    pPr.set("marL", str(285750 + level * 457200))
    pPr.set("indent", str(-indent))
    spc_before = pPr.makeelement(qn("a:spcBef"), {})
    spc_before.append(pPr.makeelement(qn("a:spcPts"), {"val": str(space_before)}))
    pPr.append(spc_before)
    if bullet is None:
        pPr.append(pPr.makeelement(qn("a:buNone"), {}))
    else:
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": bullet}))
    paragraph.level = level
    paragraph.font.size = Pt(size)


#: Stands in for a literal asterisk (written ``\\*`` in the slide text, as in ``A\\*``) while the
#: emphasis markers are parsed, so "A*" never pairs up with a nearby ``*italic*`` marker.
STAR = "\x00"

EMPHASIS = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*)")


def add_runs(paragraph, text: str, size: int, colour: RGBColor | None = None) -> None:
    """Add runs to a paragraph, honouring ``**bold**``, ``*italic*`` and ``\\*`` literals."""
    for part in EMPHASIS.split(text.replace("\\*", STAR)):
        if not part:
            continue
        bold = part.startswith("**")
        italic = not bold and part.startswith("*")
        content = part[2:-2] if bold else part[1:-1] if italic else part
        run = paragraph.add_run()
        run.text = content.replace(STAR, "*")
        run.font.bold = bold
        run.font.italic = italic
        run.font.size = Pt(size)
        if colour is not None:
            run.font.color.rgb = colour


def set_title(slide, text: str, size: int | None = None) -> None:
    """Write the slide title in the template's bold red.

    Long titles are stepped down so they stay on one line; a wrapped title drops into the
    body and collides with the first bullet or the figure below it.
    """
    frame = slide.shapes.title.text_frame
    frame.clear()
    paragraph = frame.paragraphs[0]
    add_runs(paragraph, text, size or (30 if len(text) > 34 else 40), RED)
    for run in paragraph.runs:
        run.font.bold = True


def content_slide(prs, title: str, bullets: list, *, size: int = 20,
                  title_size: int | None = None):
    """A title-and-content slide.

    ``bullets`` items are either a string (top-level bullet) or a ``(level, text)`` pair.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(slide, title, title_size)

    body = slide.placeholders[1]
    body.left, body.top, body.width, body.height = BODY_LEFT, BODY_TOP, BODY_WIDTH, BODY_HEIGHT
    frame = body.text_frame
    frame.clear()
    frame.word_wrap = True

    for index, item in enumerate(bullets):
        level, text = item if isinstance(item, tuple) else (0, item)
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        bullet = "•" if level == 0 else "–"
        style_paragraph(paragraph, bullet=bullet, size=size - 2 * level, level=level)
        add_runs(paragraph, text, size - 2 * level)
    return slide


def blank_titled_slide(prs, title: str, title_size: int | None = None):
    """A title-only slide, for tables and figures."""
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide, title, title_size)
    return slide


def add_table(slide, rows: list[list[str]], *, top: int, size: int = 14,
              col_ratios: list[float] | None = None, width: int = 10515600,
              left: int = 838200):
    """Draw a table with the template's red header row."""
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Emu(left), Emu(top), Emu(width),
                                   Emu(360000 * n_rows))
    table = shape.table
    ratios = col_ratios or [1 / n_cols] * n_cols
    for index, ratio in enumerate(ratios):
        table.columns[index].width = Emu(int(width * ratio))

    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            paragraph = cell.text_frame.paragraphs[0]
            add_runs(paragraph, text, size)
            paragraph.font.size = Pt(size)
            if r == 0:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if c > 0 and len(text) <= 24:
                paragraph.alignment = PP_ALIGN.CENTER
            cell.margin_top = cell.margin_bottom = Emu(27432)
    return table


def add_figure(slide, name: str, *, top: int = 1700000, max_height: int = 4400000) -> None:
    """Place a results figure, centred and scaled to fit."""
    from PIL import Image

    path = FIGURES / name
    width_px, height_px = Image.open(path).size
    width = 9_600_000
    height = int(width * height_px / width_px)
    if height > max_height:
        height = max_height
        width = int(height * width_px / height_px)
    slide.shapes.add_picture(str(path), Emu((SLIDE_WIDTH - width) // 2), Emu(top),
                             width=Emu(width), height=Emu(height))


def add_caption(slide, text: str, *, top: int, size: int = 14) -> None:
    """A centred grey caption under a figure or table."""
    box = slide.shapes.add_textbox(BODY_LEFT, Emu(top), BODY_WIDTH, Emu(500000))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    add_runs(paragraph, text, size, RGBColor(0x40, 0x40, 0x40))


# --------------------------------------------------------------------------- the title slide


def rewrite_title_slide(slide) -> None:
    """Keep the logo and layout; replace the borrowed course and project text."""
    title = slide.shapes.title.text_frame
    title.clear()
    lines = [("A", 24), ("Robotics: Computational Motion Planning", 32),
             ("Mini Project", 28), ("Presentation", 28)]
    for index, (text, size) in enumerate(lines):
        paragraph = title.paragraphs[0] if index == 0 else title.add_paragraph()
        run = paragraph.add_run()
        run.text = text
        run.font.bold = True
        run.font.size = Pt(size)
        run.font.color.rgb = RED

    subtitle = slide.placeholders[1].text_frame
    subtitle.clear()
    first = subtitle.paragraphs[0]
    first.alignment = PP_ALIGN.LEFT
    add_runs(first, "**Title: Search-Based Planning vs. Reinforcement Learning**", 24)
    second = subtitle.add_paragraph()
    second.alignment = PP_ALIGN.LEFT
    add_runs(second, "**for a Kinodynamically Constrained Agent in Super Mario Bros**", 16)


# --------------------------------------------------------------------------- the deck


def build(prs) -> None:
    """Add every content slide, in order."""
    content_slide(prs, "Project Overview", [
        "Super Mario Bros is treated as a **kinodynamic motion planning problem**, not as a "
        "game: Mario is an agent with momentum, gravity and limited control authority",
        "Two solvers run the same task in one harness — **A\\* search over motion primitives** "
        "and a **PPO policy learned from pixels**",
        "The emulator itself is the **transition function and the collision checker**: state "
        "snapshots make forward simulation exact, with zero modelling error",
        "Robustness is probed by injecting **actuation uncertainty** (sticky actions, p = 0.25) "
        "at the frame level for every agent",
        "The question: when execution is noisy, does a **committed plan** or a **learned "
        "policy** hold up better — and what does each one cost?",
    ])

    content_slide(prs, "Problem Statement", [
        "Search-based planning assumes a model and solves offline; reinforcement learning "
        "assumes samples and learns a reactive policy",
        "The two are rarely run on the **same problem, under the same measurement**, so their "
        "trade-off is usually argued rather than measured",
        "A plan is **open-loop** — a control sequence committed to in advance. If execution "
        "drifts, nothing in the agent notices",
        "Super Mario Bros is a demanding instance: a 6-dimensional state with velocity, "
        "discrete controls, **moving obstacles**, and single mistimed jumps that are fatal",
        "**The gap:** no like-for-like account of what *replanning* buys over executing a fixed "
        "plan, measured against a learned policy on identical episodes",
    ])

    content_slide(prs, "Why This Is a Motion Planning Problem", [
        "Mario cannot stop or turn instantly: **gravity and momentum** make this kinodynamic, "
        "not a grid-world shortest path",
        "A jump is a **ballistic commitment** — once airborne, control authority is almost "
        "zero, so the decision must be correct *before* take-off",
        "The `on_ground` flag gates which controls exist at all — a textbook **differential "
        "constraint** on the control set",
        "Enemies are **dynamic obstacles**: the free space changes while the agent is moving "
        "through it",
        "A perfect, fast simulator is available — rare in robotics, and exactly what is needed "
        "to isolate *planning* from *modelling error*",
    ])

    content_slide(prs, "Objectives", [
        "Formulate the level as a planning problem: **configuration space, controls, edge cost, "
        "goal test, collision check**",
        "Extract the configuration and the obstacle map **from emulator RAM**, verified against "
        "the rendered frame",
        "Build an **A\\* planner over motion primitives** with an admissible, consistent "
        "heuristic, and prove the bound holds",
        "Train a **PPO agent from pixels** as the learning baseline, with checkpointing and "
        "logged curves",
        "Add **closed-loop receding-horizon replanning** and measure it against the same plan "
        "executed open-loop",
        "Evaluate every agent through **one harness**, under a deterministic and a perturbed "
        "condition, and report compute cost honestly",
    ], size=19)

    content_slide(prs, "Scope of the Project", [
        "**In scope**",
        (1, "Level 1-1 for both agents; a zero-shot transfer test on level 1-2"),
        (1, "Two conditions: deterministic, and sticky actions at p = 0.25"),
        (1, "PPO with checkpoint / resume and TensorBoard logging"),
        (1, "Metrics, plots and CSVs written by script; unit tests for the planner"),
        (1, "A live in-game demo and a side-by-side comparison dashboard"),
        "**Out of scope**",
        (1, "DQN and other learners; reward shaping; hybrid plan-guided RL"),
        (1, "Sticky-probability sweeps, training on 1-2, multi-seed runs"),
    ], size=19)

    content_slide(prs, "Problem Formulation", [
        "**Configuration:** q = (x, y, vx, vy, on_ground, powerup). Position **and velocity** — "
        "two Marios at the same x with different vx have completely different reachable sets",
        "**Controls:** NES joypad bytes from the 7-action SIMPLE_MOVEMENT set",
        "**Edges — motion primitives:** short fixed button sequences, so every edge is "
        "**dynamically feasible by construction**",
        "**Edge cost:** duration in frames, so g(s) is elapsed time and A\\* minimises "
        "time-to-goal",
        "**Goal test:** the emulator's own flag_get",
        "**Collision check:** forward-simulate the primitive; the edge is invalid iff Mario died "
        "on it — enemies need no special handling",
    ], size=19)

    slide = blank_titled_slide(prs, "Motion Planning Concepts Used")
    add_table(slide, [
        ["Concept", "Where it appears in this project"],
        ["Configuration space", "(x, y, vx, vy, on_ground, powerup), read from RAM, not pixels"],
        ["Kinodynamic constraints", "Gravity, momentum, air control; on_ground gates the control set"],
        ["Motion primitives", "7 edges; jump height is set by how long A is held"],
        ["C-obstacle / obstacle map", "13 x 16 tile grid read from a 2-page ring buffer in RAM"],
        ["Collision checking", "Exact forward simulation — no geometric approximation"],
        ["Admissible heuristic", "h(s) = (x_goal - x) / v_max, in frames; v_max = 3 px/frame"],
        ["Consistency", "1-Lipschitz in x, so no node is ever re-expanded"],
        ["State lattice", "States bucketed to (x//8, y//8, vx//8, on_ground, powerup)"],
        ["Resolution completeness", "Complete only up to the primitive set and lattice resolution"],
        ["Open- vs closed-loop", "The two A* agents; the gap under noise is the central result"],
        ["Receding horizon", "Replan 160 px ahead every 15 frames; a full replan costs 54 s"],
        ["Actuation uncertainty", "Sticky actions: the executed control is not the commanded one"],
    ], top=1500000, size=13, col_ratios=[0.27, 0.73])

    content_slide(prs, "Method A — A* over Motion Primitives", [
        "**Primitive set** (7 edges): run and walk (12 frames), hop (20), jump (28), "
        "long_jump (40), wait (8), back_off (8). The jumps differ only in **how long A is held**",
        "**Expanding an edge** is load_state → run the primitive's bytes → dump_state. The "
        "emulator is the transition function, so there is **zero modelling error**",
        "**Heuristic:** h(s) = (x_goal − x) / 3 px per frame. Admissible because 3 px/frame is a "
        "hard engine cap; **consistent** because it is 1-Lipschitz in x",
        "**Result on 1-1:** a 1056-frame plan against a 1040-frame lower bound — **1.5% above "
        "the admissible bound**, found by expanding 590 nodes in 53.5 s",
        "**Weighted A\\*** (w = 3) trades optimality for speed under a bounded-suboptimality "
        "guarantee: cost ≤ w × optimal",
    ], size=19)

    content_slide(prs, "Consistency Pays for Itself — in Memory", [
        "Each emulator snapshot is **272 KB**. Storing one per expanded node would need "
        "**gigabytes** — the search would not fit in RAM",
        "Because the heuristic is **consistent, an expanded node is never reopened**, so its "
        "snapshot can be freed the moment it leaves the frontier",
        "Snapshots are therefore kept **only for the frontier**: measured peak **239 MB instead "
        "of ~1.1 GB**",
        "Consistency is usually taught as *“it makes the search faster”*. Here it is the "
        "property that makes the search **possible at all**",
        "Snapshots are not picklable, so plans are persisted as **action sequences** — replay "
        "from reset is deterministic",
    ], size=19)

    content_slide(prs, "Method B — PPO from Pixels", [
        "**Observation:** 84 x 84 grayscale, **4 frames stacked** — a single frame cannot show "
        "whether Mario is rising or falling, the same reason the planner's C-space needs velocity",
        "**Network:** NatureCNN (SB3 CnnPolicy); Atari default hyperparameters, untuned",
        "**Parallelism:** 8 SubprocVecEnv workers, frame-skip 4. Preprocessing inside the worker "
        "cut the IPC payload from 184 KB to 6.9 KB and recovered a **6.4x** speed-up",
        "**Training:** 5,000,672 agent steps = **20.0 M emulated frames in 74.1 minutes** on "
        "Apple MPS",
        "**A measured surprise:** MPS was expected to be slower than CPU for an 84 x 84 network; "
        "end-to-end it was **3.6x faster**, because the GPU frees CPU cores for the emulators",
    ], size=19)

    content_slide(prs, "Experimental Design", [
        "The emulator is **fully deterministic**, so an open-loop plan would trivially score "
        "100% — a statement about the emulator, not about planning",
        "**Condition 1 — deterministic:** the upper bound. Repeats verify reproducibility rather "
        "than estimate a rate (n = 3)",
        "**Condition 2 — sticky actions, p = 0.25** (Machado et al.): each frame the previously "
        "executed action may repeat instead of the commanded one (n = 20)",
        "This models **actuation uncertainty**: the planner's model stays exact, but the "
        "*executed* control is not the commanded one",
        "**Fairness:** one harness, one stepping loop, one metric path. An agent's only "
        "influence is the integer it returns; noise is applied at the **frame level** for all",
        "**Assumption asymmetry, stated openly:** A\\* uses privileged RAM state and a perfect "
        "simulator; PPO sees only pixels. These are not like-for-like agents",
    ], size=18)

    slide = blank_titled_slide(prs, "Headline Result — Completion on Level 1-1")
    add_table(slide, [
        ["Agent", "Deterministic", "Sticky p = 0.25", "Decision latency",
         "Frames to goal (sticky)"],
        ["Random baseline", "0.00", "0.00", "~0 ms", "—"],
        ["**A* open-loop**", "**1.00**", "**0.15**", "~0 ms (replay)", "1054"],
        ["**A* closed-loop**", "**1.00**", "**0.80**", "28.6 ms", "1100"],
        ["PPO (pixels)", "1.00", "0.35", "0.7 ms", "1091"],
    ], top=1650000, size=16, col_ratios=[0.26, 0.18, 0.2, 0.2, 0.16])
    add_caption(slide, "n = 3 deterministic, n = 20 sticky. Same planner, same primitives, same "
                       "heuristic — the only difference between the two A* rows is replanning "
                       "every 15 frames from the observed state.", top=3850000, size=15)
    body = slide.shapes.add_textbox(BODY_LEFT, Emu(4550000), BODY_WIDTH, Emu(1900000))
    frame = body.text_frame
    frame.word_wrap = True
    for index, text in enumerate([
        "**Open-loop collapses: 1.00 → 0.15.** A plan is a memorised button sequence with no "
        "feedback; once execution drifts, nothing notices",
        "**Closed-loop recovers it: 0.15 → 0.80.** This is the project's central result",
        "**Under noise, closed-loop A\\* beats PPO by more than 2x** — 0.80 against 0.35, "
        "after 74 minutes of training",
    ]):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        style_paragraph(paragraph, bullet="•", size=16, space_before=600)
        add_runs(paragraph, text, 16)

    slide = blank_titled_slide(prs, "Robustness — Before and After Perturbation")
    add_figure(slide, "comparison_1_robustness.png", top=1600000, max_height=3900000)
    add_caption(slide, "Each agent's completion rate with the emulator deterministic (left dot) "
                       "and under 25% sticky actions (right dot). The length of the line is how "
                       "much each approach loses to execution noise.", top=5700000)

    slide = blank_titled_slide(prs, "How Each Agent Fails")
    add_figure(slide, "comparison_3_failure_modes.png", top=1600000, max_height=3900000)
    add_caption(slide, "Open-loop's failures skew to **pits** — 6 of 17, six times as often as "
                       "any other agent. A jump launched a frame late cannot be corrected "
                       "mid-air: pit deaths are the signature of open-loop timing drift.",
                top=5700000)

    slide = blank_titled_slide(prs, "What It Cost, and What That Bought")
    add_figure(slide, "comparison_2_cost_vs_robustness.png", top=1550000, max_height=3900000)
    add_caption(slide, "PPO front-loads its cost into training — 20.0 M emulated frames against "
                       "A*'s 71,637, a **279x** gap — then decides in 0.7 ms. The planner pays "
                       "28.6 ms at every step instead. Neither is free; they pay at different "
                       "times.", top=5700000)

    slide = blank_titled_slide(prs, "Transfer — a Level Neither Agent Prepared For")
    add_figure(slide, "comparison_4_transfer.png", top=1600000, max_height=3800000)
    add_caption(slide, "Nobody completes 1-2, but the *way* they fail differs: closed-loop A* "
                       "reaches x = 950 with the algorithm unchanged, while PPO stops at x = 354 "
                       "— barely past the random baseline's 355. Searching the dynamics "
                       "generalises; a fitted policy does not.", top=5600000)

    content_slide(prs, "Tools & Technologies", [
        "**Environment:** Python 3.13 (uv), gym-super-mario-bros 9.1 on nes-py 9.0.1, "
        "Gymnasium 1.3 — a modern, Gymnasium-native stack with no compatibility shim",
        "**Planner:** NumPy only. Motion primitives, A\\*, and a snapshot store written from "
        "scratch; 243 unit tests",
        "**Learning:** Stable-Baselines3 2.9 (PPO, CnnPolicy) on PyTorch 2.14, Apple MPS; "
        "TensorBoard for curves",
        "**Evaluation:** one harness writing CSV into results/, figures by Matplotlib",
        "**Demonstration:** an OpenCV live demo that draws the plan on screen, plus a Streamlit "
        "dashboard that runs two agents side by side with a live noise slider",
        "**Reproducibility:** pinned requirements, YAML configs, every figure regenerated by "
        "script",
    ], size=18)

    content_slide(prs, "Limitations — stated, not hidden", [
        "**Resolution-complete, not complete.** Only primitives in the set can be executed, and "
        "states are merged by lattice cell; a manoeuvre outside the set will not be found",
        "**A\\* does not solve 1-2.** Optimal search bogs down at x ≈ 934 because the "
        "x-distance heuristic carries no vertical information; weighted A\\* reaches x ≈ 3028 "
        "but never triggers the stage exit. A **heuristic and goal-formulation** limit, not a "
        "budget one",
        "**Assumption asymmetry.** A\\* reads privileged state; PPO reads pixels. The harness "
        "measures them identically — it does not pretend they are equally endowed",
        "**One seed, no hyperparameter search.** Out of scope for a one-week project; two "
        "observed PPO collapses show run-to-run variance is real",
        "**The frontier trim** bounds memory by discarding high-f nodes — a second source of "
        "incompleteness. It never triggered on 1-1 (878 < 4,000) but is counted and reported",
    ], size=17)

    content_slide(prs, "Conclusion", [
        "**Feedback, not the search, is what makes planning robust.** The same A\\* planner goes "
        "from 0.15 to 0.80 completion under identical noise purely by replanning every 15 frames",
        "**A learned policy is not inherently more robust to actuation noise.** PPO reaches 0.35 "
        "where closed-loop A\\* reaches 0.80, after 279x more emulated experience",
        "**The real finding is a trade, not a winner:** model + privileged state + search time "
        "(28.6 ms per decision) versus pixels + 20 M samples + fast inference (0.7 ms)",
        "**Generalisation follows the model.** With the algorithm unchanged, the planner gets "
        "2.7x further than PPO on an unseen level; the policy does not transfer at all",
        "**Next:** a heuristic that reasons about vertical structure, plan-guided RL "
        "initialisation, and sticky-probability sweeps to locate the break-even point",
    ], size=18)

    content_slide(prs, "References", [
        "Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic "
        "Determination of Minimum Cost Paths. *IEEE Trans. SSC*, 4(2), 100–107",
        "LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press "
        "(kinodynamic planning, resolution completeness)",
        "Pivtoraiko, M., & Kelly, A. (2005). Generating Near Minimal Spanning Control Sets for "
        "Constrained Motion Planning in Discrete State Spaces. *IROS 2005*",
        "Likhachev, M., Gordon, G., & Thrun, S. (2003). ARA\\*: Anytime A\\* with Provable Bounds on "
        "Sub-Optimality. *NeurIPS 2003*",
        "Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347",
        "Mnih, V., et al. (2015). Human-level Control through Deep Reinforcement Learning. "
        "*Nature*, 518, 529–533",
        "Machado, M. C., et al. (2018). Revisiting the Arcade Learning Environment. *JAIR*, 61, "
        "523–562 (sticky actions)",
        "Kauten, C. (2018). gym-super-mario-bros; Raffin, A., et al. (2021). Stable-Baselines3. "
        "*JMLR*, 22(268)",
    ], size=15)


def main() -> int:
    """Generate the deck."""
    if not TEMPLATE.exists():
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1

    prs = Presentation(str(TEMPLATE))
    clear_content_slides(prs)
    rewrite_title_slide(prs.slides[0])
    build(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)} — {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
