# Start Here

Everything is built, measured and tested. This page is the fast on-ramp: what to
run, what to show, and what to say. For the full reference see `README.md`; for
the measurements behind every claim see `docs/phase*-findings.md`; the submission
deck and report are in `deliverables/`.

---

## 1. Check it works (30 seconds)

```bash
cd ~/Documents/Study/RCMP/Project
.venv/bin/python -m pytest tests/ -q
```

You should see `260 passed`. Then open the launcher and press `6`:

```bash
.venv/bin/python scripts/showcase.py
```

Two Marios appear side by side on 1-1. Press `s`. The left one dies within a few
seconds; the right one keeps going. That is the whole project in one keypress.

> If `.venv` is missing, rebuild it — **Python 3.13 is required**, `nes-py` 9.x
> declares `requires_python >=3.13`:
> ```bash
> uv venv --python 3.13 .venv
> uv pip install --python .venv/bin/python -r requirements.txt
> ```
> No compiler needed; the ROM ships with `gym-super-mario-bros`.

---

## 2. The three commands you actually need

```bash
# the menu over every demo — prints the command it runs, then runs it
.venv/bin/python scripts/showcase.py

# the headline, directly: two agents, one window, same seed
.venv/bin/python scripts/showcase.py --pair astar_open_loop,astar_closed_loop

# the same comparison in a browser, with a perturbation slider
.venv/bin/streamlit run scripts/dashboard.py
```

Keys in every window: `space` pause · `n` step · `[` `]` speed · **`s` sticky
actions** · `o` overlay layers · `r` restart · `q` quit. Every run records an mp4
into `media/` automatically, so the demo exists even if the display does not.

---

## 3. Live demo script

Run these in order. It takes about six minutes and tells a complete story.

**Step 1 — "Mario is a motion planning problem."** Launch the planner alone and
let it run clean:

```bash
.venv/bin/python scripts/showcase.py        # then 1
```

Point at the dotted cyan path drawn ahead of Mario: that is the plan, in
configuration space, flashing each time it is recomputed. Press `o` twice to
strip the overlay down to the plan alone. The state is
`(x, y, vx, vy, on_ground, powerup)` — **velocity is part of the state**, which
is what makes this kinodynamic rather than a maze.

**Step 2 — "A plan is not a controller."** Run the open-loop agent, and once it
is moving press `s` to switch sticky actions on at p = 0.25:

```bash
.venv/bin/python scripts/showcase.py        # then 2, then press s
```

The plan keeps drawing and Mario stops matching it. Nothing notices, because
nothing is looking. **Open-loop: 1.00 → 0.15.**

**Step 3 — "Here is the fix, and it is one line of control theory."** The
headline:

```bash
.venv/bin/python scripts/showcase.py --pair astar_open_loop,astar_closed_loop
```

Press `s`. Same planner, same primitives, same heuristic on both sides. The only
difference is that the right panel replans from the *observed* state every 15
frames over a 160 px horizon. Watch its `replans` counter climb while the left
side stays at 1. **0.15 → 0.80.** That is the project's contribution.

**Step 4 — "And learning?"** Against the agent that trained for 74 minutes:

```bash
.venv/bin/python scripts/showcase.py --pair astar_closed_loop,ppo
```

Press `s`. PPO holds 0.35 — better than open-loop, less than half of closed-loop.
A learned policy is not automatically robust to actuation noise.

**Step 5 — "Here are the numbers."** In the menu, `t` prints the completion table
straight from `results/tables/phase5_summary_main.csv`, and `f` opens the four
comparison figures. Nothing in the report, the deck or this page is typed by
hand; it all comes out of that CSV.

**Step 6 — "And it does not transfer."** In the menu press `l`, choose `1-2`,
then `1`. Neither method finishes a level it never prepared for: the planner gets
950 px before its node budget runs out, PPO stalls at 354 — barely past random's
355. Note that option 2 greys itself out here, because no plan was ever cached
for 1-2.

---

## 4. The numbers to memorise

| Claim | Number |
|---|---|
| Open-loop A\* on 1-1, deterministic → sticky p = 0.25 | **1.00 → 0.15** |
| Closed-loop A\* on the same perturbation | **0.80** |
| PPO on the same perturbation | **0.35** |
| Decision latency: closed-loop A\* vs PPO | **28.6 ms vs 0.7 ms** (40×) |
| Training cost: PPO vs the single A\* search | **20.0 M frames vs 71,637** (279×) |
| PPO wall clock | **74.1 min**, 5,000,672 steps, 8 workers on MPS |
| Peak planner memory, frontier-only snapshots | **239 MB**, not 1.1 GB |
| Robustness costs optimality | **1054 → 1100 frames** to the flag (≈4%) |
| Episodes evaluated | **100**, four agents, one harness |
| Tests | **260 / 260** |

---

## 5. The things to be able to say

**The pitch.** "I framed Super Mario Bros as a kinodynamic motion planning
problem — the state carries velocity, the edges are dynamically feasible motion
primitives, and collision checking is exact forward simulation in the emulator —
then compared an A\* planner against a PPO agent on the same levels, under the
same perturbation, through one harness."

**The one-line comparison.** "A\* buys robustness with run-time compute; PPO buys
run-time speed with training compute. Each pays for the same thing at a different
time."

**The central result.** "Open-loop A\* and closed-loop A\* are the *same planner*.
Same primitives, same heuristic, same code. The only difference is that one
replans from the observed state every 15 frames. Under sticky actions that single
change takes completion from 0.15 to 0.80 — and beats a policy trained for 74
minutes by more than 2×. The contribution is not the search; it is closing the
loop around it."

**Why the perturbation is necessary, if asked.** "The emulator is deterministic,
so an open-loop plan would trivially score 100% and the comparison would be
meaningless. Sticky actions at p = 0.25 (Machado et al.) repeat the previously
executed action at the frame level — actuation uncertainty, not sensing noise.
It is the standard protocol, and it is what makes open-loop fail honestly."

**The asymmetry, before they point it out.** "A\* uses privileged RAM state and a
perfect simulator; PPO uses 84×84 pixels only. This is not like-for-like, and I
say so in the report. What it measures is what *exact models plus search* buy you
over *experience*, which is the actual question."

**If they ask what was hard.** Snapshot *size*, which Phase 0 never measured.
Each emulator snapshot is ~272 KB, so storing one per expanded node would have
been 2.7 GB for a 10,000-node search — the planner would not have run. The fix
came from the theory rather than from engineering: the heuristic
`h(s) = (x_goal − x) / 3` is not just admissible but **consistent**, so an
expanded node is never reopened, so its snapshot can be dropped the moment it
leaves the open list. Memory then scales with the *frontier*, not with nodes
expanded: 878 snapshots, 239 MB. *This is the strongest thing to talk about — it
is a property of the heuristic doing real work, not a trick.*

**A second one, if they want another.** `x_pos` freezes during Mario's death
animation, so an early test "passed" by mistaking a frozen position for a valid
state. The real death signal is `info['is_dying']` / `player_state == 11` on the
contact frame. Validating the instrument before trusting the measurement.

---

## 6. Where everything is

```
start.md                 this page
README.md                full reference, per-phase run commands, concepts used
docs/phase*-findings.md  the measurements, phase by phase
deliverables/            the 21-slide deck and the 2-page report (+ PDFs)

scripts/showcase.py      the demo launcher — the only command you need live
scripts/demo.py          one agent, window + mp4, plan drawn on screen
scripts/dashboard.py     the browser comparison (live, planner internals, results)
scripts/phase3_plan.py   plans 1-1 with A*; scripts/phase4_train.py trains PPO
scripts/phase5_eval.py   the evaluation harness -> results/

src/planning/            motion primitives, forward simulator, A*, heuristic
src/rl/                  observation wrappers, vectorised env, PPO training
src/viz/                 debug overlay, live demo runner, comparison + window
results/                 tables/*.csv and figures/*.png — never hand-assembled
media/                   recorded mp4s of every demo run
tests/                   260 tests
```

Regenerating the submission documents (needs `python-pptx` / `python-docx`,
deliberately not project dependencies — see `README.md`):

```bash
uv venv /tmp/docvenv --python 3.13
uv pip install --python /tmp/docvenv/bin/python python-pptx python-docx
/tmp/docvenv/bin/python scripts/make_slides.py
/tmp/docvenv/bin/python scripts/make_report.py
```
