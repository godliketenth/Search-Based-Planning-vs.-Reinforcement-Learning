# Search-Based Planning vs. Reinforcement Learning for a Kinodynamically Constrained Agent in Super Mario Bros

Mini-project for **Robotics: Computational Motion Planning**.

Super Mario Bros is treated as a **kinodynamic motion planning problem**. Mario is a 2D agent with
gravity, momentum and limited air control, navigating static obstacles (pipes, blocks, pits) and
dynamic obstacles (enemies). Two agents solve the same levels and are compared under one harness:

1. **A\* search** over Mario's state space, using **motion primitives** as edges and **exact forward
   simulation** of the emulator for reachability and collision checking.
2. **PPO** (model-free RL, stable-baselines3) trained from pixels.

## Status

| Phase | State |
|---|---|
| 0 — Environment spike | **Done** — see [`docs/phase0-findings.md`](docs/phase0-findings.md) |
| 1 — Setup + sanity check | **Done** — see [`docs/phase1-findings.md`](docs/phase1-findings.md) |
| 2 — State extraction (C-space) | **Done** — see [`docs/phase2-findings.md`](docs/phase2-findings.md) |
| 3 — A* planner with motion primitives | **Done** — solves 1-1, see [`docs/phase3-findings.md`](docs/phase3-findings.md) |
| 4 — PPO agent | **Done** — 100% on 1-1, 0% on 1-2, see [`docs/phase4-findings.md`](docs/phase4-findings.md) |
| 5 — Evaluation harness | **Done** — see [`docs/phase5-findings.md`](docs/phase5-findings.md) |
| 5b — Live in-game demo | **Done** — `scripts/demo.py`, plan drawn on screen, live sticky toggle, mp4 recording |
| Comparison dashboard | **Done** — `scripts/dashboard.py`, two agents side by side with a live perturbation slider |
| Demo launcher | **Done** — `scripts/showcase.py`, a terminal menu over every demo, plus a two-panel OpenCV window |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and macOS/Linux. **Python 3.13 is required** — `nes-py`
9.x declares `requires_python >=3.13`.

```bash
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

No C++ compiler is needed: `nes-py` 9.x ships prebuilt `macosx_11_0_arm64` wheels. The Super Mario
Bros ROM is bundled with `gym-super-mario-bros`.

Verify the install:

```bash
.venv/bin/python -c "import gym_super_mario_bros, stable_baselines3, torch; print('ok')"
```

## How to run each phase

> Commands are listed as they are implemented; unimplemented phases are marked TODO.

```bash
# Phase 1 — random-action sanity check: benchmarks the emulator, writes a GIF + JSONL log
.venv/bin/python scripts/phase1_sanity.py                          # defaults from configs/env.yaml
.venv/bin/python scripts/phase1_sanity.py --steps 2000 --sticky 0.25
.venv/bin/python scripts/phase1_sanity.py --no-gif                 # fast, no media output

# Phase 2 — C-space extraction: verify RAM reads, write a debug-overlay GIF
.venv/bin/python scripts/phase2_inspect.py --verify
.venv/bin/python scripts/phase2_inspect.py --level 1-2 --policy reactive
.venv/bin/python scripts/phase2_inspect.py --no-gif --verify     # checks only

# Phase 3 — plan 1-1 with A*, verify by open-loop replay, draw the plan + search figure
.venv/bin/python scripts/phase3_plan.py
.venv/bin/python scripts/phase3_plan.py --weight 2.0 --no-gif    # weighted A*, faster
.venv/bin/python scripts/phase3_plan.py --goal-x 900 --no-gif    # quick partial plan

# Phase 4 — train PPO from pixels (resumable, TensorBoard)
.venv/bin/python scripts/phase4_train.py                       # 5M steps, ~1.8 h
.venv/bin/python scripts/phase4_train.py --resume              # continue after interruption
.venv/bin/python scripts/phase4_benchmark.py                   # worker scaling + MPS vs CPU
.venv/bin/python scripts/phase4_curve.py                       # learning-curve figure + CSV
tensorboard --logdir runs/

# Phase 5 — evaluation harness -> results/tables/*.csv, results/figures/*.png
TODO

# Demo launcher — a menu over everything below; prints the command it runs, then runs it
.venv/bin/python scripts/showcase.py

# Phase 5b — live demo: watch the planner plan (always records an mp4)
.venv/bin/python scripts/demo.py --agent astar_closed_loop
.venv/bin/python scripts/demo.py --agent astar_open_loop --sticky 0.25
.venv/bin/python scripts/demo.py --agent ppo --headless --frames 1200   # no window
#   space pause | n step | [ ] speed | s toggle sticky | o overlay | r reset | q quit

# Two agents side by side in one OpenCV window (same seed, s perturbs both at once)
.venv/bin/python scripts/showcase.py --pair astar_open_loop,astar_closed_loop
.venv/bin/python scripts/showcase.py --pair astar_closed_loop,ppo --sticky 0.25

# Comparison dashboard — the same comparison in the browser, with a perturbation slider
.venv/bin/streamlit run scripts/dashboard.py

# tests
.venv/bin/python -m pytest tests/ -v
```

### Submission documents

The slide deck and the two-page report are generated, not hand-assembled, so their numbers
cannot drift from `results/tables/phase5_summary_main.csv`. They reuse the institute template
(logo, red headings) from the NLP mini-project deck and need `python-pptx` / `python-docx`,
which are *not* runtime dependencies of this project — run them from a throwaway environment:

```bash
uv venv /tmp/docvenv --python 3.13
uv pip install --python /tmp/docvenv/bin/python python-pptx python-docx
/tmp/docvenv/bin/python scripts/make_slides.py    # -> deliverables/*.pptx  (21 slides)
/tmp/docvenv/bin/python scripts/make_report.py    # -> deliverables/*.docx  (2 pages)
```

## Repository layout

```
src/config.py   YAML config loading + canonical results paths
src/env/        env factories, action byte table, sticky actions, RAM map, C-space state
src/planning/   motion primitives, forward simulator, A*, heuristic, plans
src/rl/         observation wrappers, vectorised env, PPO training, policy-agent adapter
src/agents/     the shared Agent interface both agents implement
src/eval/       metrics, evaluation harness, plots
src/viz/        debug overlay (Phase 2), live demo runner (5b), comparison session + window
configs/        YAML configs (env, astar, ppo, eval)
scripts/        thin CLI entry points, one per phase
tests/          unit tests for planner components
results/        raw/ (JSONL), tables/ (CSV), figures/ (PNG)
docs/           phase findings and design notes
```

Both agents implement one `Agent` interface (`reset()`, `act(obs) -> action`) so the evaluation
harness is written once and cannot accidentally favour either.

### Two environment factories, on purpose

`src/env/factories.py` exposes two constructors, and the split is forced by the emulator rather than
by taste:

| | `make_planner_env` | `make_rl_env` |
|---|---|---|
| Wrappers | **none** | `JoypadSpace` + `TimeLimit` (+ sticky) |
| Driven by | `_frame_advance(byte)` / `_get_info()` | `step(action)` → 5-tuple |
| Actions | joypad bytes (`0x82` = right+B) | `Discrete(7)` indices |
| Used by | A* forward simulation | PPO, and the random baseline |

Gymnasium wrappers keep episode bookkeeping (elapsed steps, terminated flags) that an emulator
snapshot jump silently invalidates — stepping a wrapped env after `load_state` raises
`cannot step in a done environment`. A* jumps around the state space constantly, so it must talk to
the bare emulator. `src/env/actions.py` keeps the two action views reconciled, so action *i* for PPO
and byte *b* for the planner are provably the same button press.

## Phase 1 results

`scripts/phase1_sanity.py` measures the emulator and runs a random policy as the floor baseline.
On an M1 Max (level 1-1, seed 0):

| Quantity | Value |
|---|---|
| `_frame_advance` | 0.67 ms (**~1,500 frames/s**) |
| `dump_state` / `load_state` | 0.039 ms / 0.016 ms |
| Implied A\* edge (12-frame primitive + snapshot) | **~8.0 ms ⇒ ~125 edges/s** |
| Random policy, best `x_pos` over 2,000 steps | **595** (flag is at ~3,160) |
| Realised sticky rate at `p = 0.25` | 0.237 |

Two things worth stating in the report:

1. **The random baseline stalls at x≈594, the first pipe, on every seed tested — and never dies
   there.** It clears the first Goomba by accident but cannot clear a pipe, because doing so needs a
   *committed* multi-frame manoeuvre: jump while already moving right, and keep holding right through
   the whole arc. Mario has limited air control, so a direction reversed mid-jump cannot be undone.
   This is the kinodynamic constraint stated as an experimental observation, and it is the direct
   motivation for searching over **motion primitives** instead of over single button presses — a
   random (or 1-step greedy) controller cannot represent a committed manoeuvre at all.
2. **The measured edge cost confirms the Phase 0 budget**, so the Phase 3 design decision stands:
   closed-loop replanning must use a bounded receding horizon (~60–90 frames), because a full-level
   replan costs tens of seconds.

## Phase 2 results

`scripts/phase2_inspect.py --verify` extracts Mario's configuration from RAM and checks it
against independently observable behaviour. All six checks pass on both 1-1 and 1-2:

| Check | Result |
|---|---|
| `vx` predicts Δ`x` | mean error **0.17 px/frame** |
| `vy` predicts Δ`y` | mean error **0.23 px/frame** |
| solid tile under a grounded Mario | **171/172** frames (1-1), **1148/1148** (1-2) |
| `on_ground` ⇒ `vy == 0` | 172/172 frames |
| enemy adjacent on the death frame | **10 px** |
| tile reads non-degenerate | ✓ both levels |

The decisive one is the third: a tile grid misaligned by a single column fails it instantly,
which is why it is worth more than eyeballing the numbers. The debug overlay is the visual
counterpart — `results/figures/phase2_overlay_*.gif` draws the obstacle map, enemies, Mario's
collision box and his velocity vector over the real frame.

Three things worth carrying into the report:

1. **A one-tile-lookahead reactive controller, jumping only because the tile grid says
   something is ahead, reaches x = 701** — past the pipe at x ≈ 594 that Phase 1's random
   agent could never clear — and then dies on a Goomba. That is the exact gap motion
   primitives and search exist to close: local obstacle information is enough to beat random,
   but not enough to survive without lookahead over the dynamics.
2. **`info['y_pos']` is screen-relative and inverted**, so it cannot serve as a C-space
   coordinate; it wraps when Mario falls into a pit. We compose a level-absolute `y` instead.
3. **Death has two causes, not one.** `is_dying` is `player_state == 0x0B` *or*
   `y_viewport > 1` (a pit fall), and the second fires first. A collision checker testing only
   the former keeps expanding nodes for a Mario who is already doomed.

## Phase 3 results

**A\* solves 1-1 open-loop, within 1.5% of the admissible lower bound.**

| Quantity | Value |
|---|---|
| Plan length | **1056 frames** (bound: 1040) — ratio **1.015** |
| Nodes expanded | 590 |
| Edges simulated | 4,127 (395 fatal, 2,265 pruned as duplicates) |
| Peak frontier | 878 snapshots (~239 MB) |
| Wall clock | 53.5 s (77 edges/s) |
| Open-loop replay | **exact** |

### Weighted A\*: optimality vs. compute

| weight | nodes expanded | wall clock | plan frames | ratio to bound | guaranteed bound |
|---|---|---|---|---|---|
| **1.0** | 590 | 53.5 s | **1056** | 1.015 | 1.00 |
| 1.5 | 203 | 15.7 s | 1152 | 1.107 | 1.50 |
| 2.0 | 153 | 10.1 s | 1340 | 1.288 | 2.00 |
| 5.0 | 148 | 8.8 s | 1400 | 1.346 | 5.00 |

`w = 1.5` buys a **3.4× speed-up for a 9% longer plan** — most of optimal A\*'s cost is spent
proving that a nearly-as-good plan is not better. Realised suboptimality stays far below the
guarantee (1.35× against a permitted 5×). All four plans replay exactly: weighted A\* costs
optimality, not correctness.

### Three things worth defending in the viva

1. **Consistency is what makes the search fit in memory.** Each emulator snapshot is ~272 KB,
   so one per expanded node would need gigabytes. Because the heuristic is consistent, an
   expanded node is never reopened, so snapshots are kept only for the *frontier* and freed on
   expansion — 239 MB instead of 1.1 GB. The textbook property has a very concrete payoff.
2. **The optimal plan is almost entirely jumps** — 33 jumps, 2 runs, and zero `walk`, `wait`
   or `back_off`. Jumping is free in time, because horizontal velocity is preserved through
   the arc, so a time-minimising planner jumps constantly. The conservative primitives only
   earn their place under perturbation.
3. **9.6% of edges were fatal**, and that is the collision checker working. Enemy and pit
   avoidance is never programmed — it falls out of exact forward simulation, because the
   emulator simulates the enemies too.

## Phase 4 results

**PPO solves 1-1 to saturation — and does not transfer at all.**

| Quantity | Value |
|---|---|
| Training | **5,000,672 agent steps** = 20.0 M emulated frames, **74.1 min** |
| Throughput | 697 agent steps/s (2,789 frames/s), 8 workers on MPS |
| Completion, 1-1, greedy | **100%** (32/32 episodes) |
| Completion, 1-1, sampling | 97% |
| Completion, **1-2 zero-shot** | **0%** (0/32), mean max x **354** |
| Inference | 1.4 ms per decision |

### The compute asymmetry, measured

| | A\* (Phase 3) | PPO (Phase 4) | Ratio |
|---|---|---|---|
| Emulated frames | 71,637 | 20,002,688 | **279×** |
| Wall clock | 53.5 s | 4,446 s | **83×** |
| Run-time cost per decision | ~0 (precomputed plan) | 1.4 ms | — |
| Completion, 1-1 | 100% | 100% | — |
| Completion, 1-2 | *algorithm unchanged* | **0%** | — |

Both reach 100% on 1-1 by paying in different currencies. A\* is handed a perfect model and
privileged RAM state and spends its budget once, at plan time; change the level and it
re-plans. PPO is handed pixels and no model, spends its budget at training time, then acts in
1.4 ms; change the level and it has nothing. This is the **assumption asymmetry** the project
commits to stating rather than hiding.

### The comparison dashboard

`.venv/bin/streamlit run scripts/dashboard.py` runs two agents **side by side on the same level
from the same seed**, with a slider that changes the perturbation while they run.

Start A\* open-loop against A\* closed-loop at `sticky = 0.00` — both reach the flag. Raise the
slider to 0.25 mid-run and the fixed plan desynchronises and dies while the replanner visibly
redraws its path and continues. In one recorded run from the dashboard: open-loop died at frame
851 (x = 1417), closed-loop reached the flag at frame 1077 after 72 replans, both facing a
realised override rate of 0.24.

Tabs: **Live comparison**, **Planner internals** (replans, nodes expanded, failed replans) and
**Results** (the measured figures and the Phase 5 table, read from CSV — never recomputed).
A sidebar switch plays the recorded mp4s instead, so the demonstration does not depend on the
emulator behaving on the day.

> The two panels face the same perturbation *process*, not the same noise *realisation*. Sticky
> actions repeat the previously **executed** action, so once two agents act differently their
> override sequences necessarily diverge. Each panel reports its own realised rate.

### Three things worth defending in the viva

1. **PPO needed 1 M steps (4 M frames) before completing a single episode** — it has to
   discover that jumping is useful. A\* needed no such phase, because jumps of several
   heights are *given* in its primitive set. That gap is prior knowledge traded against
   sample cost, made visible.
2. **The learning curve collapses twice** (~2.35 M and ~4.8 M steps), from ~0.8 completion to
   ~0.15, recovering within ~200 k steps. Nothing comparable can happen to A\*: a plan does
   not degrade, because there is no policy being updated. Monotonic improvement is not a
   property RL offers.
3. **Two Phase 0 predictions were wrong, and the benchmarks say so.** MPS was expected to lose
   to CPU; it wins by **3.6× end-to-end** (the bottleneck is CPU *contention* with the eight
   emulator workers, not kernel launch overhead). `SubprocVecEnv` was expected to scale
   ~linearly; it is **IPC-bound** and caps at 2.69× until observations are reduced inside the
   worker, which recovers 6.39×.

## Phase 5 results — the central comparison

100 episodes, four agents, one harness, 26.1 min. Level 1-1:

| Agent | Deterministic | **Sticky p = 0.25** | Decision latency |
|---|---|---|---|
| Random | 0.00 | 0.00 | ~0 ms |
| **A\* open-loop** | **1.00** | **0.15** | ~0 ms (replay) |
| **A\* closed-loop** | **1.00** | **0.80** | 28.6 ms |
| PPO (pixels) | 1.00 | 0.35 | 0.7 ms |

*(n = 3 deterministic — repeats verify reproducibility, since the condition is deterministic by
construction; n = 20 sticky.)*

**Open-loop planning is brittle (1.00 → 0.15); closed-loop replanning recovers most of it
(→ 0.80).** Same planner, same primitives, same heuristic — the only difference is replanning
from the observed state every 15 frames. That gap is this project's central claim, measured.

Under perturbation **closed-loop A\* beats PPO by more than 2×** (0.80 vs 0.35), despite PPO
having trained for 74 minutes. But planning is not free at run time: 28.6 ms per decision
against PPO's 0.7 ms, a 40× gap. PPO front-loads its cost into training; the planner pays every
step.

### How each agent failed (sticky, n = 20)

| Agent | Completed | Enemy deaths | Pit deaths |
|---|---|---|---|
| A\* open-loop | 3 | 11 | **6** |
| A\* closed-loop | 16 | 3 | 1 |
| PPO | 7 | 12 | 1 |

**Open-loop's failures skew toward pits.** A pit is where committed, well-timed ballistic motion
matters most: a jump launched a frame or two late cannot be corrected mid-air and there is no
ground to recover on. Pit deaths are the signature of open-loop timing drift.

### Transfer to 1-2 (never trained on, never planned for)

| Agent | Completion | Max x |
|---|---|---|
| A\* closed-loop | 0.00 | **950** |
| PPO | 0.00 | 354 |
| Random | 0.00 | 355 |

Neither solves 1-2, but they fail differently: **closed-loop A\* gets 2.7× further than PPO**
with no retraining, because it searches the dynamics rather than replaying a fitted policy. PPO
stops barely past the random baseline — its policy does not apply to unfamiliar pixels. A\*
open-loop could not be run: no full-level plan was found for 1-2 (see "Limitations").

## Experimental design

Each agent is evaluated under **two conditions**, because the NES emulator is fully deterministic:

- **Deterministic** — an A\* plan is an open-loop action sequence that replays exactly. Establishes
  the upper bound, and exposes open-loop planning for what it is.
- **Stochastic (sticky actions, p = 0.25)** — each frame the previous action may repeat instead of the
  commanded one. Open-loop plans break; this forces **closed-loop receding-horizon replanning**.

The gap between open-loop and closed-loop A\* under sticky actions is the project's central result.
Without this second condition the comparison would be trivial (A\* replays a memorised button
sequence to 100% completion) and not a meaningful test of planning.

**Assumption asymmetry (stated explicitly, not hidden):** A\* reads privileged RAM state and uses a
perfect simulator; PPO sees only pixels. A\* pays its cost at plan time, PPO at training time. These
are not like-for-like agents, and the results table carries this caveat.

## Motion planning concepts used

| Concept | Where it appears |
|---|---|
| **Configuration space** | Mario's state is `(x, y, vx, vy, on_ground, powerup)` — not screen pixels. Implemented as `MarioState` in `src/env/state.py`; the RAM tile grid is the obstacle map. Position alone is insufficient: two Marios at the same `x` with different `vx` have completely different reachable sets. |
| **Kinodynamic constraints** | Gravity, momentum and limited air control mean Mario cannot move arbitrarily between adjacent cells; a 4/8-connected grid search would produce infeasible paths. |
| **Motion primitives** | Short action sequences (run, short hop, long jump, run-jump, wait) act as graph edges. Every edge is dynamically feasible *by construction*, which is why we search over primitives instead of grid moves. |
| **Collision checking** | Exact forward simulation in the emulator. If a primitive's simulation sets `is_dying`, that edge is invalid. |
| **Dynamic obstacles** | Because the emulator simulates enemies too, enemy avoidance falls out of forward simulation for free — no separate configuration-time-space construction needed. |
| **State lattice / discretisation** | Continuous state is bucketed into `(x, y, vx)` cells for duplicate detection. This makes the planner **resolution-complete**, not complete, and optimal only up to the primitive set and lattice resolution. |
| **Admissible heuristic** | `h(s) = (x_goal - x) / v_max`, in frames. Admissible because Mario's horizontal speed is hard-capped at 3 px/frame — measured in Phase 2 and tested against the real game in `tests/test_heuristic.py`. Also **consistent** (1-Lipschitz in `x`), which guarantees A\* never re-expands a node — and therefore lets each node's 272 KB snapshot be freed on expansion. Adding any risk penalty to the cost breaks admissibility unless `h` is rescaled. |
| **Graph search** | A\* over the primitive lattice, with `g` = elapsed frames. |
| **Open-loop vs. closed-loop control** | Open-loop = execute the plan blind. Closed-loop = receding-horizon replanning from observed state. The two conditions above measure the difference directly. |
| **Model-based vs. model-free** | A\* needs a correct world model but no training; PPO needs no model but pays in sample complexity and does not transfer to unseen levels. |

## Scope

**In:** World 1-1 for both agents; both evaluation conditions; PPO with checkpoint/resume; full
metrics and plots; unit tests; live demo; zero-shot evaluation of the trained PPO agent on 1-2
(no training) and an unchanged A\* run on 1-2.

**Out (time-boxed one-week project):** DQN, the hybrid A\*-shaped-reward experiment, sticky-probability
sweeps, training on 1-2, multi-seed runs.

One expectation set in advance: PPO on 1-1 typically needs a few million steps and **has high variance
between seeds**. With one seed, it may clear the level or may plateau. Either outcome is reportable —
sample-inefficiency and seed-variance versus a planner that solves the level in one shot is precisely
the comparison being made.
