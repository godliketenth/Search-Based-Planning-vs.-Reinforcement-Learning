# Project context pack — for generating a presentation

**Purpose of this file.** A single self-contained briefing on the project, written so that an
assistant with *only this file* can build an accurate slide deck. Every number here was
measured on the actual system; nothing is illustrative or invented.

**Project status:** Complete. Phases 0–5b all built, tested (224 tests) and measured.

---

## 1. One-paragraph summary

Super Mario Bros is reframed as a **kinodynamic motion planning problem** and used to compare
two fundamentally different ways of making an agent act: **search-based planning** (A* over
motion primitives, using the emulator itself as an exact forward model) and **reinforcement
learning** (PPO trained from raw pixels). Both are evaluated on the same levels, through one
harness, under two conditions — a deterministic one and one with **sticky actions** that
perturb execution. The headline finding is not which agent wins: both complete level 1-1.
It is *what each one buys and what each one costs* — A* spends 54 seconds of search and
generalises to any level instantly; PPO spends 74 minutes and 20 million frames of training
and does not transfer at all.

**Course:** Robotics: Computational Motion Planning (3rd-year BTech AIML mini-project).
**Title:** *Search-Based Planning vs. Reinforcement Learning for a Kinodynamically
Constrained Agent in Super Mario Bros.*

---

## 2. The thesis, in three claims

1. **Mario is a genuine kinodynamic planning problem, not a maze.** Gravity acts every frame,
   momentum carries, and air control is limited. A path through a grid of tiles is not
   something Mario can execute. *Evidence: a random policy stalls permanently at the first
   pipe (x≈594) on every seed — clearing it needs a committed multi-frame manoeuvre that a
   per-frame random controller cannot represent.*
2. **Planning and learning pay in different currencies.** A* is given a perfect model and
   privileged state and pays at plan time; PPO is given pixels and pays at training time.
   *Evidence: 279× difference in emulated frames, 83× in wall clock, for the same 100%
   completion on 1-1 — and PPO scores 0% on an unseen level where A*'s algorithm is unchanged.*
3. **Open-loop planning is brittle; closed-loop planning is the real contribution.** On a
   deterministic emulator an A* plan replays perfectly, which proves nothing. Under
   perturbation it degrades sharply, and receding-horizon replanning recovers most of it.
   *Evidence: under sticky actions, open-loop A\* falls from **1.00 to 0.15**; closed-loop
   replanning recovers it to **0.80** — and beats PPO's 0.35 on the same perturbed task.*

---

## 3. Suggested slide outline (~16 slides)

| # | Slide | Core content |
|---|---|---|
| 1 | Title | Title, name, course |
| 2 | Why Mario is a motion planning problem | C-space, kinodynamic constraints, obstacles |
| 3 | The random baseline | Stuck at the first pipe; motivates primitives |
| 4 | Problem formulation | State, controls, cost, goal |
| 5 | Configuration space | `(x, y, vx, vy, on_ground, powerup)`; why velocity matters |
| 6 | Obstacle map from RAM | Tile grid extraction + debug overlay figure |
| 7 | Motion primitives | Why primitives, not grid moves; the primitive set |
| 8 | Exact forward simulation | Snapshots; collision checking is the simulation |
| 9 | A* and the heuristic | Admissibility, consistency, what consistency buys |
| 10 | A* results on 1-1 | 1056 frames vs 1040 bound; search figure |
| 11 | Weighted A* | Optimality vs compute trade-off table |
| 12 | PPO setup | Pixels, NatureCNN, wrappers, parallelism |
| 13 | PPO learning curve | Curve figure; the flat first million steps |
| 14 | Experimental design | Two conditions; why sticky actions are necessary |
| 15 | **Headline results** | Completion table + open-loop vs closed-loop gap |
| 16 | Compute asymmetry & transfer | 279×; PPO 0% on 1-2 |
| 17 | Limitations & honest findings | Resolution-completeness, A* fails on 1-2, one seed |

---

## 4. Problem formulation (use verbatim on a slide)

- **Configuration:** `q = (x, y, vx, vy, on_ground, powerup)`. Position *and* velocity, because
  two Marios at the same `x` with different `vx` have completely different reachable sets.
- **Controls:** NES joypad bytes from the `SIMPLE_MOVEMENT` set (7 actions).
- **Edges:** motion primitives — short fixed button sequences, so every edge is **dynamically
  feasible by construction**.
- **Edge cost:** duration in frames. So `g` is elapsed time and the planner minimises
  time-to-goal.
- **Goal test:** the emulator's own `flag_get`.
- **Collision check:** forward-simulate; the edge is invalid iff Mario died on it.

---

## 5. Motion planning concepts → where each appears

| Concept | Where it appears in this project |
|---|---|
| **Configuration space** | `MarioState` = `(x, y, vx, vy, on_ground, powerup)`, extracted from RAM, not pixels |
| **Kinodynamic constraints** | Gravity, momentum, limited air control; `on_ground` gates which controls are even available |
| **Motion primitives** | 7 primitives; jump variants differ by **how long A is held**, which is what sets jump height |
| **C-obstacle / obstacle map** | 13×16 tile grid read from a 2-page ring buffer in RAM |
| **Collision checking** | Exact forward simulation — no geometric approximation, zero model error |
| **Dynamic obstacles** | Enemies need no special handling: the emulator simulates them, so a fatal primitive simply dies |
| **State lattice** | States bucketed to `(x//8, y//8, vx//8, on_ground, powerup)` for duplicate detection |
| **Admissible heuristic** | `h(s) = (x_goal − x) / v_max`, in frames; `v_max` = 3 px/frame is a hard game cap |
| **Consistency** | `h` is 1-Lipschitz in `x`; guarantees no re-expansion — and that is what makes the search fit in memory |
| **Resolution completeness** | Complete only up to the primitive set and lattice resolution — stated, not glossed over |
| **Weighted A\*** | Bounded suboptimality: cost ≤ `w` × optimal |
| **Open-loop vs closed-loop** | The two A* agents; the gap under perturbation is the central result |
| **Receding horizon** | Closed-loop replans 160 px ahead every 15 frames — a full replan costs 54 s and is unaffordable |
| **Actuation uncertainty** | Sticky actions: the model is exact but the *executed* control differs from the commanded one |

---

## 6. Method A — A* over motion primitives

**The primitive set** (7 edges; jump height is set by A-hold duration):

| Name | Buttons | Frames | A held |
|---|---|---|---|
| `run` | right+B | 12 | — |
| `walk` | right | 12 | — |
| `hop` | right+A+B → right+B | 20 | 4 |
| `jump` | right+A+B → right+B | 28 | 12 |
| `long_jump` | right+A+B → right+B | 40 | 24 |
| `wait` | NOOP | 8 | — |
| `back_off` | left | 8 | — |

**Exact forward simulation.** Expanding an edge is: `load_state(snapshot)` (0.016 ms) → run the
primitive's bytes → `dump_state()` (0.039 ms). The emulator *is* the transition function, so
there is **zero modelling error** and collision checking is exact rather than conservative.

**The memory constraint — a good slide.** Each emulator snapshot is **272 KB**. Storing one per
expanded node would need gigabytes. Because the heuristic is **consistent**, an expanded node is
never reopened, so snapshots are kept only for the **frontier** and freed on expansion:
**239 MB instead of ~1.1 GB**. This is a textbook property with a very concrete payoff —
consistency is usually taught as "it's faster"; here it is what makes the search fit in memory.

---

## 7. Method B — PPO from pixels

- **Observation:** 84×84 grayscale, 4 frames stacked (stacking matters: a single frame cannot
  show whether Mario is rising or falling — the same reason the planner's C-space needs velocity).
- **Network:** NatureCNN (SB3 `CnnPolicy`), hyperparameters = SB3 Atari defaults, untuned.
- **Parallelism:** 8 `SubprocVecEnv` workers; frame-skip 4; frame stacking at the vector level.
- **Reward:** the environment's built-in dense x-progress reward. No shaping.
- **Training:** 5,000,672 agent steps = 20.0 M emulated frames in 74.1 minutes on Apple MPS.

---

## 8. Experimental design

Two conditions, because the emulator is fully deterministic:

1. **Deterministic** — the upper bound. An open-loop plan replays exactly. Repeats here
   *verify* reproducibility rather than estimate a rate.
2. **Sticky actions, p = 0.25** — each frame the previously executed action may repeat instead
   of the commanded one (Machado et al.). This models **actuation uncertainty**: the planner's
   model is still exact, but the executed control is not the commanded one.

**Why condition 2 is necessary (important for the viva).** Without it, "A* completes the level
100% of the time" is a statement about the emulator's determinism, not about planning. The
comparison would be trivial.

**Fairness.** All four agents go through one harness: same env construction, same stepping loop,
same frame accounting, same metrics. An agent's only influence is the integer it returns.
Sticky actions are applied at the **frame level** for both agents, deliberately departing from
the ALE convention of perturbing once per agent step, so both face an identical process.

**Assumption asymmetry — state this explicitly on a slide.** A* reads privileged RAM state and
uses a perfect simulator; PPO sees only pixels. A* pays at plan time, PPO at training time.
These are **not like-for-like agents**. The harness measures them identically; it does not
pretend they are equally endowed.

---

## 9. Results

### 9.1 Random baseline (Phase 1)
- Stalls at **x ≈ 594**, the first pipe, on every seed; often does not even die there.
- Reason: clearing a pipe needs a *committed* multi-frame manoeuvre — jump while already moving
  right and hold right through the arc. Limited air control means a mid-jump reversal cannot be
  undone. **This is the motivation for motion primitives, as an experimental observation.**

### 9.2 State extraction (Phase 2) — all 6 checks pass on 1-1 and 1-2

| Check | Result |
|---|---|
| `vx` predicts Δx | mean error **0.17 px/frame** |
| `vy` predicts Δy | mean error **0.23 px/frame** |
| Solid tile under a grounded Mario | **171/172** (1-1), **1148/1148** (1-2) |
| `on_ground` ⇒ `vy == 0` | 172/172 |
| Enemy adjacent on the death frame | **10 px** |

A one-tile-lookahead reactive controller, jumping only because the tile grid says something is
ahead, reaches **x = 701** — past the pipe random could never clear — then dies on a Goomba.
Local obstacle information beats random; surviving needs lookahead over the dynamics.

### 9.3 A* on 1-1 (Phase 3)

| Quantity | Value |
|---|---|
| Plan length | **1056 frames** |
| Admissible lower bound | 1040 frames |
| **Ratio to bound** | **1.015** |
| Nodes expanded | 590 |
| Edges simulated | 4,127 (395 fatal = 9.6%, 2,265 pruned) |
| Frames emulated | 71,637 |
| Peak frontier | 878 snapshots (~239 MB) |
| Wall clock | **53.5 s** |
| Open-loop replay | **exact** |

**The optimal plan is almost entirely jumps:** 15 `long_jump`, 9 `jump`, 9 `hop`, 2 `run`, and
**zero** `walk`/`wait`/`back_off`. Jumping is free in time — horizontal velocity carries through
the arc — so a time-minimising planner jumps constantly.

**9.6% of edges were fatal**, and that is the collision checker working. Enemy and pit avoidance
is never programmed; it falls out of exact forward simulation.

### 9.4 Weighted A* — optimality vs compute

| weight | Nodes | Wall clock | Plan frames | Ratio to bound | Guaranteed bound |
|---|---|---|---|---|---|
| **1.0** | 590 | 53.5 s | **1056** | 1.015 | 1.00 |
| 1.5 | 203 | 15.7 s | 1152 | 1.107 | 1.50 |
| 2.0 | 153 | 10.1 s | 1340 | 1.288 | 2.00 |
| 5.0 | 148 | 8.8 s | 1400 | 1.346 | 5.00 |

Two points worth making: `w = 1.5` buys a **3.4× speed-up for a 9% longer plan** — most of
optimal A*'s cost is spent *proving* a nearly-as-good plan is not better. And realised
suboptimality stays far below the guarantee (1.35× against a permitted 5×).

### 9.5 PPO (Phase 4)

| Quantity | Value |
|---|---|
| Training | 5,000,672 steps = 20.0 M frames, **74.1 min** |
| Throughput | 697 agent steps/s, 8 workers on MPS |
| Completion on 1-1 (greedy) | **100%** (32/32) |
| Completion on 1-1 (sampling) | 97% |
| **Completion on 1-2 (zero-shot)** | **0%** (0/32), mean max x = **354** |
| Inference | 1.4 ms per decision |

**Learning curve shape** (figure available):

| Steps | Completion |
|---|---|
| 0 – 1.0 M | **0.00** |
| 1.7 M | 0.42 |
| 2.2 M | 0.80 |
| 3.6 M – 5.0 M | **0.95 – 1.00** |

Three features worth a slide each:
- **A flat first million steps** (4 M frames) before a single completion — PPO must *discover*
  that jumping is useful. A* needed no such phase: jumps are *given* in its primitive set. This
  is prior knowledge traded against sample cost, made visible.
- **Two sharp collapses** (~2.35 M and ~4.8 M steps) from ~0.8 to ~0.15, recovering in ~200 k
  steps. Nothing comparable can happen to A*: a plan does not degrade, because no policy is
  being updated. **Monotonic improvement is not a property RL offers.**
- **An apparent plateau is weak evidence of convergence.** At 1.9 M steps completion had sat at
  0.34–0.43 for 50 k steps and looked converged; it reached 0.80 within another 250 k steps.

### 9.6 Head-to-head: the compute asymmetry

| | A* | PPO | Ratio |
|---|---|---|---|
| Emulated frames | 71,637 | 20,002,688 | **279×** |
| Wall clock | 53.5 s | 4,446 s | **83×** |
| Cost per decision at run time | ~0 (precomputed) | 1.4 ms | — |
| Completion on 1-1 | 100% | 100% | — |
| **Completion on 1-2** | *algorithm unchanged* | **0%** | — |

### 9.7 Final evaluation (Phase 5) — THE HEADLINE RESULT

100 episodes, four agents, one harness, 26.1 min of compute. Level 1-1:

| Agent | Deterministic | **Sticky p = 0.25** | Decision latency |
|---|---|---|---|
| Random | 0.00 | 0.00 | ~0 ms |
| **A\* open-loop** | **1.00** | **0.15** | ~0 ms (replay) |
| **A\* closed-loop** | **1.00** | **0.80** | 28.6 ms |
| PPO (pixels) | 1.00 | 0.35 | 0.7 ms |

*(n = 3 deterministic — repeats verify reproducibility, since the condition is deterministic by
construction; n = 20 sticky.)*

**Three things this table says, in order of importance:**

1. **Open-loop planning is brittle: 1.00 → 0.15.** A plan is a memorised button sequence with no
   feedback; once execution drifts, nothing notices.
2. **Closed-loop replanning recovers most of it: 0.15 → 0.80.** Same planner, same primitives,
   same heuristic — the only difference is replanning from the observed state every 15 frames.
   **This gap is the project's central contribution.**
3. **Under perturbation, closed-loop A\* (0.80) beats PPO (0.35) by more than 2×** — despite PPO
   having been trained for 74 minutes and A* planning in milliseconds per step. A learned policy
   is not inherently more robust to actuation noise than an explicit replanner.

**But planning is not free at run time.** Closed-loop A* costs **28.6 ms per decision** against
PPO's **0.7 ms** — a 40× gap. PPO front-loads its cost into training; the planner pays every
step. That is the same trade as §9.6, seen from the other end.

#### How each agent failed (sticky, 1-1, n = 20)

| Agent | Completed | Enemy deaths | Pit deaths |
|---|---|---|---|
| A\* open-loop | 3 | 11 | **6** |
| A\* closed-loop | 16 | 3 | 1 |
| PPO | 7 | 12 | **1** |
| Random | 0 | 20 | 0 |

**Open-loop's failures skew toward pits (6/17) — closed-loop and PPO almost never fall in one.**
A pit is where committed, well-timed ballistic motion matters most: a jump launched a frame or
two late cannot be corrected mid-air, and there is no ground to recover on. Enemy deaths are
survivable mistakes; pit deaths are the signature of open-loop timing drift.

#### Mean frames to goal (completed episodes)

| Agent | Frames |
|---|---|
| A\* open-loop | **1054** (the optimal plan) |
| PPO | 1080 |
| A\* closed-loop | 1100 |

The planner's open-loop plan is the fastest trajectory, as expected — it was optimised for exactly
that. Closed-loop is ~4% slower because replanning repeatedly commits to a fresh 160 px horizon
rather than one globally optimal route. **Robustness costs a little optimality.**

#### Transfer test on 1-2 (never trained on, never planned for)

| Agent | Completion | Max x reached |
|---|---|---|
| A\* closed-loop | 0.00 | **950** |
| PPO | 0.00 | 354 |
| Random | 0.00 | 355 |
| A\* open-loop | *not run* | — (no plan could be found; see §10.2) |

Neither approach solves 1-2, but they fail differently and the difference is informative.
**Closed-loop A\* gets 2.7× further than PPO** with no retraining and no modification, because it
is searching the dynamics rather than replaying a fitted policy. PPO stops at x = 354, barely past
the random baseline's 355 — its policy simply does not apply to unfamiliar pixels. A*'s run cost
69.9 ms/decision there, because it fails to find a plan and burns its full search budget on most
of its 535 replans.

## 10. Limitations — state these, do not hide them

1. **Resolution-complete, not complete.** Only primitives in the set can be executed, and states
   are merged by lattice cell, so a solution needing a manoeuvre outside the set — or
   distinguishable only below the lattice resolution — will not be found.
2. **A\* does not solve 1-2.** Optimal A* bogs down at x ≈ 934 (the x-distance heuristic
   degenerates toward breadth-first on a vertically structured level); weighted A* reaches
   x ≈ 3028, essentially the end, but never triggers stage completion, because 1-2's exit works
   differently from 1-1's flagpole. This is a limitation of the **heuristic and goal
   formulation**, not a budget shortfall.
3. **Assumption asymmetry** (section 8) — the agents are not equally endowed.
4. **One seed, no hyperparameter search.** Out of scope for a one-week project; the two PPO
   collapses are a reminder that run-to-run variance is real.
5. **The frontier trim** bounds memory by discarding high-`f` nodes, a second source of
   incompleteness. It did not trigger on 1-1 (878 < 4,000) but is counted and reported.

---

## 11. Engineering findings worth one slide

Measured results that contradicted the project's own initial assumptions — good material for a
"what we learned" slide:

| Assumption | Measured reality |
|---|---|
| MPS would be *slower* than CPU for an 84×84 NatureCNN | MPS is **3.6× faster end-to-end** (33.6× on the isolated network). The bottleneck was CPU *contention* with 8 emulator workers, not kernel launch overhead |
| `SubprocVecEnv` scales ~linearly to 8 workers | It is **IPC-bound**: capped at 2.69× while shipping 184 KB RGB frames. Reducing to 84×84 grayscale *inside* the worker recovers **6.39×** |
| More workers is better | 10 and 12 workers are **slower** than 8 — the main process needs a core for the gradient step |
| `ram_values()` is the fast batched RAM read | It is **80× slower** than numpy fancy-indexing (14.4 µs vs 0.17 µs) |

Two harness bugs that produced plausible but wrong behaviour without raising an error:
- **Frame-stack spacing.** PPO's 4-frame stack is 4 game frames apart during training; the
  harness initially pushed every frame, giving a quarter of the expected motion. PPO appeared to
  fail at 0% until this was found by cross-checking against the Phase 4 number.
- **Replan backoff.** A failed replan immediately retried instead of respecting the interval, so
  on a level the planner cannot solve, one episode cost hours instead of minutes.

---

## 12. Phase 5b — live in-game demo (BUILT)

`scripts/demo.py` makes planning *observable* instead of an invisible internal detail.

- **OpenCV window** showing the game with the planner's trajectory drawn on it: a dotted path
  ahead of Mario, with a yellow border flash on every replan.
- **Live sticky-action toggle (`s`)** — the demo's most valuable control. Pressing it mid-run
  reproduces the headline result in front of the viewer: the open-loop plan desynchronises and
  Mario dies, while the closed-loop planner visibly redraws its path and recovers. That is 0.15
  versus 0.80, shown rather than tabulated.
- **HUD:** agent, current action, `x`, `y`, velocity, on-ground, enemies, sticky rate, playback
  speed, replan count, nodes expanded.
- **Controls:** `space` pause · `n` step one frame · `[` `]` slow-mo / fast-forward ·
  `s` sticky · `o` cycle overlay layers · `r` reset · `q` quit.
- **Always records an mp4**, and `--headless` produces one with no window at all — so the
  deliverable never depends on a GUI working on the day.
- Cached plans load instantly (a live full-level search takes 54 s and would freeze the window).
- `--agent human` exists but is **best-effort**: `cv2.waitKey` has no key-up events, so holding
  a button cannot be detected. It latches instead — a press persists until the next one. A human
  score collected this way is a rough floor, not a fair measure of human skill.

**Recorded demos** (in `media/`, ~1,100 frames each):
`demo_astar_open_loop_1-1.mp4` · `demo_astar_closed_loop_1-1.mp4` (sticky on, 70 replan
flashes) · `demo_ppo_1-1.mp4`.

---

## 13. Likely viva questions and answers

**Why not just A* on a grid?** Mario cannot move between adjacent cells at will — gravity and
momentum act every frame and air control is limited, so a grid path is dynamically infeasible.
The random baseline demonstrates this empirically: it never clears the first pipe.

**Why is the heuristic admissible?** `h = (x_goal − x)/v_max` with `v_max` = 3 px/frame, a hard
cap the game imposes, measured directly rather than assumed. No trajectory can close distance
faster, so `h` never overestimates.

**Why does consistency matter here specifically?** Beyond preventing re-expansion, it is what
lets each expanded node's 272 KB snapshot be freed immediately — the search fits in 239 MB
instead of gigabytes.

**Isn't 100% completion for A* trivial?** Yes, under determinism — which is exactly why the
sticky-action condition exists. The meaningful number is how much of that survives perturbation
and how much replanning recovers.

**Why does PPO fail on 1-2?** It learned *this level*, not *Mario*. It has no model of the
dynamics to transfer; the policy is a mapping from pixels to buttons fitted to one level's
pixel distribution.

**What would you do with more time?** A heuristic that accounts for vertical structure (so A*
handles 1-2), multi-seed runs, and a hybrid where the planner's primitives seed RL exploration.

---

## 14. Available figures

| Path | Shows |
|---|---|
| `results/figures/phase1_random_1-1_sticky0.00_seed0.gif` | Random agent stuck at the first pipe |
| `results/figures/phase2_overlay_1-1_reactive_seed0.gif` | Debug overlay: tile grid, enemies, Mario's box, velocity vector |
| `results/figures/phase2_overlay_1-2_reactive_seed0.gif` | Same, underground tileset |
| `results/figures/phase3_search_1-1_w1.0_seed0.png` | **A\* expanded node set over the level** — dense clusters at obstacles |
| `results/figures/phase3_plan_1-1_w1.0_seed0.gif` | The plan drawn as a dotted arc ahead of Mario |
| `results/figures/phase4_curve_ppo_1-1.png` | **PPO learning curve** — flat start, rise, two collapses |
| `results/figures/phase5_completion_1-2_main.png` | Transfer test completion (all zero) |
| `results/figures/phase5_distance_1-2_main.png` | Transfer: how far each got on 1-2 |
| `results/figures/phase5_completion_1-1_main.png` | **Headline completion bars** — open-loop vs closed-loop gap |
| `results/figures/phase5_distance_1-1_main.png` | Distance reached, per agent and condition |
| `results/figures/phase5_deaths_1-1_main.png` | Failure causes — open-loop's pit deaths |
| `results/figures/phase5_compute_main.png` | Compute cost, log scale (279× gap) |
| `media/demo_astar_closed_loop_1-1.mp4` | **Live demo** — plan drawn on screen, replan flashes, sticky on |
| `media/demo_astar_open_loop_1-1.mp4` | Live demo — open-loop plan, deterministic |
| `media/demo_ppo_1-1.mp4` | Live demo — PPO completing 1-1 |

---

## 15. Numbers to quote precisely

Random ceiling **x = 594** · Flagpole **x = 3161** · A* plan **1056 frames** vs bound **1040**
(ratio **1.015**) · A* search **53.5 s**, **590 nodes**, **71,637 frames** · Snapshot **272 KB**,
peak **239 MB** · PPO **5,000,672 steps** = **20.0 M frames** in **74.1 min** · PPO **100%** on
1-1, **0%** on 1-2 · Compute ratio **279×** frames, **83×** wall clock · PPO inference **1.4 ms**,
closed-loop A* **28.6 ms** · Max speed **3 px/frame** · Emulation **~1,500 frames/s** single core,
**7,697 frames/s** on 8 workers · **Sticky p = 0.25: open-loop 0.15, closed-loop 0.80, PPO 0.35** ·
Transfer to 1-2: closed-loop A* reaches **x = 950**, PPO **x = 354** · 100 evaluation episodes in
**26.1 min**.

---

## 16. Tone guidance for the deck

- Lead with the *comparison*, not with Mario. The game is the testbed, not the subject.
- Do not claim A* "beats" PPO or vice versa. The finding is the **trade**: model + privileged
  state + search time, versus pixels + samples + fast inference.
- Keep the honest negatives in — A* failing on 1-2, PPO failing to transfer, the assumption
  asymmetry. They are the strongest evidence the work was actually done.
