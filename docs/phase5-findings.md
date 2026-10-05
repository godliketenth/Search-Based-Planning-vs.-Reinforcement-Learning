# Phase 5 & 5b — evaluation and demo findings

Date: 2026-10-05. 100 episodes, four agents, one harness, 26.1 min of compute.

## Headline: open-loop planning is brittle; closed-loop replanning is the contribution

Level 1-1, completion rate:

| Agent | Deterministic | **Sticky p = 0.25** | Decision latency | Mean frames to goal |
|---|---|---|---|---|
| Random | 0.00 | 0.00 | ~0 ms | — |
| **A\* open-loop** | **1.00** | **0.15** | ~0 ms (replay) | 1054 |
| **A\* closed-loop** | **1.00** | **0.80** | 28.6 ms | 1100 |
| PPO (pixels) | 1.00 | 0.35 | 0.7 ms | 1080 |

n = 3 deterministic (repeats *verify* reproducibility — the condition is deterministic by
construction), n = 20 sticky.

Three readings, in order of importance:

1. **Open-loop collapses: 1.00 → 0.15.** A plan is a memorised button sequence with no feedback.
   Once execution drifts, nothing notices.
2. **Closed-loop recovers it: 0.15 → 0.80.** Same planner, same primitives, same heuristic. The
   only difference is replanning from the observed state every 15 frames. **This is the
   project's central result.**
3. **Under perturbation, closed-loop A\* beats PPO by more than 2×** (0.80 vs 0.35), despite PPO
   having trained for 74 minutes and A\* planning in milliseconds per step. A learned policy is
   not inherently more robust to actuation noise than an explicit replanner.

**But planning is not free at run time.** 28.6 ms per decision against PPO's 0.7 ms — a 40×
gap. PPO front-loads its cost into training; the planner pays it every step. This is the same
trade as the Phase 4 compute asymmetry, seen from the other end.

**Robustness costs a little optimality.** Open-loop's completed runs take 1054 frames (the
optimal plan); closed-loop takes 1100, about 4% slower, because it repeatedly commits to a fresh
160 px horizon rather than following one globally optimal route.

## How each agent fails

Sticky condition, n = 20:

| Agent | Completed | Enemy deaths | Pit deaths |
|---|---|---|---|
| A\* open-loop | 3 | 11 | **6** |
| A\* closed-loop | 16 | 3 | 1 |
| PPO | 7 | 12 | 1 |
| Random | 0 | 20 | 0 |

**Open-loop's failures skew toward pits (6 of 17); the others almost never fall in one.** A pit
is where committed, well-timed ballistic motion matters most: a jump launched a frame or two
late cannot be corrected mid-air, and there is no ground to recover on. Enemy deaths are
survivable mistakes; **pit deaths are the signature of open-loop timing drift**. Recording the
cause of failure, not just the rate, is what makes that visible.

## Transfer to 1-2

Never trained on, never planned for:

| Agent | Completion | Max x reached |
|---|---|---|
| A\* closed-loop | 0.00 | **950** |
| PPO | 0.00 | 354 |
| Random | 0.00 | 355 |
| A\* open-loop | *not run* | — (no full-level plan found) |

Neither approach solves 1-2, but the *way* they fail differs. **Closed-loop A\* gets 2.7×
further than PPO** with no retraining and no modification, because it searches the dynamics
rather than replaying a fitted policy. PPO stops at x = 354, barely past the random baseline's
355 — its policy simply does not apply to unfamiliar pixels. A\*'s cost there was 69.9
ms/decision, because it fails to find a plan and burns its full budget on most of its 535
replans.

## A result I predicted wrongly

I expected open-loop A\* to score ≈ 0% under sticky actions. It scores **0.15**, and the reason
is worth reporting: the optimal plan is long runs of *held* buttons (35 primitives over 1056
frames), so a frame-level override usually repeats the action that was already commanded and
does nothing. Only the ~34 primitive boundaries are genuinely vulnerable. The perturbation is
weaker against this kind of plan than against a plan that switches buttons constantly.

## Closed-loop tuning: replan frequency

5 episodes per setting, sticky 0.25:

| `replan_every` | `horizon_px` | Completed | Mean x | Replans | ms/decision |
|---|---|---|---|---|---|
| **15** | **160** | 4/5 | **3080** | 74 | 29.3 |
| 30 | 160 | 4/5 | 2919 | 34 | 13.9 |
| 15 | 96 | 3/5 | 2542 | 63 | 13.0 |

Replanning twice as often doubles the compute for a modest distance gain. A **96 px horizon is
clearly too short** — the lookahead must be long enough to see past an obstacle, or the planner
commits to a trajectory that cannot clear it.

## Why A* cannot solve 1-2 (an honest limitation)

| Setting | Nodes in 150 s | Max x expanded |
|---|---|---|
| Optimal A\* (w = 1.0) | 1,615 | **934** |
| Weighted A\* (w = 3.0) | 1,760 | **3,028** |

Optimal A\* bogs down early, expanding almost uniformly across the first 900 px: the
x-distance heuristic carries no information about vertical structure, so on a level built around
descents it degenerates toward breadth-first. Weighted A\* reaches x ≈ 3028, essentially the end
of the level, but never triggers stage completion — 1-2's exit works differently from 1-1's
flagpole.

This is a limitation of **the heuristic and the goal formulation**, not a budget shortfall.
Fixing it would need a heuristic that accounts for vertical structure and a goal test matching
1-2's exit, both of which are beyond the one-week scope.

## Two harness bugs that produced plausible, wrong behaviour

Neither raised an error; both would have corrupted the report.

1. **Frame-stack spacing.** PPO's 4-frame stack is fed by a frame-skip-4 env during training, so
   slots are *four game frames* apart. The harness initially pushed a frame every frame while
   the agent decided every fourth, giving a quarter of the expected motion between slots. PPO
   appeared to fail at 0% completion — flatly contradicting Phase 4's measured 100% — and was
   only caught by that cross-check. **The shared harness must reproduce each agent's
   training-time observation contract exactly**, not merely give it an observation of the right
   shape.
2. **Replan backoff.** A failed replan immediately retried instead of respecting the interval
   (`or not self._queue` in the gate). On a level the planner cannot solve the queue is *always*
   empty, so it replanned every frame at the full search budget: one episode cost hours instead
   of minutes. A failed replan must back off like any other.

## Phase 5b — the live demo

`scripts/demo.py` draws the planner's trajectory on the game itself — a dotted path ahead of
Mario, with a border flash on each replan — and always records an mp4.

The control that matters is **`s`, the live sticky toggle**. Pressing it mid-run reproduces the
headline result in front of the viewer: the open-loop plan desynchronises and Mario dies, while
the closed-loop planner visibly redraws its path and recovers. 0.15 versus 0.80, shown rather
than tabulated.

Recorded: `media/demo_astar_open_loop_1-1.mp4`, `media/demo_astar_closed_loop_1-1.mp4` (sticky
on, 70 replan flashes over 1,156 frames, matching `replan_every = 15`), `media/demo_ppo_1-1.mp4`.

`--headless` produces the mp4 with no window, and a failed window init falls back to it
automatically, so the deliverable never depends on a GUI working on the day.

**`--agent human` is best-effort and labelled as such.** `cv2.waitKey` reports key presses but
not key-up events, so *holding* a button — exactly what Mario needs — cannot be detected. It
latches instead: a press persists until the next one. A human score collected this way is a
rough floor, not a fair measure of human skill. Doing it properly needs a different input
backend and is out of scope.
