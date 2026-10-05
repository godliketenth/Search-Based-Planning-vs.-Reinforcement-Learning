# Phase 3 — A* planner findings

Date: 2026-10-04. Level 1-1, seed 0, Apple M1 Max, single core.

## Headline: A* solves 1-1 open-loop, within 1.5% of the admissible lower bound

| Quantity | Value |
|---|---|
| Goal | flagpole, x = **3161** (found by the planner, now pinned in `configs/astar.yaml`) |
| Plan length | **1056 frames** (~17.6 s of game time) |
| Admissible lower bound | 1040 frames |
| Ratio to bound | **1.015** |
| Nodes expanded | 590 |
| Edges simulated | 4,127 (395 fatal, 2,265 pruned as duplicates) |
| Frames emulated | 71,637 |
| Peak frontier | 878 snapshots (~239 MB) |
| Wall clock | **53.5 s** (77 edges/s) |
| Open-loop replay | **exact** — reproduces x = 3161 and `flag_get` from `reset()` |

The plan is optimal with respect to the primitive set and the lattice resolution. The 1.5%
gap to the bound is not suboptimality: the bound assumes Mario travels at the 3 px/frame cap
from the first frame, whereas he must accelerate from rest.

## The snapshot memory constraint (new, and it shaped the design)

Phase 0 measured snapshot *time* (0.02 ms) but not *size*. Measured here:

```
5,000 snapshots: RSS 44 MB -> 1,402 MB   =>  ~272 KB each
```

Storing one snapshot per expanded node is therefore not viable — 10,000 nodes would be
2.7 GB. The resolution is standard A*: **snapshots are needed only for frontier nodes.**
Because the heuristic is *consistent*, an expanded node is never reopened, so its snapshot
can be released the moment it is expanded. The closed set keeps only lattice keys and `g`
values, a few tens of bytes each.

Peak memory then scales with the open list, not with nodes expanded: **878 snapshots
(~239 MB)** against 590 nodes expanded and 4,127 edges simulated. `SnapshotStore` makes that
lifetime explicit, and raises `KeyError` if a node is ever expanded twice — which would mean
the consistency assumption had been violated.

This is a direct, practical consequence of a textbook property. Consistency is usually
presented as "nodes are not re-expanded, so the search is faster"; here it is what makes the
search *fit in memory at all*.

## The optimal plan is almost entirely jumps

Primitives used in the 1056-frame plan:

| Primitive | Count |
|---|---|
| `long_jump` | 15 |
| `hop` | 9 |
| `jump` | 9 |
| `run` | 2 |
| `walk`, `wait`, `back_off` | **0** |

Jumping costs nothing in time: horizontal velocity is preserved through the arc, so a jump
travels just as far as running while also clearing hazards. A time-minimising planner
therefore jumps almost constantly, and never walks, waits or retreats. The `walk`, `wait` and
`back_off` primitives earn their place only under perturbation — Phase 5 will show whether
the closed-loop planner uses them to recover.

## Weighted A*: the optimality/compute trade-off

Inflating the heuristic by `weight` gives a plan guaranteed to cost at most `weight ×` the
optimal one.

| weight | nodes expanded | wall clock | plan frames | ratio to bound | guaranteed bound |
|---|---|---|---|---|---|
| **1.0** | 590 | 53.5 s | **1056** | 1.015 | 1.00 |
| 1.5 | 203 | 15.7 s | 1152 | 1.107 | 1.50 |
| 2.0 | 153 | 10.1 s | 1340 | 1.288 | 2.00 |
| 5.0 | 148 | 8.8 s | 1400 | 1.346 | 5.00 |

Two things worth saying in the report:

1. **w = 1.5 buys a 3.4× speed-up for a 9% longer plan.** Most of the search cost of optimal
   A* is spent proving that a nearly-as-good plan is not better.
2. **Realised suboptimality is far below the guarantee.** At w = 5.0 the bound permits a plan
   5× optimal; the plan is 1.35×. The bound is worst-case and, as usual, loose.

All four plans replay exactly. Weighted A* loses optimality, not correctness.

## Search behaviour

`results/figures/phase3_search_1-1_w1.0_seed0.png` plots every expanded configuration over
the level. Three things are visible:

* **Expansion order is nearly monotonic in x** — the heuristic is strong, and A* backtracks
  very little.
* **Dense clusters at obstacles**, especially the pipes near x ≈ 500 and the final staircase
  at x ≈ 2800, where the planner had to work out which jump height succeeds.
* **Downward excursions at x ≈ 1150 and x ≈ 1400** — the planner exploring into the pits,
  expanding mid-fall states before those branches die.

395 of 4,127 edges (9.6%) were fatal. Those are the collision checker doing its job: enemy
and pit avoidance is never programmed, it falls out of forward simulation, because the
emulator simulates enemies too.

## Confirmations and corrections

* **Snapshots are still not picklable** (re-confirmed). Plans are persisted as byte
  sequences, which is sufficient because replay from `reset()` is deterministic.
* **Edge throughput is 77–117 edges/s**, against the ~125 edges/s predicted in Phase 1. The
  gap is simply primitive length: Phase 1 assumed 12-frame edges, but the set averages ~21
  frames because jumps are long. Cost tracks *frames emulated*, not edges.
* **The frames-per-second budget holds**: 71,637 frames in 53.5 s is 1,339 frames/s, in line
  with the 1,500 frames/s measured in Phase 1.

## Honest limitations

* **Resolution-complete, not complete.** Only primitives in the set can be executed, and
  states are merged by lattice cell, so a solution needing a manoeuvre outside the set, or
  distinguishable only below the lattice resolution, will not be found.
* **The frontier trim is a second source of incompleteness.** If the open list exceeds
  `max_frontier`, the highest-`f` nodes are discarded to bound memory. This did not trigger
  on 1-1 (878 < 4,000), but it is counted and reported rather than hidden.
* **This is open-loop.** The plan is a memorised button sequence, exact only because the
  emulator is deterministic. That is precisely why Phase 5 adds sticky actions: without a
  perturbation, "A* completes the level 100% of the time" measures the emulator, not the
  planner.
