# Phase 1 — setup and sanity check findings

Date: 2026-10-04. Level 1-1, seed 0, Apple M1 Max, single core.

## Purpose

Three things, none of them "get Mario to move":

1. Prove the Gymnasium stack runs end to end on this machine through the 5-tuple API.
2. **Re-measure the emulation throughput the entire A\* design depends on**, from committed
   code rather than a one-off spike, because the Phase 3 cost model rests on it.
3. Produce the floor baseline — how far an agent gets with no model and no lookahead at all.

## Measured: the emulator cost model

| Operation | Cost |
|---|---|
| `_frame_advance` | **0.675 ms** (**1,482 frames/s**) |
| `dump_state` | 0.043 ms |
| `load_state` | 0.018 ms |
| **Implied A\* edge** (12-frame primitive + snapshot round trip) | **≈ 8.2 ms ⇒ ~123 edges/s** |

This is the number that forces a design decision rather than merely describing the machine:
**closed-loop replanning must use a bounded receding horizon.** A full-level replan expands
hundreds of nodes and costs tens of seconds; replanning that way every few frames is
impossible. Phase 5 later measured bounded replans at 84–500 ms, which is what makes
closed-loop control affordable at all.

Phase 0 measured 1,210 frames/s on the same machine; 1,482 here. The difference is noise plus
a warmer cache — the budget holds either way.

Throughput through the full wrapper stack was **1,168 agent steps/s**, essentially the raw
emulation rate, confirming the Gymnasium wrappers add negligible overhead.

## The random baseline: stuck at the first pipe

2,000 agent steps on 1-1:

| Quantity | Value |
|---|---|
| Max `x_pos` | **595** (flagpole is at 3,161) |
| Deaths | **0** |
| Episodes | 1 (truncated at the step limit) |

**The random policy stalls permanently at x ≈ 594 — the first pipe — on every seed tested, and
does not even die there.** It passes the first Goomba at x ≈ 315 by accident, then jams against
the pipe for the remaining ~1,600 steps.

This is the most useful result in the phase, and it is a motion-planning observation, not a
curiosity:

> Clearing the pipe requires a **committed multi-frame manoeuvre** — jump while already moving
> right, and hold right through the entire arc. Mario's air control is limited, so a direction
> reversed mid-jump cannot be undone. A controller that re-samples its button every frame cannot
> represent commitment, so it cannot clear the obstacle at any sampling rate or for any length
> of time.

That is the kinodynamic constraint stated as an experiment, and it is the direct motivation for
searching over **motion primitives** — short, atomic, committed action sequences — instead of
over individual button presses or grid moves. Phase 2's reactive controller (one tile of
lookahead) later reached x = 701 past this pipe, and Phase 3's planner reached the flag.

## Sticky actions verified

At the configured `p = 0.25`, the measured override rate over 2,000 steps was **0.254**. The
perturbation is applied as designed and its realised rate is reported in every run's summary,
so a configured value can always be checked against what actually happened.

## Design decisions locked in this phase

**Two environment factories, because the emulator forces it.** Phase 0 (Finding 4) established
that a wrapped Gymnasium env cannot be stepped after `load_state` — `TimeLimit` and
`OrderEnforcing` keep episode bookkeeping that a snapshot jump invalidates. A* jumps around the
state space constantly, so:

| | `make_planner_env` | `make_rl_env` |
|---|---|---|
| Wrappers | **none** | `JoypadSpace` + `TimeLimit` (+ sticky) |
| Driven by | `_frame_advance(byte)` / `_get_info()` | `step(action)` → 5-tuple |
| Actions | joypad bytes (`0x82` = right+B) | `Discrete(7)` indices |
| Used by | A* forward simulation | PPO and the random baseline |

**One action table, asserted at import.** `src/env/actions.py` derives the byte for each action
exactly as `JoypadSpace` does, then asserts the result matches the table verified in Phase 0.
A dependency upgrade that reordered `SIMPLE_MOVEMENT` would otherwise silently invalidate every
cached plan and every logged action; now it fails loudly at import.

**Sticky actions implemented once, at byte level.** `StickyActionSampler` works on plain ints so
the planner's executor can use it directly (the planner has no Gymnasium wrappers to hang
behaviour off), with a thin `ActionWrapper` over it for PPO. Both agents therefore face an
identical perturbation process — which the Phase 5 comparison depends on.

## A bug worth recording

`env.render()` returns a **view into the emulator's screen buffer**, and hands back the *same
array object* every call. The first GIF written this phase collapsed to a single frame, because
all 1,000 captured frames aliased the final screen. Frames are now explicitly copied, and
`tests/test_env_factories.py` pins the aliasing behaviour so the Phase 5b mp4 recorder — which
has the same exposure — cannot regress.

A second, smaller one found later while writing this document: the realised sticky rate was read
from the sampler *after* an episode reset had cleared its counters, so it reported 0.0 whenever
the step budget coincided with the episode limit. It now accumulates across episodes.

## Artefacts

- `results/figures/phase1_random_1-1_sticky0.00_seed0.gif` — the random agent jammed at the pipe
- `results/raw/phase1_random_1-1_*.jsonl` — per-step log
- `results/raw/phase1_summary_1-1_*.json` — summary plus the benchmark above

Reproduce with:

```bash
.venv/bin/python scripts/phase1_sanity.py --steps 2000
.venv/bin/python scripts/phase1_sanity.py --steps 2000 --sticky 0.25 --no-gif
```
