# Phase 4 — PPO findings

Date: 2026-10-05. Level 1-1, seed 0, Apple M1 Max (10 CPU cores / 32 GPU cores).

## Headline: PPO solves 1-1 — and does not transfer at all

| Quantity | Value |
|---|---|
| Training | **5,000,672 agent steps** = 20.0 M emulated frames |
| Wall clock | **74.1 min** (697 agent steps/s, 2,789 frames/s) |
| Completion rate, 1-1, greedy policy | **100%** (32/32 episodes) |
| Completion rate, 1-1, sampling policy | 97% (30 episodes) |
| Completion rate, **1-2 zero-shot** | **0%** (0/32), mean max x = **354** |
| Inference cost | **1.4 ms per decision** |

PPO learns 1-1 to saturation and then fails 1-2 completely — it does not get past x = 354
of roughly 3,000. The policy has not learned "play Mario"; it has learned *this level*.
A\* will run on 1-2 with no modification at all, because it searches the dynamics rather
than memorising a trajectory. That contrast is the strongest single result in the project
and Phase 5 will quantify it properly.

## Learning curve

`results/figures/phase4_curve_ppo_1-1.png`, from the run's TensorBoard events.

| Steps | Completion rate | Mean max x |
|---|---|---|
| 0 – 1.0 M | **0.00** | 650 → 1,500 |
| 1.1 M | 0.20 | 2,220 |
| 1.7 M | 0.42 | 2,660 |
| 2.2 M | 0.80 | 2,950 |
| 3.0 M | 0.92 | 3,120 |
| 3.6 M – 5.0 M | **0.95 – 1.00** | 3,150 – 3,161 |

Three features worth describing in the report:

* **A flat first million steps.** PPO makes essentially no progress on completion for 1 M
  steps (4 M frames) while it discovers that jumping is useful at all. A\* needed no such
  phase: its primitive set *starts* with jumps of several heights, so the manoeuvre is given
  rather than discovered. This is prior knowledge versus sample cost, made visible.
* **Two sharp collapses**, at ~2.35 M and ~4.8 M steps, where completion falls from ~0.8 to
  ~0.15 and recovers within ~200 k steps. This is ordinary PPO policy collapse. Nothing
  comparable can happen to A\*: a plan does not degrade, because there is no policy to
  update. **Monotonic improvement is not a property RL offers.**
* **It beat the plateau I would have stopped at.** At 1.9 M steps the completion rate had sat
  between 0.34 and 0.43 for ~50 k steps and looked converged. It was not — the run went to
  0.80 within another 250 k steps. An apparent RL plateau is weak evidence of convergence.

## Compute asymmetry: the number that frames the whole comparison

| | A\* (Phase 3) | PPO (Phase 4) | Ratio |
|---|---|---|---|
| Emulated frames | 71,637 | 20,002,688 | **279x** |
| Wall clock | 53.5 s | 4,446 s | **83x** |
| Per-decision cost at run time | ~0 (plan is precomputed) | 1.4 ms | — |
| Completion on 1-1 (deterministic) | 100% | 100% | — |
| Completion on 1-2 | *unchanged algorithm* | **0%** | — |

Both reach 100% on 1-1, by paying in entirely different currencies:

* **A\*** is given a perfect model and privileged RAM state, and spends its budget *at plan
  time*, once, per level. Change the level and it simply re-plans.
* **PPO** is given pixels and no model, spends its budget *at training time*, and then acts
  in 1.4 ms. Change the level and it has nothing.

This is the **assumption asymmetry** the project committed to stating rather than hiding.
They are not like-for-like agents and the results table must say so.

## Correction to a Phase 0 expectation: MPS wins, and by a lot

Phase 0 predicted MPS would be **slower** than CPU for an 84x84 NatureCNN, reasoning that
the kernels are small, launch overhead would dominate, and the real bottleneck is
single-threaded emulation. Measured:

| Benchmark | CPU | MPS | MPS advantage |
|---|---|---|---|
| NatureCNN forward+backward, batch 256 | 287.11 ms | **8.54 ms** | **33.6x** |
| End-to-end PPO training, 8 workers | 187 agent steps/s | **672 agent steps/s** | **3.6x** |

The reasoning was wrong about the mechanism. The bottleneck is not kernel launch overhead
but **CPU contention**: eight emulator workers already saturate the cores, so running the
gradient step on the CPU makes it compete with the very processes feeding it. MPS moves that
work to the GPU and leaves the cores for emulation. The end-to-end gain (3.6x) is smaller
than the isolated gain (33.6x) precisely because emulation still dominates.

`configs/ppo.yaml` now defaults to `mps`, and `resolve_device("auto")` prefers it.
Reproduce with `scripts/phase4_benchmark.py`.

## Correction to a Phase 0 expectation: SubprocVecEnv scaling is IPC-bound

Phase 0 predicted ~9,600 frames/s from 8 workers (roughly linear). Measured, with the
observation reduced **inside** each worker:

| Workers | frames/s | Speedup |
|---|---|---|
| 1 | 1,205 | 1.00x |
| 2 | 2,234 | 1.85x |
| 4 | 4,185 | 3.47x |
| **8** | **7,697** | **6.39x** |
| 10 | 6,269 | 5.20x |

Two corrections:

1. **Shipping raw RGB caps scaling at 2.69x** (2,823 frames/s at 8 workers). Every step
   pickles a 184 KB frame through a pipe; at 84x84 grayscale it is 6.9 KB. Grayscaling and
   resizing in the worker is worth **2.7x on its own** — the pipe, not the emulator, was the
   bottleneck. Frame stacking is likewise done at the *vector* level, so one frame crosses
   the pipe per step rather than four.
2. **More workers than 8 is slower**, on a 10-core machine. The main process needs cores for
   the PPO update; oversubscribing makes workers and optimiser contend.

## Incidental: two Gymnasium wrappers are unusable here

* `FrameStackObservation` raises `ValueError: Output array has wrong dimensionality` at
  `reset()` on this observation shape.
* `ResizeObservation` returns `(84, 84)` while declaring `(84, 84, 1)`, which breaks
  `VecFrameStack` downstream with a broadcast error.

`src/rl/wrappers.py` provides small replacements. The grayscale one also reads the
emulator's **native** grayscale buffer (Phase 0, Finding 5), so the RGB→gray conversion
never happens in Python at all.

## Sticky actions apply at the frame level, deliberately

`StickyActionWrapper` sits *inside* `FrameSkip`, so each of the four held frames is
perturbed independently. This departs from the ALE convention of perturbing once per agent
step, so that PPO and the A\* plan executor face **the same perturbation process at the same
granularity**. Without that, "25% sticky" would mean two different things for the two agents
and the Phase 5 comparison would not be like-for-like.

Training itself used `sticky_p = 0.0`: sticky actions are an *evaluation* condition, not a
training one.

## Honest limitations

* **One seed.** Multi-seed runs are out of scope for the one-week timebox, so the learning
  curve is one sample. The two collapses are a reminder that RL run-to-run variance is real.
* **No hyperparameter search.** SB3's Atari defaults were used unchanged. The comparison is
  planning versus RL, not a tuning exercise, and a tuned PPO would presumably need fewer
  samples.
* **The 1-2 evaluation here is a quick 32-episode check**, not the full harness, and it was
  run from an ad-hoc script rather than a committed one (`results/raw/phase4_eval_quick.json`
  holds its output). Phase 5 evaluates both agents properly, under both conditions, through
  one shared interface, and that is the number the report should cite.

Everything else here is reproducible from committed scripts: `scripts/phase4_train.py`
(training and the end-to-end device comparison), `scripts/phase4_benchmark.py` (worker
scaling, NatureCNN device timing) and `scripts/phase4_curve.py` (figure and CSV, read from
the run's TensorBoard events).
