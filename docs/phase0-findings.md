# Phase 0 — Environment spike findings

Date: 2026-09-29. Machine: MacBook (Apple M1 Max, 10-core CPU / 32-core GPU), macOS 27 (Darwin 27.0.0),
Apple clang 21. All measurements single-threaded unless stated.

## Headline: the stack is far healthier than the 2022-era tutorials suggest

The original plan assumed the well-known legacy stack (`nes-py 8.2.1` + `gym 0.21` + a `shimmy`
compatibility bridge + `numpy<2` + compiling a C++ extension against clang 21). **None of that is
needed.** `nes-py` 9.x and `gym-super-mario-bros` 9.x are modernised:

| Assumption going in | Reality |
|---|---|
| Must compile C++ ext; clang 21 risk | **Prebuilt `macosx_11_0_arm64` wheels** (cp313/cp314). No compiler used. |
| Old `gym`, needs `shimmy` bridge | **Gymnasium-native** (`gymnasium>=1.0`). 5-tuple `step()`. No bridge. |
| Must pin `numpy<2` | Runs on **numpy 2.5.3**. |
| Use Python 3.11 (3.12 risky) | `nes-py>=9.0.0` requires **`>=3.13`**. Using **3.13.14**. |
| Single snapshot slot; need replay | **Arbitrary multi-slot snapshots** natively supported. |

Consequence: Phase 1 is no longer the project's main risk, and `stable-retro` is not needed as a
fallback. Total install time is seconds.

## Pinned stack

Python 3.13.14 (via `uv`). `gym-super-mario-bros==9.1.0`, `nes-py==9.0.1`, `gymnasium==1.3.0`,
`stable-baselines3==2.9.0`, `torch==2.14.0`, `numpy==2.5.3`. Fully locked in `requirements.txt`
(49 packages). No dependency conflicts: SB3 wants `gymnasium<2.0,>=0.29.1`, nes-py wants `>=1.0.0`,
and 1.3.0 satisfies both.

## Finding 1 — Exact forward simulation is available (decides Phase 3)

`NESEnv` exposes:

```python
snapshot = env.unwrapped.dump_state()     # -> opaque NativeStateSnapshot
env.unwrapped.load_state(snapshot)        # O(1) restore
```

Verified:
- **Round trip** exact (x=117 -> advance -> restore -> x=117).
- **Multi-slot**: 4 snapshots at distinct positions (x = 162, 207, 252, 297) restored **out of order**,
  repeatedly, all exact.
- **Deterministic**: 3 identical action sequences from one snapshot -> identical `(x, y, player_state)`.
- **Portable across emulator instances**: a snapshot from env A loads into env B.

This means A* node expansion is an **exact** `load_state -> run primitive -> dump_state`, so the
deterministic-replay workaround in the original plan is **dropped**. Collision checking is exact
rather than approximate, and no approximate physics model is needed.

**Limitation:** snapshots are **not picklable** — in-memory only, cannot be cached to disk. Plans
must therefore be persisted as *action sequences*, not as states. (Fine: replaying an action
sequence from `reset()` is deterministic.)

## Finding 2 — Speed budget: ~1,200 emulated frames/s per core

| Operation | Cost |
|---|---|
| `frame_advance` | 0.826 ms  (**1,210 frames/s**) |
| `+ _get_info()` | 0.850 ms  (info costs 0.018 ms) |
| `dump_state` | 0.020 ms |
| `load_state` | 0.012 ms |

Python call overhead is negligible (calling the native method directly is the same speed), so this is
real emulation cost including PPU rendering. There is **no headless / render-disabled fast path** —
the only advance method is `frame_advance`.

Implications:
- **A\***: a 12-frame primitive plus snapshot costs ~10.2 ms => **~98 edges/s**. A full-level plan
  that expands a few thousand nodes takes tens of seconds — acceptable offline, but it means
  **closed-loop replanning must use a bounded receding horizon** (~60–90 frames, 5–8 primitives deep)
  to stay inside a sub-second budget. Full-level replans at every step are not affordable.
- **RL**: ~1,200 frames/s per core = ~300 agent steps/s at frame-skip 4. Parallelism is mandatory.

## Finding 3 — `NativeVectorEmulator` is NOT parallel

`nes_py._native.NativeVectorEmulator(rom, n)` exists and looked promising, but total throughput is
**flat** as n grows:

```
n_envs    steps/s   total frames/s   speedup
     1      1,197            1,197     1.00x
     8        150            1,200     1.00x
    24         50            1,194     1.00x
```

It is a *serial* C++ loop over a same-ROM group — a convenience API (batched observation copies,
per-index `dump_state(i)`/`load_state(i, snap)`), not a speedup. **Phase 4 parallelism must come from
SB3's `SubprocVecEnv`** (real OS processes). With 8 worker processes expect ~9,600 frames/s
=> ~2,400 agent steps/s at frame-skip 4, so ~5M steps is roughly 35 min of pure env time and
plausibly 1–2 h wall clock including PPO updates.

## Finding 4 — The planner must bypass the Gymnasium wrapper stack

Stepping through `JoypadSpace` / `TimeLimit` after a `load_state` raises
`ValueError: cannot step in a done environment`. Those wrappers keep episode bookkeeping
(elapsed-step counters, done flags) that a snapshot jump silently invalidates.

**Design rule:** the planner's simulator drives the raw emulator — `unwrapped._frame_advance(byte)`
plus `unwrapped._get_info()` — with no Gymnasium wrappers. The RL agent keeps the full wrapper stack.
The two therefore need separate env factories.

## Finding 5 — Phase 2 is much cheaper than planned

`_get_info()` already returns 40+ parsed fields, including `x_pos`, `y_pos`, `y_pixel`,
`y_viewport`, `x_pos_max`, `left_x_pos`, `enemy_types` (5 slots), `player_state`, `powerup_level`,
`status`, `is_dying`, `is_dead`, `flag_get`, `time`, `score`, and a decomposed `reward_components`
dict. Still to extract from RAM ourselves: **velocity (vx, vy)**, the **tile grid**, and **enemy
x/y positions** (only enemy *types* are in `info`).

Useful native helpers found:
- `unwrapped.ram` -> zero-copy `(2048,) uint8` view of system RAM.
- `unwrapped.ram_values([addr, ...])` -> native batched read into a `uint32` array, with BCD/endian
  encodings available. Ideal for cheap per-frame tile-grid extraction.
- `unwrapped.observation(mode='grayscale')` -> native `(240, 256) uint8`. **Removes the need for a
  Python/cv2 grayscale wrapper** in the RL preprocessing chain.

## Finding 6 — Death detection for the collision checker

Holding `right+B` from reset, Mario hits the first Goomba at x≈315:

```
frame  x_pos  player_state  is_dying
  100    297             8     False
  120    315            11      True     <- contact
```

`player_state` flips 8 -> 11 and `is_dying` goes True on the contact frame, while `x_pos` freezes for
the death animation. So **`is_dying` / `player_state == 11` is a clean, immediate edge-invalidity
signal** for the planner: if a primitive's forward simulation sets it, that edge is a collision.

(This also caused a false positive during the spike — snapshots taken after death all reported the
same frozen `x_pos` and looked like a passing multi-slot test. Worth guarding in the unit tests.)

## Action space

`SIMPLE_MOVEMENT` (7 actions) maps to NES button bytes as:

| index | buttons | byte |
|---|---|---|
| 0 | NOOP | 0x00 |
| 1 | right | 0x80 |
| 2 | right, A | 0x81 |
| 3 | right, B | 0x82 |
| 4 | right, A, B | 0x83 |
| 5 | A | 0x01 |
| 6 | left | 0x40 |

The planner works in raw bytes; motion primitives will be defined over these.

## Torch / device

MPS is available and built. Per the plan this is expected to be **slower than CPU** for an 84x84
NatureCNN (kernel-launch overhead dominates, and the real bottleneck is single-threaded emulation).
To be benchmarked in Phase 4 and reported as a result. `torch.get_num_threads() == 8`.

## Open questions deferred to later phases

1. Exact RAM addresses for vx/vy and the tile grid need empirical verification (Phase 2 visualiser).
2. Whether `y_pos` from `info` is screen-relative or level-relative (matters for the planner's state);
   the run-jump trace showed it varying 79 -> 132, needs disambiguating against `y_viewport`/`y_pixel`.
3. Whether 1-2's underground tileset breaks the tile-grid extraction (noted as a possible reportable
   finding).
