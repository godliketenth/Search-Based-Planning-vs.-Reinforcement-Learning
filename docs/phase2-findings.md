# Phase 2 — state extraction findings

Date: 2026-10-04. Every address below was verified against observable behaviour on a running
emulator, not taken on trust from a published RAM map. The checks are automated in
`scripts/phase2_inspect.py --verify` and in `tests/test_ram_map.py`, so they re-run on demand.

## Verified addresses

| Quantity | Address | Units / encoding | How it was verified |
|---|---|---|---|
| `x` | `0x6D` × 256 + `0x86` | pixels, level-absolute | Equals `info['x_pos']` on every frame |
| viewport left edge | `0x071C` | pixels | Always brackets Mario within 256 px |
| **`vx`** | `0x0057` | **signed, 1/16 px per frame** | Mean \|Δx − vx/16\| = **0.17 px** over a run |
| **`vy`** | `0x009F` | **signed, whole px per frame, +down** | Mean \|Δy − vy\| = **0.23 px** |
| **`on_ground`** | `0x001D == 0` | 0 ground, 1 air, 3 flagpole | Flips at the exact jump/landing frames; `vy == 0` on 172/172 grounded frames |
| `y` (screen) | `0x00CE` | px, downwards | Equals `255 − info['y_pos']` |
| `y` (level) | `0x00B5` × 256 + `0x00CE` | px, downwards | Monotonic through a pit fall; does not wrap |
| powerup | `0x0756` | 0 small, 1 big, 2 fire | Matches `info['powerup_level']` |
| player state | `0x000E` | `0x08` normal, `0x0B` dying, `0x06` dead | — |
| **tile grid** | `0x0500`–`0x069F` | 2 pages × 13 rows × 16 cols | Solid tile under Mario's feet on **1148/1148** grounded frames (1-2), 171/172 (1-1) |
| enemy active | `0x000F + slot` | 5 slots | — |
| enemy type | `0x0016 + slot` | `0x06` = Goomba | Matches `info['enemy_types']` exactly |
| enemy x | `0x006E + slot` × 256 + `0x0087 + slot` | px, level-absolute | Converges on Mario; **10 px away on the death frame** |
| enemy y | `0x00CF + slot` | px, screen | Drawn boxes land on the sprites |

**The two velocity components use different units.** `vx` counts sixteenths of a pixel
(`0x30` = 48 = the 3 px/frame running cap); `vy` counts whole pixels. This is a real hardware
quirk and an easy silent bug, so `ram_map.Velocity` names its fields rather than returning a
bare tuple.

## Tile buffer: a two-page ring, not a flat array

The 416 bytes at `0x0500` are **two screens** of 13 × 16 tiles, indexed by level page:

```
index = 0x0500 + (level_page % 2) * 208 + row * 16 + (level_col % 16)
    level_col = x_level // 16        level_page = level_col // 16
    row        = (y_screen - 32) // 16      # row 0 starts below the status bar
```

Verified by dumping the buffer beside a rendered frame at x = 561: buffer page 1 columns
12–13 is the short pipe at the screen's left edge, page 0 columns 6–7 is the tall pipe at
screen x ≈ 160. The debug overlay then confirms the alignment continuously.

Tile vocabulary observed: `0x00` empty, `0x54` ground, `0x51`/`0x52` brick, `0xC0`/`0xC1`
question block, `0xC2` coin, `0x12`–`0x15` pipe, `0x5C`/`0x5D`/`0x61`/`0x23` underground
scenery. **Non-zero means solid, except the coin (`0xC2`)**, which is drawn but passable.

> **Limitation that matters for Phase 3:** the buffer only ever holds about two screens, so
> this is a **local** obstacle map, not a global one. The planner cannot consult tiles far
> ahead of Mario. This is acceptable only because collision checking is done by **exact
> forward simulation**, not by reading this grid — the grid is for state features, the
> heuristic and visualisation.

## Phase 0's open questions, resolved

**1. vx/vy and the tile grid — resolved.** Addresses above, all verified.

**2. Is `y_pos` screen- or level-relative? — screen-relative, and inverted.**
`smb_env` computes `y_pos = 255 - y_pixel`, so it increases *upwards* and is measured from
the screen, not the level. It is therefore **not usable as a C-space coordinate**: it wraps
when Mario falls out of the viewport. `ram_map.read_y_level()` composes
`y_viewport * 256 + y_pixel` instead, giving a value that increases downwards and stays
continuous through a pit fall.

**3. Does 1-2's underground tileset break extraction? — no.**
1-2 uses a different tile vocabulary (`0x52` brick, `0x61`, `0x5C`, `0x5D`, `0x23`) but the
same ring-buffer layout and the same "non-zero is solid" rule. The feet-are-on-solid-ground
check passes on **1148/1148** grounded frames on 1-2, and the overlay aligns pixel-perfectly
with the ceiling, blocks and floor. Zero-shot evaluation on 1-2 is unaffected.

## Corrections to Phase 0

**`ram_values()` is the slow path, not the fast one.** Phase 0 suggested it was "ideal for
cheap per-frame tile-grid extraction". Measured:

| Method | 30 scattered addresses |
|---|---|
| `env.ram_values(addrs)` | 14.41 µs |
| `ram[np_index_array]` | **0.17 µs** |
| `ram[0x500:0x6A0]` (view) | 0.09 µs |

Native marshalling costs more than it saves. `ram_map` uses numpy fancy-indexing over the
zero-copy `env.ram` view with precomputed index arrays. All of these are negligible beside a
660 µs `frame_advance`, but the cheap option is also the simpler one.

## New finding: death has two causes, not one

`smb_env._is_dying` is `player_state == 0x0B` **or `y_viewport > 1`**. The second clause is
a **pit fall**, and it fires *before* the death animation starts. Phase 0 recorded only the
`player_state` half.

A collision checker that tests only `player_state == 0x0B` therefore keeps expanding nodes
for a Mario who has already fallen into a pit and is merely still in the air. `ram_map.read_is_dying()`
implements both clauses, and `tests/test_ram_map.py` asserts it agrees with `info['is_dying']`
on every frame of a run.

As in Phase 0: `x_pos` freezes once dying, so position must never be used to infer death.

## Sprite geometry

Mario's stored y (`0x00CE`) is the top of the **big**-Mario bounding box and does not change
with powerup level — his feet are always at `y + 32`, and small Mario is drawn in the lower
half. Verified for small Mario against the ground surface at screen y = 208. The big-Mario
case is inferred from the y byte being unchanged when the powerup byte is altered, and should
be re-checked the first time a mushroom is actually collected; it affects only the drawn
collision box, not planning.

## Evidence the obstacle map is usable

`scripts/phase2_inspect.py --policy reactive` runs a controller that holds run-right and
jumps **only because the tile grid reports something solid or a gap ahead**. It reaches
**x = 701**, clearing the pipe at x ≈ 594 that Phase 1's random agent could never pass, then
dies on a Goomba. One tile of lookahead and no dynamics model is enough to beat random but
not enough to survive — which is the gap Phase 3's motion primitives and search exist to close.
