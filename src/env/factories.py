"""Environment factories: one env for the planner, one for the RL agent.

Why two factories and not one
-----------------------------
Phase 0 (Finding 4) established that stepping a wrapped env after an emulator snapshot
restore raises ``ValueError: cannot step in a done environment``. ``TimeLimit`` and
``OrderEnforcing`` keep episode bookkeeping -- an elapsed-step counter, a terminated flag --
that a ``load_state`` jump silently invalidates. A* expands nodes by jumping around the
state space, so it cannot tolerate wrappers that assume time moves forward monotonically.

So:

* :func:`make_planner_env` returns a **bare** :class:`SuperMarioBrosEnv`. The planner drives
  it with ``_frame_advance(byte)`` / ``_get_info()`` and treats it purely as a transition
  function ``(state, control) -> state`` -- it is a *simulator*, not an RL environment.
* :func:`make_rl_env` returns the **wrapped** env PPO trains against: discrete actions via
  ``JoypadSpace``, a step limit, and optional sticky actions.

The shared ``Agent`` interface (Phase 5) hides this split from the evaluation harness, so
neither agent can be accidentally favoured by its env.
"""

from __future__ import annotations

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace

from .sticky import StickyActionWrapper

#: Levels that exist in Super Mario Bros: worlds 1-8, stages 1-4.
_N_WORLDS = 8
_N_STAGES = 4


def parse_level(level: str) -> tuple[int, int]:
    """Parse a ``"world-stage"`` string into integers.

    Args:
        level: e.g. ``"1-1"`` or ``"1-2"``.

    Returns:
        ``(world, stage)``, both 1-indexed.

    Raises:
        ValueError: if the string is malformed or names a level that does not exist.
    """
    try:
        world_str, stage_str = level.split("-")
        world, stage = int(world_str), int(stage_str)
    except ValueError:
        raise ValueError(f"level must look like 'world-stage', e.g. '1-1'; got {level!r}") from None
    if not 1 <= world <= _N_WORLDS or not 1 <= stage <= _N_STAGES:
        raise ValueError(
            f"level {level!r} does not exist; worlds are 1-{_N_WORLDS}, stages 1-{_N_STAGES}"
        )
    return world, stage


def make_planner_env(level: str = "1-1", render_mode: str | None = None) -> SuperMarioBrosEnv:
    """Build the raw emulator the planner uses as its forward simulator.

    No Gymnasium wrappers are applied, by design -- see the module docstring. The returned
    object is driven with the private emulator API rather than ``step()``:

    >>> env = make_planner_env("1-1")          # doctest: +SKIP
    >>> env.reset(seed=0)                      # doctest: +SKIP
    >>> env._frame_advance(0x82)               # one frame of "right + B"
    >>> info = env._get_info()                 # doctest: +SKIP
    >>> snap = env.dump_state()                # doctest: +SKIP
    >>> env.load_state(snap)                   # exact, ~0.012 ms

    Args:
        level: ``"world-stage"`` to load. The env is built in single-stage mode, so the
            episode ends at this stage's flag rather than advancing to the next stage.
        render_mode: ``None`` for no rendering, or ``"rgb_array"`` to allow ``render()``.
            There is no headless fast path -- the PPU renders regardless (Phase 0,
            Finding 2) -- so this only controls whether frames are *returned*.

    Returns:
        An unwrapped :class:`SuperMarioBrosEnv`.
    """
    world, stage = parse_level(level)
    return SuperMarioBrosEnv(target=(world, stage), render_mode=render_mode)


def make_rl_env(
    level: str = "1-1",
    sticky_p: float = 0.0,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Build the wrapped environment the RL agent acts in.

    Observation preprocessing (grayscale, resize to 84x84, frame-stack, frame-skip) is
    deliberately *not* applied here: it belongs to the PPO policy's input pipeline and is
    added in Phase 4, so that Phase 1 and the Phase 5b demo can work with full-resolution
    RGB frames.

    Args:
        level: ``"world-stage"`` to load.
        sticky_p: sticky-action probability. ``0.0`` is the deterministic condition and adds
            no wrapper at all; ``0.25`` is the stochastic condition of the experimental
            design.
        max_episode_steps: truncate episodes after this many agent steps. ``None`` keeps
            the registered default (effectively unlimited), leaving the in-game 400-second
            timer as the only bound.
        render_mode: ``None``, ``"rgb_array"`` (frames returned by ``render()``, used for
            GIF/mp4 capture) or ``"human"`` (native pyglet window).
        seed: seeds ``reset()`` and the sticky-action RNG, so a run is reproducible.

    Returns:
        A ``Discrete(7)``-action Gymnasium env returning the 5-tuple ``step()`` API.
    """
    world, stage = parse_level(level)
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    kwargs: dict = {"render_mode": render_mode}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym_super_mario_bros.make(env_id, **kwargs)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if sticky_p > 0.0:
        env = StickyActionWrapper(env, p=sticky_p, seed=seed)
    if seed is not None:
        env.reset(seed=seed)
    return env
