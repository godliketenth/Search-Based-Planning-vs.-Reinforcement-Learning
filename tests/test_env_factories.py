"""Tests for the two environment factories.

The emulator-backed tests here are the foundation of Phase 3: if snapshot restore is not
*exact*, A* is searching a different state space from the one it will execute in, and every
plan it produces is unsound. These tests are deliberately slow-but-few.
"""

from __future__ import annotations

import pytest
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

from src.env.actions import ACTION_BYTES, RIGHT_B
from src.env.factories import make_planner_env, make_rl_env, parse_level

#: Frames of "right + B" that reliably move Mario without reaching the first Goomba, which
#: Phase 0 located at x ~ 315.
_SAFE_FRAMES = 60


@pytest.fixture(scope="module")
def planner_env():
    """A reset raw emulator, shared across tests to avoid repeated ROM loads."""
    env = make_planner_env("1-1")
    env.reset(seed=0)
    yield env
    env.close()


# --------------------------------------------------------------------------- level parsing


@pytest.mark.parametrize(
    ("level", "expected"), [("1-1", (1, 1)), ("1-2", (1, 2)), ("8-4", (8, 4))]
)
def test_parse_level_accepts_valid_levels(level: str, expected: tuple[int, int]) -> None:
    """The levels the project actually uses must parse."""
    assert parse_level(level) == expected


@pytest.mark.parametrize("level", ["1", "1-1-1", "x-y", "0-1", "9-1", "1-5", ""])
def test_parse_level_rejects_invalid_levels(level: str) -> None:
    """A typo'd level must fail at parse time, not load a silently different stage."""
    with pytest.raises(ValueError):
        parse_level(level)


# -------------------------------------------------------------------------- planner env


def test_planner_env_has_no_wrappers() -> None:
    """Phase 0, Finding 4: wrappers break after load_state, so there must be none.

    ``TimeLimit``/``OrderEnforcing`` keep episode bookkeeping that a snapshot jump
    invalidates, raising 'cannot step in a done environment'. A* jumps constantly.
    """
    env = make_planner_env("1-1")
    try:
        assert isinstance(env, SuperMarioBrosEnv)
        assert not hasattr(env, "env"), "planner env is wrapped; snapshots will break"
        assert env.unwrapped is env
    finally:
        env.close()


def test_planner_env_exposes_the_raw_simulator_api(planner_env) -> None:
    """The planner drives these four calls and nothing else."""
    for attribute in ("_frame_advance", "_get_info", "dump_state", "load_state", "ram"):
        assert hasattr(planner_env, attribute), f"missing {attribute}"
    assert planner_env.ram.shape == (2048,)


def test_planner_env_advances_on_frame_advance(planner_env) -> None:
    """A control input must actually move the agent -- the transition function works."""
    planner_env.reset(seed=0)
    start_x = planner_env._get_info()["x_pos"]
    for _ in range(_SAFE_FRAMES):
        planner_env._frame_advance(RIGHT_B)
    info = planner_env._get_info()
    assert info["x_pos"] > start_x
    assert not info["is_dying"], "test fixture died early; _SAFE_FRAMES is too long"


def test_snapshot_restore_is_exact(planner_env) -> None:
    """load_state must restore state exactly -- the premise of exact forward simulation.

    Guarded against the false positive from the Phase 0 spike: after death ``x_pos``
    freezes, so a 'restored' position can match simply because nothing was moving. The
    test asserts the agent is alive *and* that the position genuinely changed before the
    restore, so an exact match afterwards is meaningful.
    """
    planner_env.reset(seed=0)
    for _ in range(_SAFE_FRAMES):
        planner_env._frame_advance(RIGHT_B)

    before = planner_env._get_info()
    assert not before["is_dying"] and before["player_state"] != 11, "snapshot taken while dying"
    snapshot = planner_env.dump_state()

    for _ in range(30):
        planner_env._frame_advance(RIGHT_B)
    moved = planner_env._get_info()
    assert moved["x_pos"] != before["x_pos"], "state did not change; restore proves nothing"
    assert not moved["is_dying"], "died mid-test; x_pos freezes and the check is vacuous"

    planner_env.load_state(snapshot)
    after = planner_env._get_info()
    for key in ("x_pos", "y_pos", "player_state", "time", "score"):
        assert after[key] == before[key], f"{key} not restored exactly"


def test_snapshots_restore_out_of_order(planner_env) -> None:
    """A* restores arbitrary nodes, not a stack, so order must not matter."""
    planner_env.reset(seed=0)
    snapshots = []
    for _ in range(4):
        for _ in range(20):
            planner_env._frame_advance(RIGHT_B)
        info = planner_env._get_info()
        assert not info["is_dying"], "died while collecting snapshots"
        snapshots.append((planner_env.dump_state(), info["x_pos"]))

    assert len({x for _, x in snapshots}) == len(snapshots), "positions not distinct"
    for snapshot, expected_x in reversed(snapshots):
        planner_env.load_state(snapshot)
        assert planner_env._get_info()["x_pos"] == expected_x


def test_death_is_detectable_immediately() -> None:
    """Phase 0, Finding 6: the collision checker's edge-invalidity signal.

    Holding 'right + B' from reset runs into the first Goomba. ``player_state`` flips to 11
    and ``is_dying`` goes True on the contact frame -- and ``x_pos`` then freezes, which is
    exactly why position must never be used to infer death.
    """
    env = make_planner_env("1-1")
    try:
        env.reset(seed=0)
        death_frame = None
        for frame in range(200):
            env._frame_advance(RIGHT_B)
            info = env._get_info()
            if info["is_dying"] or info["player_state"] == 11:
                death_frame = frame
                break

        assert death_frame is not None, "expected a death on the first Goomba"
        info = env._get_info()
        assert info["is_dying"]
        assert info["player_state"] == 11
        assert 250 < info["x_pos"] < 400, f"died at an unexpected x_pos: {info['x_pos']}"

        # x_pos freezes during the death animation: the Phase 0 false positive.
        frozen_x = info["x_pos"]
        for _ in range(30):
            env._frame_advance(RIGHT_B)
        assert env._get_info()["x_pos"] == frozen_x
    finally:
        env.close()


def test_simulation_is_deterministic_across_instances() -> None:
    """Identical control sequences must give identical trajectories.

    Without this, a cached plan (stored as an action sequence, because snapshots are not
    picklable) could not be replayed, and the deterministic evaluation condition would not
    be deterministic.
    """
    sequence = [RIGHT_B] * 40 + [ACTION_BYTES[4]] * 20 + [RIGHT_B] * 40

    def trajectory() -> list[tuple[int, int, int]]:
        env = make_planner_env("1-1")
        try:
            env.reset(seed=0)
            out = []
            for byte in sequence:
                env._frame_advance(byte)
                info = env._get_info()
                out.append((info["x_pos"], info["y_pos"], info["player_state"]))
            return out
        finally:
            env.close()

    assert trajectory() == trajectory()


# ------------------------------------------------------------------------------- RL env


def test_rl_env_uses_the_discrete_action_set() -> None:
    """PPO sees 7 discrete actions, not 256 joypad bytes."""
    env = make_rl_env("1-1", max_episode_steps=50)
    try:
        assert env.action_space.n == len(ACTION_BYTES)
    finally:
        env.close()


def test_rl_env_action_map_matches_our_byte_table() -> None:
    """The RL agent's action i and the planner's byte must be the same button press.

    This is the cross-check that keeps the two agents comparable at the action level.
    """
    env = make_rl_env("1-1", max_episode_steps=50)  # sticky_p = 0, so this *is* JoypadSpace
    try:
        action_map = env._action_map
        assert tuple(action_map[i] for i in range(len(ACTION_BYTES))) == ACTION_BYTES
    finally:
        env.close()


def test_rl_env_returns_the_gymnasium_five_tuple() -> None:
    """Phase 0: this stack is Gymnasium-native; no compatibility shim is involved."""
    env = make_rl_env("1-1", max_episode_steps=50, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (240, 256, 3)
        assert "x_pos" in info
        result = env.step(3)
        assert len(result) == 5
        _, reward, terminated, truncated, info = result
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(float(reward), float)
    finally:
        env.close()


def test_rl_env_truncates_at_the_step_limit() -> None:
    """max_episode_steps must actually bound an episode, for Phase 5's step budget."""
    env = make_rl_env("1-1", max_episode_steps=10)
    try:
        env.reset(seed=0)
        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        assert truncated
    finally:
        env.close()


def test_rl_env_applies_sticky_actions_only_when_requested() -> None:
    """p = 0 must add no wrapper at all, so the deterministic condition is truly clean."""
    deterministic = make_rl_env("1-1", sticky_p=0.0, max_episode_steps=10)
    stochastic = make_rl_env("1-1", sticky_p=0.25, max_episode_steps=10, seed=0)
    try:
        assert not hasattr(deterministic, "sampler")
        assert hasattr(stochastic, "sampler")
        assert stochastic.sampler.p == 0.25
    finally:
        deterministic.close()
        stochastic.close()


def test_rl_env_render_returns_a_reused_buffer() -> None:
    """render() hands back a *view* into the emulator screen, reused every call.

    Pinned deliberately: forgetting to copy makes every captured frame alias the final
    screen, which silently collapsed the Phase 1 GIF to a single image. Phase 5b's mp4
    recorder depends on this still being true.
    """
    env = make_rl_env("1-1", max_episode_steps=50, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        first = env.render()
        for _ in range(10):
            env.step(3)
        second = env.render()
        assert first is second, "render() no longer reuses its buffer; revisit frame copying"
    finally:
        env.close()
