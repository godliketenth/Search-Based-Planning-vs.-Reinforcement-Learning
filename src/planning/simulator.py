"""Exact forward simulation: the planner's transition function and collision checker.

Most motion planners need an approximate dynamics model and a geometric collision checker,
and both are sources of error. Here the emulator *is* the model. Expanding an edge means:

1. ``load_state(snapshot)`` -- restore the parent configuration exactly (~0.016 ms);
2. run the primitive's bytes through ``_frame_advance`` -- the real physics, including
   enemies, which is why enemy avoidance needs no special handling at all;
3. ``dump_state()`` -- capture the child configuration (~0.039 ms).

So the transition function has **zero modelling error**, and collision checking is exact
rather than conservative: an edge is invalid precisely when the game says Mario died on it.

Snapshot lifetime (the binding constraint)
------------------------------------------
A snapshot costs about **272 KB**. Storing one per expanded node would need gigabytes, so
snapshots are held **only for frontier nodes** and released the moment a node is expanded.
This is safe because the heuristic is consistent, so an expanded node is never reopened.
:class:`SnapshotStore` makes that lifetime explicit instead of leaving it to chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.env import ram_map
from src.env.state import MarioState
from src.planning.primitives import MotionPrimitive

#: Measured size of one emulator snapshot. Used only for reporting memory in search stats.
SNAPSHOT_BYTES = 272_000


@dataclass(frozen=True, slots=True)
class Transition:
    """The result of simulating one primitive from one state.

    Attributes:
        state: the resulting configuration.
        frames: how many frames were actually emulated. Equals the primitive's length
            unless simulation stopped early because Mario died.
        alive: ``False`` if the primitive killed Mario, which makes this edge invalid.
        flag_get: whether this primitive reached the goal flag.
        snapshot: the emulator state afterwards, or ``None`` for an invalid edge (there is
            no point keeping 272 KB for a node that can never be expanded).
    """

    state: MarioState
    frames: int
    alive: bool
    flag_get: bool
    snapshot: Any | None


class SnapshotStore:
    """Owns frontier snapshots and enforces that they are released on expansion.

    Snapshots are not reference-counted by anything else, so dropping the reference here is
    what actually frees the 272 KB. Tracking the live count turns a silent memory blow-up
    into an observable number.
    """

    def __init__(self) -> None:
        self._snapshots: dict[int, Any] = {}
        self.peak_live = 0

    def put(self, node_id: int, snapshot: Any) -> None:
        """Store a frontier node's snapshot."""
        self._snapshots[node_id] = snapshot
        self.peak_live = max(self.peak_live, len(self._snapshots))

    def take(self, node_id: int) -> Any:
        """Retrieve and **release** a snapshot, as a node is expanded.

        Raises:
            KeyError: if the snapshot was already released -- which would mean a node was
                expanded twice, i.e. the consistency assumption is broken.
        """
        return self._snapshots.pop(node_id)

    def discard(self, node_id: int) -> None:
        """Release a snapshot for a node that will never be expanded (pruned)."""
        self._snapshots.pop(node_id, None)

    def __len__(self) -> int:
        """How many snapshots are currently live."""
        return len(self._snapshots)

    @property
    def peak_bytes(self) -> int:
        """Approximate peak memory held in snapshots."""
        return self.peak_live * SNAPSHOT_BYTES


class ForwardSimulator:
    """Drives the raw emulator as a transition function.

    The env must be a **planner** env (no Gymnasium wrappers): stepping a wrapped env after
    ``load_state`` raises ``cannot step in a done environment``, because wrapper bookkeeping
    assumes time moves forward (Phase 0, Finding 4).
    """

    def __init__(self, env: Any) -> None:
        """Wrap a planner env.

        Args:
            env: a raw :class:`SuperMarioBrosEnv` from ``make_planner_env``.
        """
        self.env = env
        self.frames_simulated = 0
        self.edges_expanded = 0

    def root(self, seed: int = 0) -> tuple[MarioState, Any]:
        """Reset the emulator and return the start configuration and its snapshot."""
        self.env.reset(seed=seed)
        return MarioState.from_env(self.env), self.env.dump_state()

    def expand(self, snapshot: Any, primitive: MotionPrimitive) -> Transition:
        """Simulate one primitive from a snapshot.

        Simulation stops early the moment Mario dies -- there is nothing to learn from the
        rest of a fatal trajectory, and the saved frames add up across thousands of edges.

        Args:
            snapshot: the parent configuration's emulator state.
            primitive: the edge to simulate.

        Returns:
            A :class:`Transition`. Check ``alive`` before using it as a graph edge.
        """
        env = self.env
        env.load_state(snapshot)
        ram = env.ram

        frames = 0
        alive = True
        for byte in primitive.bytes_:
            env._frame_advance(byte)
            frames += 1
            if ram_map.read_is_dying(ram):
                alive = False
                break

        self.frames_simulated += frames
        self.edges_expanded += 1

        if not alive:
            # Do not pay 272 KB to snapshot a dead end.
            return Transition(
                state=MarioState.from_ram(ram), frames=frames,
                alive=False, flag_get=False, snapshot=None,
            )

        flag_get = bool(env._get_info()["flag_get"])
        return Transition(
            state=MarioState.from_ram(ram), frames=frames,
            alive=True, flag_get=flag_get, snapshot=env.dump_state(),
        )

    def trajectory(self, snapshot: Any, byte_sequence: list[int]) -> list[tuple[int, int]]:
        """Replay a byte sequence and record positions, for drawing a plan on screen.

        Args:
            snapshot: state to start from.
            byte_sequence: the bytes to apply.

        Returns:
            ``(x_level, y_screen)`` per frame, usable directly by the debug overlay.
        """
        self.env.load_state(snapshot)
        ram = self.env.ram
        points: list[tuple[int, int]] = []
        for byte in byte_sequence:
            self.env._frame_advance(byte)
            points.append((ram_map.read_x(ram), ram_map.read_y_screen(ram) + 16))
            if ram_map.read_is_dying(ram):
                break
        return points
