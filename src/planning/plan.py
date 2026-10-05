"""Plans: what the planner produces, and how it survives to disk.

A plan is an **open-loop control sequence** -- a list of joypad bytes to apply, one per
frame, with no feedback. That is the whole point of the Phase 5 comparison: open-loop
execution is exact on a deterministic emulator and falls apart the moment anything perturbs
it.

Plans are persisted as **action sequences, never as emulator states**. Snapshots are not
picklable (re-confirmed in Phase 3), and in any case replaying a byte sequence from
``reset()`` is deterministic, so the sequence is a complete description.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.env.actions import action_for_byte


@dataclass(frozen=True, slots=True)
class SearchStats:
    """What the search cost, for the report's tables.

    Attributes:
        nodes_expanded: nodes popped from the open list and expanded.
        nodes_generated: edges simulated, i.e. children produced.
        nodes_pruned: children discarded as duplicates or as fatal edges.
        deaths: edges that ended in Mario dying -- the collision checker's work.
        peak_frontier: largest number of snapshots held at once. Multiplied by 272 KB this
            is the planner's peak memory, and the reason snapshots are frontier-only.
        frames_simulated: total emulated frames, the dominant cost.
        wall_s: wall-clock seconds.
        goal_reached: whether the search terminated at the goal.
        exhausted: whether the search ran out of budget instead.
    """

    nodes_expanded: int = 0
    nodes_generated: int = 0
    nodes_pruned: int = 0
    deaths: int = 0
    peak_frontier: int = 0
    frames_simulated: int = 0
    wall_s: float = 0.0
    goal_reached: bool = False
    exhausted: bool = False

    @property
    def edges_per_s(self) -> float:
        """Edge expansion rate, comparable to the ~125 edges/s predicted in Phase 1."""
        return self.nodes_generated / self.wall_s if self.wall_s else 0.0

    @property
    def peak_frontier_mb(self) -> float:
        """Peak snapshot memory in megabytes."""
        from src.planning.simulator import SNAPSHOT_BYTES

        return self.peak_frontier * SNAPSHOT_BYTES / 1e6


@dataclass(frozen=True, slots=True)
class Plan:
    """An open-loop control sequence, plus enough provenance to reproduce it.

    Attributes:
        level: the level this was planned for.
        seed: the reset seed it starts from. A plan is only valid from this seed.
        bytes_: one joypad byte per frame.
        primitive_names: the primitives, in order, that produced ``bytes_``. Kept for the
            demo HUD and for explaining a plan in the viva.
        goal_x: the heuristic's goal, recorded so a stale plan is recognisable.
        final_x: where the planned trajectory ended.
        flag_get: whether the plan reaches the flag.
        stats: the search statistics that produced it.
    """

    level: str
    seed: int
    bytes_: tuple[int, ...]
    primitive_names: tuple[str, ...] = ()
    goal_x: int = 0
    final_x: int = 0
    flag_get: bool = False
    stats: SearchStats = field(default_factory=SearchStats)

    def __len__(self) -> int:
        """Plan length in frames, which is also its cost ``g``."""
        return len(self.bytes_)

    @property
    def actions(self) -> tuple[int, ...]:
        """The plan as ``SIMPLE_MOVEMENT`` indices, for replay through the wrapped RL env."""
        return tuple(action_for_byte(byte) for byte in self.bytes_)

    def save(self, path: str | Path) -> Path:
        """Write the plan to JSON.

        Args:
            path: destination; parent directories are created.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "level": self.level,
            "seed": self.seed,
            "bytes": list(self.bytes_),
            "primitive_names": list(self.primitive_names),
            "goal_x": self.goal_x,
            "final_x": self.final_x,
            "flag_get": self.flag_get,
            "stats": asdict(self.stats),
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        """Read a plan back from JSON."""
        payload = json.loads(Path(path).read_text())
        return cls(
            level=payload["level"],
            seed=payload["seed"],
            bytes_=tuple(payload["bytes"]),
            primitive_names=tuple(payload.get("primitive_names", ())),
            goal_x=payload.get("goal_x", 0),
            final_x=payload.get("final_x", 0),
            flag_get=payload.get("flag_get", False),
            stats=SearchStats(**payload.get("stats", {})),
        )
