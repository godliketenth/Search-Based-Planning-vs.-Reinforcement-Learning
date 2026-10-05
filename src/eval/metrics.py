"""What we measure, and how failures are classified.

A single completion rate hides most of what distinguishes these agents. The metrics here are
chosen so the results table answers three separate questions:

* **Did it work?** completion rate, distance reached.
* **How did it fail?** death cause -- enemy contact, pit, or timeout. Phase 2 established
  that ``is_dying`` has two distinct causes, so this is readable from the game rather than
  guessed.
* **What did it cost?** decision latency at run time, and total emulated frames, which is the
  only currency in which A*'s search and PPO's training are directly comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.env import ram_map


class DeathCause:
    """How an episode ended. Values are used verbatim as CSV cell contents."""

    #: Touched an enemy or a hazard: ``player_state == 0x0B``.
    ENEMY = "enemy"
    #: Fell out of the world: ``y_viewport > 1``. Fires *before* the death animation.
    PIT = "pit"
    #: Step budget or in-game timer ran out without dying.
    TIMEOUT = "timeout"
    #: Reached the flag.
    NONE = "none"


def classify_death(ram: Any, info: dict, flag_get: bool) -> str:
    """Decide how an episode ended.

    The pit test comes first on purpose. Falling into a pit sets ``y_viewport > 1`` several
    frames *before* ``player_state`` flips to the dying value, so testing ``player_state``
    first would misattribute most pit deaths to enemy contact.

    Args:
        ram: the emulator RAM view at the end of the episode.
        info: the final ``info`` dict.
        flag_get: whether the goal was reached.

    Returns:
        One of the :class:`DeathCause` values.
    """
    if flag_get:
        return DeathCause.NONE
    if int(ram[ram_map.ADDR_Y_VIEWPORT]) > 1:
        return DeathCause.PIT
    if int(ram[ram_map.ADDR_PLAYER_STATE]) in (
        ram_map.PLAYER_STATE_DYING, ram_map.PLAYER_STATE_DEAD
    ):
        return DeathCause.ENEMY
    if info.get("is_dying") or info.get("is_dead"):
        return DeathCause.ENEMY
    return DeathCause.TIMEOUT


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One episode's outcome, one row of the results CSV.

    Attributes:
        agent: agent name.
        level: ``"world-stage"``.
        condition: ``"deterministic"`` or ``"sticky_0.25"``.
        episode: index within this (agent, level, condition) cell.
        seed: reset seed used.
        completed: whether the flag was reached.
        max_x: furthest level x reached. Tracked as a maximum because ``x_pos`` freezes
            during the death animation and can decrease when Mario retreats.
        frames: frames emulated in the episode -- "steps to goal" when completed.
        death_cause: one of :class:`DeathCause`.
        reward: summed environment reward, reported for completeness but not relied on.
        decisions: how many times the agent was asked to act.
        decision_ms_mean: mean wall-clock cost of one decision.
        decision_ms_max: worst-case decision cost, which is what a real-time system cares
            about far more than the mean.
        planning_calls: searches run during the episode.
        planning_seconds: time spent searching during the episode.
        nodes_expanded: search nodes expanded during the episode.
        compute_frames: emulated frames spent on planning (A*) or training (PPO).
        sticky_realised: measured fraction of frames where the executed action differed from
            the commanded one, so the configured perturbation can be checked against reality.
        notes: agent-specific extras.
    """

    agent: str
    level: str
    condition: str
    episode: int
    seed: int
    completed: bool
    max_x: int
    frames: int
    death_cause: str
    reward: float = 0.0
    decisions: int = 0
    decision_ms_mean: float = 0.0
    decision_ms_max: float = 0.0
    planning_calls: int = 0
    planning_seconds: float = 0.0
    nodes_expanded: int = 0
    compute_frames: int = 0
    sticky_realised: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """Flatten to a CSV row, rendering ``notes`` as a compact string."""
        row = asdict(self)
        notes = row.pop("notes")
        row["notes"] = ";".join(f"{key}={value}" for key, value in notes.items())
        return row


def summarise(results: list[EpisodeResult]) -> list[dict[str, Any]]:
    """Aggregate episodes into one row per (agent, level, condition).

    Means over ``max_x`` and ``frames`` are taken across all episodes; **mean frames to goal
    is computed over completed episodes only**, because averaging in the frame count of a
    death would make a fast-dying agent look fast.

    Args:
        results: individual episode results.

    Returns:
        Summary rows, sorted by agent, level and condition.
    """
    grouped: dict[tuple[str, str, str], list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault((result.agent, result.level, result.condition), []).append(result)

    rows: list[dict[str, Any]] = []
    for (agent, level, condition), group in sorted(grouped.items()):
        completions = [r for r in group if r.completed]
        causes = [r.death_cause for r in group]
        rows.append(
            {
                "agent": agent,
                "level": level,
                "condition": condition,
                "episodes": len(group),
                "completion_rate": len(completions) / len(group),
                "mean_max_x": sum(r.max_x for r in group) / len(group),
                "median_max_x": sorted(r.max_x for r in group)[len(group) // 2],
                "mean_frames_to_goal": (
                    sum(r.frames for r in completions) / len(completions)
                    if completions else None
                ),
                "deaths_enemy": causes.count(DeathCause.ENEMY),
                "deaths_pit": causes.count(DeathCause.PIT),
                "timeouts": causes.count(DeathCause.TIMEOUT),
                "decision_ms_mean": sum(r.decision_ms_mean for r in group) / len(group),
                "decision_ms_max": max(r.decision_ms_max for r in group),
                "mean_planning_calls": sum(r.planning_calls for r in group) / len(group),
                "mean_planning_s": sum(r.planning_seconds for r in group) / len(group),
                "compute_frames": max(r.compute_frames for r in group),
            }
        )
    return rows
