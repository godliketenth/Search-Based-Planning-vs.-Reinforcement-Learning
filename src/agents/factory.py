"""Constructing agents from config, in one place.

Three entry points need to build the same five agents from the same YAML: the live OpenCV demo,
the Phase 5 evaluation harness, and the comparison dashboard. This module is that one place --
before it existed the logic was duplicated between ``scripts/demo.py`` and
``scripts/phase5_eval.py``, and a third copy would have guaranteed drift.

:func:`agent_availability` exists separately from :func:`build_agent` because the UI needs to
know *in advance* whether an agent can run, and why not if it cannot. A dashboard that raises
halfway through a demo is worse than one that greys out an option and explains itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.astar_closed_loop import AStarClosedLoopAgent
from src.agents.astar_open_loop import AStarOpenLoopAgent
from src.agents.human_agent import HumanAgent
from src.agents.ppo_agent import PPOEvalAgent
from src.agents.random_agent import RandomAgent
from src.config import PROJECT_ROOT
from src.env.factories import make_planner_env
from src.planning.plan import Plan
from src.planning.primitives import primitives_from_config

#: Every agent the project can evaluate or demonstrate.
AGENT_NAMES: tuple[str, ...] = (
    "random", "astar_open_loop", "astar_closed_loop", "ppo", "human",
)

#: The subset the evaluation harness compares. ``human`` is excluded because a human run cannot
#: be collected unattended, and the input path cannot capture held buttons anyway.
EVALUABLE_AGENTS: tuple[str, ...] = ("random", "astar_open_loop", "astar_closed_loop", "ppo")


def _plan_path(level: str, config: dict[str, Any]) -> Path | None:
    """Resolve the cached open-loop plan for a level, if the config names one."""
    relative = (config.get("plans") or {}).get(level)
    return PROJECT_ROOT / relative if relative else None


def agent_availability(name: str, level: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Report whether an agent can be built, and why not if it cannot.

    Args:
        name: one of :data:`AGENT_NAMES`.
        level: ``"world-stage"``.
        config: a loaded demo or eval config.

    Returns:
        ``(available, reason)``. ``reason`` is empty when available, and otherwise names the
        command that would produce the missing artefact.
    """
    if name in ("random", "human", "astar_closed_loop"):
        return True, ""

    if name == "astar_open_loop":
        path = _plan_path(level, config)
        if path is None:
            return False, f"no plan configured for {level}"
        if not path.exists():
            return False, (
                f"no cached plan for {level} — run: "
                f"scripts/phase3_plan.py --level {level}"
            )
        return True, ""

    if name == "ppo":
        model = PROJECT_ROOT / config["ppo"]["model"]
        if not model.exists():
            return False, "no trained model — run: scripts/phase4_train.py"
        return True, ""

    return False, f"unknown agent {name!r}"


def build_agent(
    name: str,
    level: str,
    config: dict[str, Any],
    astar_config: dict[str, Any],
) -> tuple[Any, Plan | None]:
    """Construct one agent.

    Args:
        name: one of :data:`AGENT_NAMES`.
        level: ``"world-stage"``.
        config: a loaded demo or eval config, supplying plan paths, closed-loop settings,
            goal positions and the PPO model path.
        astar_config: the planner config, supplying the primitive set and lattice resolution.

    Returns:
        ``(agent, plan)``. ``plan`` is the cached :class:`Plan` for the open-loop agent and
        ``None`` otherwise -- callers that want to draw the plan on screen need it, and
        reloading it a second time would be wasteful.

    Raises:
        ValueError: if ``name`` is not a known agent.
        FileNotFoundError: if a required plan or model is missing. Call
            :func:`agent_availability` first to avoid this.
    """
    available, reason = agent_availability(name, level, config)
    if not available:
        raise FileNotFoundError(reason)

    if name == "random":
        return RandomAgent(seed=config.get("seed", 0)), None

    if name == "human":
        return HumanAgent(), None

    if name == "astar_open_loop":
        plan = Plan.load(_plan_path(level, config))
        return AStarOpenLoopAgent(plan), plan

    if name == "astar_closed_loop":
        settings = config["closed_loop"]
        return AStarClosedLoopAgent(
            planner_env=make_planner_env(level),
            primitives=primitives_from_config(astar_config.get("primitives")),
            goal_x=config["goal_x"][level],
            replan_every=settings["replan_every"],
            horizon_px=settings["horizon_px"],
            weight=settings["weight"],
            max_nodes=settings["max_nodes"],
            max_seconds=settings["max_seconds"],
            lattice=astar_config.get("lattice"),
        ), None

    if name == "ppo":
        ppo = config["ppo"]
        return PPOEvalAgent(
            model_path=PROJECT_ROOT / ppo["model"],
            device=ppo.get("device", "cpu"),
            deterministic=ppo.get("deterministic", True),
            training_frames=ppo.get("training_frames", 0),
            training_seconds=ppo.get("training_seconds", 0.0),
        ), None

    raise ValueError(f"unknown agent {name!r}")
