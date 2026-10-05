"""Evaluation: one harness, one set of metrics, every agent.

The loop in :mod:`src.eval.harness` constructs the environment, steps it, counts frames and
records metrics identically for every agent, which is what makes the Phase 5 comparison a
comparison rather than two separate demonstrations.
"""

from .harness import run_cell, run_episode, write_results_csv, write_summary_csv
from .metrics import DeathCause, EpisodeResult, classify_death, summarise

__all__ = [
    "DeathCause",
    "EpisodeResult",
    "classify_death",
    "run_cell",
    "run_episode",
    "summarise",
    "write_results_csv",
    "write_summary_csv",
]
