"""Tests for the figure generation.

Plot code is easy to leave broken because nobody reads the output until report day. These
check the figures are produced, handle the degenerate cases, and do not crash on data shapes
the real run will actually contain -- an agent that never completed, a level with no failures,
a cell with a single episode.
"""

from __future__ import annotations

from src.eval import plots
from src.eval.metrics import DeathCause, EpisodeResult


def _result(agent="ppo", level="1-1", condition="deterministic", completed=True,
            max_x=3161, cause=DeathCause.NONE, compute=100) -> EpisodeResult:
    return EpisodeResult(
        agent=agent, level=level, condition=condition, episode=0, seed=0,
        completed=completed, max_x=max_x, frames=1000, death_cause=cause,
        compute_frames=compute,
    )


def _mixed() -> list[EpisodeResult]:
    return [
        _result("random", completed=False, max_x=594, cause=DeathCause.TIMEOUT, compute=0),
        _result("astar_open_loop", condition="sticky_0.25", completed=False,
                max_x=1787, cause=DeathCause.PIT, compute=71637),
        _result("astar_open_loop", completed=True, compute=71637),
        _result("astar_closed_loop", condition="sticky_0.25", completed=True, compute=90000),
        _result("ppo", completed=True, compute=20002688),
        _result("ppo", condition="sticky_0.25", completed=False, max_x=2100,
                cause=DeathCause.ENEMY, compute=20002688),
    ]


def test_completion_figure_is_written(tmp_path) -> None:
    """The headline figure."""
    path = tmp_path / "completion.png"
    plots.plot_completion(_mixed(), path, level="1-1")
    assert path.exists() and path.stat().st_size > 0


def test_distance_figure_is_written(tmp_path) -> None:
    """Where agents stop, not just whether they finish."""
    path = tmp_path / "distance.png"
    plots.plot_distance(_mixed(), path, level="1-1")
    assert path.exists() and path.stat().st_size > 0


def test_compute_figure_is_written(tmp_path) -> None:
    """Log axis, because the approaches differ by orders of magnitude."""
    path = tmp_path / "compute.png"
    plots.plot_compute_cost(_mixed(), path)
    assert path.exists() and path.stat().st_size > 0


def test_death_cause_figure_is_written(tmp_path) -> None:
    """How each agent failed."""
    path = tmp_path / "deaths.png"
    plots.plot_death_causes(_mixed(), path, level="1-1")
    assert path.exists() and path.stat().st_size > 0


def test_figures_skip_a_level_with_no_data(tmp_path) -> None:
    """Evaluating a subset of levels must not produce an empty or broken figure."""
    path = tmp_path / "missing.png"
    plots.plot_completion(_mixed(), path, level="8-4")
    assert not path.exists()


def test_death_figure_skips_when_nothing_failed(tmp_path) -> None:
    """A perfect run has no failure breakdown to draw."""
    path = tmp_path / "none.png"
    plots.plot_death_causes([_result(completed=True)], path, level="1-1")
    assert not path.exists()


def test_compute_figure_handles_a_zero_cost_agent(tmp_path) -> None:
    """The random baseline burns no planning frames; a log axis cannot plot zero."""
    path = tmp_path / "zero.png"
    plots.plot_compute_cost([_result("random", compute=0)], path)
    assert path.exists() and path.stat().st_size > 0


def test_single_episode_cell_does_not_break_error_bars(tmp_path) -> None:
    """n = 1 gives a zero-width binomial error bar, not a crash."""
    path = tmp_path / "one.png"
    plots.plot_completion([_result()], path, level="1-1")
    assert path.exists()
