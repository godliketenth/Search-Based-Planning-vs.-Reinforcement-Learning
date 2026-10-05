"""Tests for YAML config loading.

The project convention is that no script hard-codes a hyperparameter, so a result can
always be traced to a config file plus logged overrides. That only holds if overrides
behave predictably -- in particular if a mistyped flag fails instead of being ignored.
"""

from __future__ import annotations

import pytest

from src.config import CONFIG_DIR, load_config


def test_loads_the_env_config_by_bare_name() -> None:
    """Scripts pass 'env.yaml'; it must resolve inside configs/."""
    config = load_config("env.yaml")
    assert config["level"] == "1-1"
    assert config["sticky_p"] == 0.0
    assert set(config) >= {"level", "seed", "sticky_p", "max_episode_steps", "render_mode"}


def test_loads_by_explicit_path() -> None:
    """An explicit path must work too, for ad-hoc configs outside configs/."""
    assert load_config(CONFIG_DIR / "env.yaml")["level"] == "1-1"


def test_overrides_replace_config_values() -> None:
    """A CLI flag must win over the file."""
    config = load_config("env.yaml", {"level": "1-2", "sticky_p": 0.25})
    assert config["level"] == "1-2"
    assert config["sticky_p"] == 0.25


def test_none_overrides_are_ignored() -> None:
    """An unset argparse flag is None and must leave the config value alone."""
    config = load_config("env.yaml", {"level": None, "seed": None})
    assert config["level"] == "1-1"
    assert config["seed"] == 0


def test_unknown_override_key_raises() -> None:
    """A typo'd override must fail loudly rather than silently do nothing."""
    with pytest.raises(KeyError, match="levle"):
        load_config("env.yaml", {"levle": "1-2"})


def test_missing_config_file_raises() -> None:
    """A wrong --config path should not fall back to defaults."""
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
