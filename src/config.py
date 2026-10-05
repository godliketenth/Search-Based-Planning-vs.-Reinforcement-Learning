"""YAML configuration loading.

Project convention: every tunable lives in a YAML file under ``configs/``, and scripts in
``scripts/`` load one and may override individual keys from the command line. No script
hard-codes a hyperparameter, so a reported result can always be traced back to a config
file plus a logged set of overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Repository root, resolved from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Directory holding the YAML configs.
CONFIG_DIR = PROJECT_ROOT / "configs"

#: Directories results are written to. Nothing here is ever assembled by hand.
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply command-line overrides.

    Args:
        path: path to the YAML file. A bare name (``"env.yaml"``) is resolved inside
            ``configs/``, so scripts can accept either form.
        overrides: keys to override. Entries whose value is ``None`` are ignored, which
            lets a script pass its parsed ``argparse`` namespace straight through: a flag
            the user did not set leaves the config value alone.

    Returns:
        The merged config as a plain dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if an override names a key the config does not define -- a typo'd flag
            should fail loudly rather than be silently ignored.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open() as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in config:
            raise KeyError(
                f"override {key!r} is not a key in {path.name}; "
                f"known keys: {sorted(config)}"
            )
        config[key] = value
    return config
