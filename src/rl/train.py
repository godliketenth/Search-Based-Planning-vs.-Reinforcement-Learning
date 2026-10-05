"""PPO training, with checkpointing, resume and TensorBoard.

PPO is the counterpart to the planner in this project, and the contrast is the point:

* A* is given a **perfect model** of the dynamics (the emulator itself) and privileged state
  (RAM), and pays its entire cost at plan time -- about 54 s for 1-1.
* PPO is given **pixels only** and no model, has to discover the dynamics by trial and
  error, and pays its cost at training time -- millions of frames.

Neither is "better"; they buy different things with different currencies, and the results
table has to state that asymmetry rather than hide it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

#: Filename stem for checkpoints, so resume can find and order them.
CHECKPOINT_PREFIX = "ppo_mario"


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Where a run's artefacts live.

    Attributes:
        run_dir: root directory for this run.
        checkpoints: periodic checkpoints, for resume.
        tensorboard: TensorBoard event files.
        final_model: the model saved at the end of training.
    """

    run_dir: Path

    @property
    def checkpoints(self) -> Path:
        """Directory holding periodic checkpoints."""
        return self.run_dir / "checkpoints"

    @property
    def tensorboard(self) -> Path:
        """Directory holding TensorBoard logs."""
        return self.run_dir / "tensorboard"

    @property
    def final_model(self) -> Path:
        """Path of the final saved model."""
        return self.run_dir / "final_model.zip"

    def ensure(self) -> None:
        """Create the directories."""
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.tensorboard.mkdir(parents=True, exist_ok=True)


class OutcomeCallback(BaseCallback):
    """Log task outcomes -- not just reward -- to TensorBoard.

    Episode return is a poor description of progress here: a run that dies late with a high
    score and a run that completes the level can look similar. What matters for the Phase 5
    table is how far Mario got and whether he reached the flag, so those are logged
    directly.
    """

    def __init__(self, window: int = 100) -> None:
        """Track a rolling window of completed episodes."""
        super().__init__()
        self.window = window
        self.max_x_history: list[int] = []
        self.flag_history: list[bool] = []
        self.best_x = 0
        self.flags = 0

    def _on_step(self) -> bool:
        """Harvest ``episode_extra`` from any episode that ended this step."""
        for info in self.locals.get("infos", []):
            extra = info.get("episode_extra")
            if extra is None:
                continue
            self.max_x_history.append(extra["max_x"])
            self.flag_history.append(extra["flag_get"])
            self.best_x = max(self.best_x, extra["max_x"])
            self.flags += int(extra["flag_get"])
            del self.max_x_history[: -self.window]
            del self.flag_history[: -self.window]

        if self.max_x_history:
            self.logger.record(
                "mario/mean_max_x", sum(self.max_x_history) / len(self.max_x_history)
            )
            self.logger.record(
                "mario/completion_rate", sum(self.flag_history) / len(self.flag_history)
            )
            self.logger.record("mario/best_x", self.best_x)
            self.logger.record("mario/flags_total", self.flags)
        return True


def resolve_device(requested: str) -> str:
    """Pick a torch device, honouring a request but falling back cleanly.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"`` or ``"cuda"``.

    Returns:
        A device string torch can use.

    Note:
        ``"auto"`` prefers **mps** on Apple Silicon. The project's original expectation was
        that MPS would lose to CPU for an 84x84 NatureCNN, on the grounds that the kernels
        are small and launch overhead would dominate. Measured, it is the other way round:
        3.6x faster end-to-end (672 vs 187 agent steps/s). The mechanism is contention --
        eight emulator workers already saturate the CPU, so running the gradient step there
        too makes them compete, whereas MPS offloads it to the GPU and leaves the cores for
        emulation. See ``docs/phase4-findings.md``.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def latest_checkpoint(directory: Path) -> Path | None:
    """Find the checkpoint with the highest step count.

    Args:
        directory: the checkpoints directory.

    Returns:
        The newest checkpoint path, or ``None`` if there are none.
    """
    if not directory.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"):
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates)[1]


def build_model(venv: VecEnv, config: dict[str, Any], paths: TrainingPaths) -> PPO:
    """Construct a fresh PPO model from config.

    Args:
        venv: the vectorised training environment.
        config: hyperparameters from ``configs/ppo.yaml``.
        paths: where to write TensorBoard logs.

    Returns:
        An untrained :class:`PPO`.
    """
    return PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        tensorboard_log=str(paths.tensorboard),
        device=resolve_device(config.get("device", "auto")),
        verbose=1,
        seed=config.get("seed", 0),
    )


def train(
    venv: VecEnv,
    config: dict[str, Any],
    paths: TrainingPaths,
    total_timesteps: int,
    resume: bool = False,
) -> PPO:
    """Train PPO, optionally resuming from the newest checkpoint.

    Resuming preserves the step counter (``reset_num_timesteps=False``), so TensorBoard
    curves stay continuous across interruptions and ``total_timesteps`` means the same
    thing whether or not the run was restarted.

    Args:
        venv: the vectorised training environment.
        config: hyperparameters.
        paths: artefact locations.
        total_timesteps: additional timesteps to run.
        resume: load the newest checkpoint before training.

    Returns:
        The trained model.
    """
    paths.ensure()
    checkpoint = latest_checkpoint(paths.checkpoints) if resume else None

    if checkpoint is not None:
        print(f"resuming from {checkpoint.name}")
        model = PPO.load(
            checkpoint,
            env=venv,
            device=resolve_device(config.get("device", "auto")),
            tensorboard_log=str(paths.tensorboard),
        )
    else:
        if resume:
            print("no checkpoint found; starting a fresh run")
        model = build_model(venv, config, paths)

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, config["checkpoint_every"] // venv.num_envs),
            save_path=str(paths.checkpoints),
            name_prefix=CHECKPOINT_PREFIX,
        ),
        OutcomeCallback(),
    ]
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=checkpoint is None,
        progress_bar=False,
    )
    model.save(paths.final_model)
    return model
