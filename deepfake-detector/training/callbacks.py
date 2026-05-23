"""
training/callbacks.py
─────────────────────
Training callbacks for EarlyStopping, checkpoint management, and LR monitoring.

Callback design
────────────────
Each callback is a plain class with hook methods that the trainer calls at
specific points. No base class is required — duck typing is sufficient for
a small, focused callback set. All callbacks are stateful and serialisable
(their state is saved as part of the trainer checkpoint for resume support).

All callbacks follow the same interface:
  on_epoch_end(epoch, metrics) → optional action
  state_dict()                 → dict for serialisation
  load_state_dict(d)           → restore from serialised dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Prevents wasting compute on epochs that no longer improve generalisation.
    Particularly important for deepfake detection where overfitting on
    training-domain deepfakes is a significant risk — the model can achieve
    near-100% train AUC while val AUC plateaus or drops.

    Args:
        monitor:   Metric key to monitor. Use "val_auc" (primary metric for
                   deepfake detection — more informative than accuracy on
                   imbalanced datasets). "val_loss" is a valid alternative.
        patience:  Number of epochs to wait after the last improvement.
                   Recommended: 5–10 epochs. Too low = stops before convergence.
                   Too high = wastes compute.
        min_delta: Minimum change to qualify as an improvement.
                   0.001 means a val_auc increase of < 0.1% is treated as
                   no improvement. Prevents stopping on noise.
        mode:      "max" when higher is better (AUC, F1, accuracy).
                   "min" when lower is better (loss).
        verbose:   Log a message on every epoch (including non-improvement).
    """

    def __init__(
        self,
        monitor: str = "val_auc",
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.monitor   = monitor
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.verbose   = verbose

        # Internal state
        self._best_value:       float = float("-inf") if mode == "max" else float("inf")
        self._epochs_no_improve: int  = 0
        self._stopped_epoch:     int  = 0
        self.should_stop:        bool = False

        logger.info(
            "[early_stopping] monitor=%s  patience=%d  min_delta=%.4f  mode=%s",
            monitor, patience, min_delta, mode,
        )

    def _is_improvement(self, current: float) -> bool:
        """Return True if current value represents a meaningful improvement."""
        if self.mode == "max":
            return current >= self._best_value + self.min_delta
        else:
            return current <= self._best_value - self.min_delta

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """
        Evaluate whether to continue training.

        Args:
            epoch:   Current epoch number (1-indexed).
            metrics: Dict of metric name → value for this epoch.

        Returns:
            True if training should STOP, False to continue.
        """
        if self.monitor not in metrics:
            logger.warning(
                "[early_stopping] Metric '%s' not found in metrics dict. "
                "Keys: %s. Skipping check.",
                self.monitor, list(metrics.keys()),
            )
            return False

        current = metrics[self.monitor]

        if self._is_improvement(current):
            if self.verbose:
                logger.info(
                    "[early_stopping] %s improved: %.6f → %.6f  "
                    "(patience counter reset)",
                    self.monitor, self._best_value, current,
                )
            self._best_value        = current
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
            if self.verbose:
                logger.info(
                    "[early_stopping] %s did not improve (%.6f).  "
                    "No improvement for %d / %d epochs.",
                    self.monitor, current,
                    self._epochs_no_improve, self.patience,
                )

        if self._epochs_no_improve >= self.patience:
            self._stopped_epoch = epoch
            self.should_stop    = True
            logger.info(
                "[early_stopping] Triggered at epoch %d. "
                "Best %s = %.6f",
                epoch, self.monitor, self._best_value,
            )
            return True

        return False

    @property
    def best_value(self) -> float:
        return self._best_value

    def state_dict(self) -> dict:
        return {
            "monitor":           self.monitor,
            "patience":          self.patience,
            "min_delta":         self.min_delta,
            "mode":              self.mode,
            "best_value":        self._best_value,
            "epochs_no_improve": self._epochs_no_improve,
            "stopped_epoch":     self._stopped_epoch,
            "should_stop":       self.should_stop,
        }

    def load_state_dict(self, d: dict) -> None:
        self.monitor              = d["monitor"]
        self.patience             = d["patience"]
        self.min_delta            = d["min_delta"]
        self.mode                 = d["mode"]
        self._best_value          = d["best_value"]
        self._epochs_no_improve   = d["epochs_no_improve"]
        self._stopped_epoch       = d["stopped_epoch"]
        self.should_stop          = d["should_stop"]

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(monitor={self.monitor!r}, patience={self.patience}, "
            f"best={self._best_value:.4f}, no_improve={self._epochs_no_improve})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Saves two checkpoints:
      1. best.pth — the best model according to the monitored metric.
                    Overwritten whenever a new best is found.
      2. last.pth — the most recent completed epoch.
                    Always overwritten. Used for training resumption.

    Checkpoint format (consistent with trainer.py and detector.py):
      {
          "model_state_dict":  model.state_dict(),
          "optimizer_state":   optimizer.state_dict(),
          "scheduler_state":   scheduler.state_dict() or None,
          "scaler_state":      scaler.state_dict(),
          "epoch":             epoch,
          "best_val_metric":   float,
          "config":            dict,
          "model_name":        str,
          "pytorch_version":   str,
          "metrics_history":   list[dict],
      }

    Args:
        checkpoint_dir: Directory to save checkpoints. Created if missing.
        monitor:        Metric to track for best-model selection.
        mode:           "max" (AUC, accuracy) or "min" (loss).
        save_last:      Always save last.pth after every epoch.
        verbose:        Log a message when a new best is saved.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_auc",
        mode: str = "max",
        save_last: bool = True,
        verbose: bool = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor        = monitor
        self.mode           = mode
        self.save_last      = save_last
        self.verbose        = verbose

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_value: float = float("-inf") if mode == "max" else float("inf")
        self._best_epoch: int   = 0

        logger.info(
            "[checkpoint] dir=%s  monitor=%s  mode=%s",
            checkpoint_dir, monitor, mode,
        )

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "best.pth"

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / "last.pth"

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self._best_value
        else:
            return current < self._best_value

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_data: dict,
    ) -> bool:
        """
        Save checkpoints based on the epoch metrics.

        Args:
            epoch:           Current epoch number (1-indexed).
            metrics:         Dict of metric name → value.
            checkpoint_data: Dict ready to pass to torch.save().
                             Must contain all required keys (see class docstring).

        Returns:
            True if a new best was saved, False otherwise.
        """
        saved_best = False

        if self.monitor in metrics and self._is_improvement(metrics[self.monitor]):
            self._best_value = metrics[self.monitor]
            self._best_epoch = epoch
            saved_best       = True

            torch.save(checkpoint_data, self.best_path)

            if self.verbose:
                logger.info(
                    "[checkpoint] ✓ New best at epoch %d  "
                    "%s=%.6f  saved → %s",
                    epoch, self.monitor, self._best_value, self.best_path,
                )

        if self.save_last:
            torch.save(checkpoint_data, self.last_path)
            logger.debug("[checkpoint] last.pth saved (epoch %d)", epoch)

        return saved_best

    def save_named(self, name: str, checkpoint_data: dict) -> Path:
        """
        Save a named checkpoint (e.g. 'phase1_end.pth') to checkpoint_dir.

        Used by the trainer to mark phase transitions (frozen → unfrozen).
        """
        path = self.checkpoint_dir / name
        torch.save(checkpoint_data, path)
        logger.info("[checkpoint] Named checkpoint saved: %s", path)
        return path

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    def state_dict(self) -> dict:
        return {
            "monitor":     self.monitor,
            "mode":        self.mode,
            "best_value":  self._best_value,
            "best_epoch":  self._best_epoch,
        }

    def load_state_dict(self, d: dict) -> None:
        self.monitor     = d["monitor"]
        self.mode        = d["mode"]
        self._best_value = d["best_value"]
        self._best_epoch = d["best_epoch"]

    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint(dir={self.checkpoint_dir}, "
            f"monitor={self.monitor!r}, best={self._best_value:.4f} "
            f"@ epoch {self._best_epoch})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class LearningRateMonitor:
    """
    Log learning rates from all optimizer parameter groups each epoch.

    In differential-LR setups (backbone LR ≠ head LR), this logs both values
    so you can verify the backbone is being trained slower than the head.

    Also tracks LR history for post-training analysis and debug plots.

    Args:
        log_every_n_epochs: Log every N epochs. Default 1 (every epoch).
    """

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        self.log_every_n_epochs = log_every_n_epochs
        self._history: list[dict[str, float]] = []

    def on_epoch_end(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Log and record current LRs from all parameter groups.

        Args:
            epoch:     Current epoch number.
            optimizer: The optimizer being used.
            metrics:   Current epoch metrics dict (will have LR values added).

        Returns:
            Dict of {"lr_group_0": float, "lr_group_1": float, ...}
        """
        if epoch % self.log_every_n_epochs != 0:
            return {}

        lr_dict: dict[str, float] = {}

        for i, group in enumerate(optimizer.param_groups):
            key = f"lr" if i == 0 else f"lr_group_{i}"
            lr_dict[key] = group["lr"]

        self._history.append({"epoch": epoch, **lr_dict})

        # Build log message
        lr_str = "  ".join(
            f"{k}={v:.2e}" for k, v in lr_dict.items()
        )
        logger.info("[lr_monitor] epoch=%d  %s", epoch, lr_str)

        # Inject into metrics so callbacks and trainer can access LR values
        metrics.update(lr_dict)

        return lr_dict

    @property
    def history(self) -> list[dict[str, float]]:
        """Return full LR history as a list of per-epoch dicts."""
        return list(self._history)

    def state_dict(self) -> dict:
        return {"history": self._history}

    def load_state_dict(self, d: dict) -> None:
        self._history = d.get("history", [])

    def __repr__(self) -> str:
        return (
            f"LearningRateMonitor("
            f"log_every={self.log_every_n_epochs}, "
            f"recorded_epochs={len(self._history)})"
        )


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateMonitor",
]