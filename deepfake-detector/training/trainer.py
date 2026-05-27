"""
training/trainer.py
────────────────────
Production training loop for binary deepfake image classification.

MIXED PRECISION TRAINING
──────────────────────────
torch.cuda.amp (Automatic Mixed Precision) is enabled by default on GPU.

How it works:
  • Forward pass and loss computation run in float16 (half precision)
  • Backward pass (gradient computation) runs in float16
  • Gradient scaler multiplies loss before backward to prevent underflow
  • Scaler unscales gradients before optimizer.step()
  • Optimizer update (weight update) uses float32 master weights

Benefits for deepfake detection:
  1. Speed: ~1.5–2× faster training on modern GPUs (RTX 20xx/30xx/40xx)
     with Tensor Cores. Particularly significant for large images.
  2. Memory: ~40% less GPU memory → larger batch sizes → more stable gradients
  3. No accuracy loss: the scaler prevents the float16 underflow that would
     otherwise corrupt small gradient values

When AMP is disabled (CPU training or explicit opt-out):
  All operations run in float32. Training is slower but numerically identical.

GRADIENT SCALING
─────────────────
GradScaler is required for AMP because float16 has a limited dynamic range.
Small gradient values underflow to 0 in float16, causing silently incorrect
parameter updates (vanishing gradients).

Scaler strategy:
  1. Scale loss up by a large factor (default 65536) before backward pass
  2. This amplifies small gradients so they don't underflow in float16
  3. Before optimizer.step(), scaler divides gradients back to original scale
  4. If inf/nan is detected in gradients, scaler skips the update (that step)
     and halves the scale factor for subsequent steps

RESUME TRAINING SUPPORT
────────────────────────
The trainer saves complete state at every epoch (last.pth):
  • model weights
  • optimizer state (momentum buffers, adaptive LR estimates)
  • scheduler state (current LR, step count)
  • scaler state (current scale factor)
  • epoch number
  • best metric value
  • callback states

On resume, ALL of these are restored exactly, so training continues
as if it had never stopped. This is critical for long training runs
that may be interrupted by cloud instance preemption.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# torch.amp is the stable API from PyTorch 2.0+.
# torch.cuda.amp.GradScaler / autocast are deprecated since 2.0 and will be
# removed in a future release. The new API requires explicit device_type.
try:
    from torch.amp import GradScaler, autocast   # PyTorch >= 2.0
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch 1.x fallback
    _AMP_NEW_API = False

from training.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from training.optimizers import get_current_lr

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TrainingConfig:
    """
    Typed configuration container for the Trainer.

    Args:
        num_epochs:         Maximum number of training epochs.
        grad_clip_norm:     Max gradient norm for clipping. None = disabled.
                            1.0 is a safe default for fine-tuning.
        use_amp:            Enable mixed-precision training. Auto-disabled on CPU.
        log_every_n_steps:  Log batch metrics every N steps.
        phase1_epochs:      Epochs to train with frozen backbone (head warm-up).
                            0 = skip Phase 1, start fully unfrozen immediately.
        model_name:         Architecture registry key (stored in checkpoint).
    """

    def __init__(
        self,
        num_epochs: int = 30,
        grad_clip_norm: Optional[float] = 1.0,
        use_amp: bool = True,
        log_every_n_steps: int = 50,
        phase1_epochs: int = 2,
        model_name: str = "efficientnet_b0",
    ) -> None:
        self.num_epochs        = num_epochs
        self.grad_clip_norm    = grad_clip_norm
        self.use_amp           = use_amp
        self.log_every_n_steps = log_every_n_steps
        self.phase1_epochs     = phase1_epochs
        self.model_name        = model_name

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(epochs={self.num_epochs}, "
            f"amp={self.use_amp}, phase1={self.phase1_epochs})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Training orchestrator for deepfake image classification.

    Manages:
      • Training loop (forward, loss, backward, optimizer step)
      • Validation loop (no_grad, metric computation)
      • Mixed precision (AMP + GradScaler)
      • Gradient clipping
      • Two-phase training (frozen backbone → full fine-tuning)
      • Checkpoint save and resume
      • Callbacks (EarlyStopping, ModelCheckpoint, LRMonitor)
      • tqdm progress bars

    Args:
        model:          nn.Module to train (from model_registry.build()).
        optimizer:      Configured optimizer (from optimizers.build_optimizer()).
        loss_fn:        Loss function (from losses.build_loss()).
        scheduler:      LR scheduler or dict (from optimizers.build_scheduler()).
        scheduler_needs_metric: True if scheduler.step() needs a metric arg.
        checkpoint_cb:  ModelCheckpoint callback instance.
        early_stop_cb:  EarlyStopping callback instance.
        lr_monitor:     LearningRateMonitor callback instance.
        config:         TrainingConfig instance.
        device:         torch.device to train on.
        extra_config:   Arbitrary dict stored in checkpoints (data config, etc.).
    """

    def __init__(
        self,
        model:                  nn.Module,
        optimizer:              torch.optim.Optimizer,
        loss_fn:                nn.Module,
        scheduler,
        scheduler_needs_metric: bool,
        checkpoint_cb:          ModelCheckpoint,
        early_stop_cb:          EarlyStopping,
        lr_monitor:             LearningRateMonitor,
        config:                 TrainingConfig,
        device:                 torch.device,
        extra_config:           Optional[dict] = None,
    ) -> None:
        self.model                  = model.to(device)
        self.optimizer              = optimizer
        self.loss_fn                = loss_fn
        self.scheduler              = scheduler
        self.scheduler_needs_metric = scheduler_needs_metric
        self.checkpoint_cb          = checkpoint_cb
        self.early_stop_cb          = early_stop_cb
        self.lr_monitor             = lr_monitor
        self.config                 = config
        self.device                 = device
        self.extra_config           = extra_config or {}

        # Mixed precision: only active on CUDA, ignored on CPU.
        # GradScaler with device_type is the new API (PyTorch >= 2.0).
        # On CPU, AMP is silently disabled regardless of config.use_amp.
        self._amp_enabled  = config.use_amp and device.type == "cuda"
        self._device_type  = device.type  # stored for autocast calls

        if _AMP_NEW_API :
            self._scaler = GradScaler(enabled=self._amp_enabled)
        else:
            self._scaler = GradScaler(enabled=self._amp_enabled)

        # Training state
        self._start_epoch:    int         = 1
        self._metrics_history: list[dict] = []

        logger.info(
            "[trainer] Initialised  device=%s  amp=%s  model=%s  config=%s",
            device, self._amp_enabled, type(model).__name__, config,
        )

    # ── AUC computation (scikit-learn free) ──────────────────────────────────

    @staticmethod
    def _compute_auc(labels: list[int], probs: list[float]) -> float:
        """
        Compute AUC-ROC using the trapezoidal rule without scikit-learn.

        This implementation:
          1. Sorts predictions by probability (descending)
          2. Sweeps threshold from 1 to 0
          3. Computes TPR and FPR at each threshold
          4. Integrates using the trapezoidal rule

        Returns AUC in [0, 1]. 0.5 = random classifier, 1.0 = perfect.
        """
        n     = len(labels)
        n_pos = sum(labels)
        n_neg = n - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.0

        # Sort by predicted probability (descending)
        sorted_pairs = sorted(zip(probs, labels), reverse=True)

        tp, fp = 0, 0
        auc    = 0.0
        prev_fp = 0
        prev_tp = 0

        for _, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            # Trapezoid: area when FP changes
            if fp != prev_fp:
                auc    += (fp - prev_fp) * (tp + prev_tp) / 2.0
                prev_fp = fp
                prev_tp = tp

        # Final edge case
        auc += (n_neg - prev_fp) * (tp + prev_tp) / 2.0

        return auc / (n_pos * n_neg)

    # ── Checkpoint utilities ──────────────────────────────────────────────────

    def _build_checkpoint(self, epoch: int, best_val_metric: float) -> dict:
        """Build the complete checkpoint dict for serialisation."""
        scheduler_state = None
        if self.scheduler is not None:
            if isinstance(self.scheduler, dict):
                # cosine_warmup returns a dict of schedulers
                scheduler_state = {
                    k: (v.state_dict() if hasattr(v, "state_dict") else v)
                    for k, v in self.scheduler.items()
                }
            elif hasattr(self.scheduler, "state_dict"):
                scheduler_state = self.scheduler.state_dict()

        return {
            "model_state_dict":  self.model.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "scheduler_state":   scheduler_state,
            "scaler_state":      self._scaler.state_dict(),
            "epoch":             epoch,
            "best_val_metric":   best_val_metric,
            "config":            {
                "training": vars(self.config),
                **self.extra_config,
            },
            "model_name":        self.config.model_name,
            "pytorch_version":   torch.__version__,
            "metrics_history":   self._metrics_history,
            "callback_states": {
                "early_stopping":    self.early_stop_cb.state_dict(),
                "model_checkpoint":  self.checkpoint_cb.state_dict(),
                "lr_monitor":        self.lr_monitor.state_dict(),
            },
        }

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Restore full training state from a checkpoint file.

        Restores model weights, optimizer state, scheduler state, scaler state,
        epoch counter, metrics history, and all callback states.

        Call before fit() to resume an interrupted training run.

        Args:
            checkpoint_path: Path to the .pth checkpoint file (typically last.pth).
        """
        logger.info("[trainer] Loading checkpoint: %s", checkpoint_path)

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if ckpt.get("scaler_state") and self._amp_enabled:
            self._scaler.load_state_dict(ckpt["scaler_state"])

        if ckpt.get("scheduler_state") and self.scheduler is not None:
            if isinstance(self.scheduler, dict):
                for k, state in ckpt["scheduler_state"].items():
                    if k in self.scheduler and hasattr(self.scheduler[k], "load_state_dict"):
                        self.scheduler[k].load_state_dict(state)
            elif hasattr(self.scheduler, "load_state_dict"):
                self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self._start_epoch    = ckpt["epoch"] + 1
        self._metrics_history = ckpt.get("metrics_history", [])

        cb_states = ckpt.get("callback_states", {})
        if "early_stopping" in cb_states:
            self.early_stop_cb.load_state_dict(cb_states["early_stopping"])
        if "model_checkpoint" in cb_states:
            self.checkpoint_cb.load_state_dict(cb_states["model_checkpoint"])
        if "lr_monitor" in cb_states:
            self.lr_monitor.load_state_dict(cb_states["lr_monitor"])

        logger.info(
            "[trainer] Resumed from epoch %d  best_val=%s",
            self._start_epoch - 1,
            ckpt.get("best_val_metric", "N/A"),
        )

    # ── Training phase management ─────────────────────────────────────────────

    def _enter_phase2(self, epoch: int) -> None:
        """
        Unfreeze backbone and transition to Phase 2 (full fine-tuning).

        Saves 'phase1_end.pth' at the transition boundary so:
          1. Phase 1 (head-only) weights are preserved for ablation
          2. Training can be restarted from the frozen-head state if needed

        After unfreeze, the backbone participates in gradient updates using
        the lower LR already set in the optimizer's second parameter group
        (backbone_lr_multiplier applied in build_optimizer).
        """
        if hasattr(self.model, "unfreeze"):
            self.model.unfreeze()
            logger.info(
                "[trainer] Phase 2: backbone UNFROZEN at epoch %d  "
                "(backbone LR = optimizer param_group[1] LR)",
                epoch,
            )
        else:
            logger.warning(
                "[trainer] Model has no unfreeze() method. "
                "All parameters are assumed to be trainable already."
            )

        # Save phase boundary checkpoint using the callback's save_named method
        transition_ckpt = self._build_checkpoint(
            epoch=epoch - 1,                         # last completed frozen epoch
            best_val_metric=self.checkpoint_cb.best_value,
        )
        saved_path = self.checkpoint_cb.save_named("phase1_end.pth", transition_ckpt)
        logger.info("[trainer] Phase 1 end checkpoint saved: %s", saved_path)

    # ── Single-epoch training loop ────────────────────────────────────────────

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Run one full training epoch.

        Returns:
            Dict with keys: train_loss, train_acc
        """
        self.model.train()

        total_loss   = 0.0
        correct      = 0
        total        = 0
        step         = 0

        pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_tensors, batch_labels in pbar:
            batch_tensors = batch_tensors.to(self.device, non_blocking=True)
            batch_labels  = batch_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward pass with AMP ────────────────────────────────────────
            # autocast requires device_type in PyTorch 2.0+. The context
            # manager is a no-op when enabled=False (CPU training).
            with autocast(device_type=self._device_type, enabled=self._amp_enabled):
                logits = self.model(batch_tensors)
                loss   = self.loss_fn(logits, batch_labels)

            # ── Backward + optimizer step with gradient scaling ───────────────
            # scaler.scale() multiplies loss by current scale factor
            # scaler.unscale_() divides gradients back before clip/step
            # scaler.step() skips update if inf/nan gradients detected
            # scaler.update() adjusts scale factor for next iteration
            self._scaler.scale(loss).backward()

            if self.config.grad_clip_norm is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )

            self._scaler.step(self.optimizer)
            self._scaler.update()

            # ── Per-step warmup scheduler step ───────────────────────────────
            # cosine_warmup uses a LambdaLR for the warmup phase that must be
            # stepped once per BATCH (not per epoch) until warmup_steps is
            # reached. After that, the cosine scheduler takes over per epoch.
            if (
                isinstance(self.scheduler, dict)
                and "warmup" in self.scheduler
                and step < self.scheduler.get("warmup_steps", 0)
            ):
                self.scheduler["warmup"].step()

            # ── Accumulate metrics ───────────────────────────────────────────
            batch_size  = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total      += batch_size

            # Detach logits before argmax — we are in training mode and do not
            # want preds to participate in any downstream autograd graph.
            with torch.no_grad():
                preds   = logits.detach().argmax(dim=1)
                correct += (preds == batch_labels).sum().item()

            if step % self.config.log_every_n_steps == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc":  f"{correct / max(total, 1):.3f}",
                })

            step += 1

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {
            "train_loss": avg_loss,
            "train_acc":  accuracy,
        }

    # ── Validation loop ───────────────────────────────────────────────────────

    def _validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Run validation epoch.

        Computes: val_loss, val_acc, val_auc, val_f1

        Returns:
            Dict with validation metric values.
        """
        self.model.eval()

        total_loss = 0.0
        total      = 0
        all_labels: list[int]  = []
        all_probs:  list[float] = []
        all_preds:  list[int]  = []

        pbar = tqdm(
            val_loader,
            desc=f"Val   epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch_tensors, batch_labels in pbar:
                batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                batch_labels  = batch_labels.to(self.device, non_blocking=True)

                with autocast(device_type=self._device_type, enabled=self._amp_enabled):
                    logits = self.model(batch_tensors)
                    loss   = self.loss_fn(logits, batch_labels)

                batch_size  = batch_labels.size(0)
                total_loss += loss.item() * batch_size
                total      += batch_size

                probs  = torch.softmax(logits, dim=1)[:, 1]  # P(Fake)
                preds  = logits.argmax(dim=1)

                all_labels.extend(batch_labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(total, 1)

        # ── Compute metrics ───────────────────────────────────────────────────
        accuracy = sum(
            p == l for p, l in zip(all_preds, all_labels)
        ) / max(len(all_labels), 1)

        auc = self._compute_auc(all_labels, all_probs)

        # F1 for the positive (Fake) class
        tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

        return {
            "val_loss": avg_loss,
            "val_acc":  accuracy,
            "val_auc":  auc,
            "val_f1":   f1,
        }

    # ── Scheduler step ────────────────────────────────────────────────────────

    def _scheduler_step(self, metrics: dict[str, float]) -> None:
        """Step the LR scheduler after each epoch."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, dict):
            # cosine_warmup: step the cosine scheduler per epoch
            # warmup is handled per-step in the training loop
            cosine = self.scheduler.get("cosine")
            if cosine is not None:
                cosine.step()
        elif self.scheduler_needs_metric:
            # ReduceLROnPlateau — pass val_auc or val_loss
            monitor_key = "val_auc" if "val_auc" in metrics else "val_loss"
            self.scheduler.step(metrics[monitor_key])
            new_lr = get_current_lr(self.optimizer)
            logger.info("[trainer] Plateau scheduler stepped  new_lr=%.2e", new_lr)
        else:
            self.scheduler.step()

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> list[dict[str, float]]:
        """
        Run the full training and validation loop.

        Implements:
          Phase 1 (if config.phase1_epochs > 0):
            Backbone frozen, only the classification head trains.
            Runs for config.phase1_epochs epochs.

          Phase 2:
            Backbone unfrozen, all layers train with differential LR.
            Continues until num_epochs is reached or EarlyStopping triggers.

        Args:
            train_loader: Training DataLoader from DeepfakeDataModule.
            val_loader:   Validation DataLoader from DeepfakeDataModule.

        Returns:
            List of per-epoch metric dicts (full training history).
        """
        logger.info(
            "[trainer] Starting training  epochs=%d  start=%d  "
            "phase1=%d  device=%s  amp=%s",
            self.config.num_epochs,
            self._start_epoch,
            self.config.phase1_epochs,
            self.device,
            self._amp_enabled,
        )

        for epoch in range(self._start_epoch, self.config.num_epochs + 1):
            epoch_start = time.perf_counter()

            # ── Phase transition ─────────────────────────────────────────────
            if (
                self.config.phase1_epochs > 0
                and epoch == self.config.phase1_epochs + 1
            ):
                self._enter_phase2(epoch)

            # ── Train + validate ─────────────────────────────────────────────
            train_metrics = self._train_one_epoch(train_loader, epoch)
            val_metrics   = self._validate(val_loader, epoch)

            # ── Merge metrics ────────────────────────────────────────────────
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}

            # ── LR monitor ───────────────────────────────────────────────────
            self.lr_monitor.on_epoch_end(epoch, self.optimizer, metrics)

            # ── LR scheduler step ────────────────────────────────────────────
            self._scheduler_step(metrics)

            # ── Log epoch summary ─────────────────────────────────────────────
            elapsed = time.perf_counter() - epoch_start
            logger.info(
                "[trainer] Epoch %d/%d  "
                "train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f  val_auc=%.4f  val_f1=%.4f  "
                "lr=%.2e  time=%.1fs",
                epoch, self.config.num_epochs,
                metrics["train_loss"], metrics["train_acc"],
                metrics["val_loss"],   metrics["val_acc"],
                metrics["val_auc"],    metrics["val_f1"],
                metrics.get("lr", get_current_lr(self.optimizer)),
                elapsed,
            )

            self._metrics_history.append(metrics)

            # ── Checkpoint callback ───────────────────────────────────────────
            checkpoint_data = self._build_checkpoint(
                epoch, self.checkpoint_cb.best_value
            )
            self.checkpoint_cb.on_epoch_end(epoch, metrics, checkpoint_data)

            # ── Early stopping callback ───────────────────────────────────────
            if self.early_stop_cb.on_epoch_end(epoch, metrics):
                logger.info(
                    "[trainer] Early stopping triggered at epoch %d. "
                    "Best %s = %.6f @ epoch %d",
                    epoch,
                    self.early_stop_cb.monitor,
                    self.early_stop_cb.best_value,
                    self.checkpoint_cb.best_epoch,
                )
                break

        logger.info(
            "[trainer] Training complete.  "
            "Best val_auc=%.4f  best checkpoint: %s",
            self.checkpoint_cb.best_value,
            self.checkpoint_cb.best_path,
        )

        return self._metrics_history


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "Trainer",
    "TrainingConfig",
]