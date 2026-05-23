"""
training/optimizers.py
───────────────────────
Configurable optimizer and learning rate scheduler factory.

All functions return standard PyTorch optimizer/scheduler objects —
no custom subclasses — so they work with torch.save/load transparently.

OPTIMIZER SELECTION GUIDE FOR DEEPFAKE DETECTION
─────────────────────────────────────────────────

AdamW (recommended default):
  AdamW decouples weight decay from the gradient update, which is the
  correct implementation of L2 regularisation for adaptive optimisers.
  Original Adam applies weight decay incorrectly (it interacts with the
  adaptive learning rate). AdamW fixes this, producing better regularisation.

  Use AdamW when:
    • Fine-tuning a pretrained backbone (the standard case here)
    • You want fast convergence with good generalisation
    • Learning rate is in range 1e-4 to 3e-4

SGD with Momentum:
  SGD typically generalises better than Adam on vision tasks when training
  from scratch, because the noise in the gradient estimate acts as regularisation.
  However, it requires careful learning rate tuning and warm-up to converge.

  Use SGD when:
    • Training from scratch (not applicable here — we always use pretrained)
    • You want to reproduce published benchmark results (many papers use SGD)
    • You have time for careful LR search

SCHEDULER SELECTION GUIDE
──────────────────────────

CosineAnnealingLR (recommended default):
  Smoothly decays LR from the initial value to near-zero following a cosine
  curve. Avoids the abrupt LR drops of StepLR/MultiStepLR that can cause
  training instability. Works well with early stopping — the model converges
  smoothly toward the end of the schedule.

ReduceLROnPlateau:
  Reduces LR by a factor when a metric (typically val_auc) stops improving.
  Excellent for long training runs where the model may plateau multiple times.
  More conservative than CosineAnnealing — only drops LR when needed.
  Pairs naturally with EarlyStopping.

WARMUP STRATEGY
────────────────
Linear warmup increases the LR from near-zero to the target LR over a fixed
number of steps. This is critical when:
  1. Fine-tuning with the backbone unfrozen from the start — the first few
     batches with large LR can corrupt pretrained backbone weights.
  2. Using large batch sizes — the effective gradient variance is lower and
     the LR should ramp up gradually.

Without warmup:
  First epoch with backbone unfrozen + LR=3e-4 can produce loss spikes that
  permanently damage the pretrained feature representations.

With warmup (500 steps typical):
  LR ramps from 1e-7 to 3e-4 over 500 steps, giving the model time to
  stabilise the new binary head before applying large backbone gradients.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    backbone_lr_multiplier: float = 0.1,
    momentum: float = 0.9,
    nesterov: bool = True,
) -> Optimizer:
    """
    Build and return a configured optimizer.

    Supports differential learning rates: the backbone uses a lower LR
    than the classifier head. This is critical for Phase 2 fine-tuning —
    the pretrained backbone should be updated slowly (LR × 0.1) while
    the freshly-initialised head uses the full LR.

    Args:
        model:                  The nn.Module to optimise.
        name:                   "adamw" or "sgd".
        lr:                     Base learning rate for the classifier head.
        weight_decay:           L2 regularisation coefficient.
                                1e-4 is a good default for fine-tuning.
        backbone_lr_multiplier: Multiplier applied to the backbone LR.
                                0.1 means backbone LR = lr × 0.1.
                                Set to 1.0 for uniform LR across all layers.
        momentum:               SGD momentum coefficient (ignored for AdamW).
        nesterov:               Use Nesterov momentum for SGD (ignored for AdamW).
                                Typically improves SGD convergence.

    Returns:
        Configured PyTorch Optimizer instance.

    Raises:
        ValueError: if name is not recognised.
    """
    # Build parameter groups with differential LRs
    # This handles both EfficientNetB0Classifier and any nn.Module with
    # a 'backbone' attribute and a 'backbone.classifier' or equivalent.
    param_groups = _build_param_groups(model, lr, weight_decay, backbone_lr_multiplier)

    name = name.lower().strip()

    if name == "adamw":
        optimizer = AdamW(param_groups)
        logger.info(
            "[optimizer] AdamW  lr=%.2e  backbone_lr=%.2e  wd=%.2e",
            lr, lr * backbone_lr_multiplier, weight_decay,
        )
        return optimizer

    elif name == "sgd":
        optimizer = SGD(
            param_groups,
            momentum=momentum,
            nesterov=nesterov,
        )
        logger.info(
            "[optimizer] SGD  lr=%.2e  momentum=%.2f  nesterov=%s",
            lr, momentum, nesterov,
        )
        return optimizer

    else:
        raise ValueError(
            f"Unknown optimizer: {name!r}. Available: 'adamw', 'sgd'."
        )


def _build_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_multiplier: float,
) -> list[dict]:
    """
    Build parameter groups with differential learning rates.

    If the model has a 'backbone' attribute with a 'features' submodule
    (EfficientNetB0Classifier structure), backbone parameters get
    lr * backbone_lr_multiplier. All other parameters get the full lr.

    If the model does not have this structure, all parameters use lr uniformly.
    """
    backbone_params = []
    head_params     = []

    has_differential = (
        hasattr(model, "backbone")
        and hasattr(model.backbone, "features")
    )

    if has_differential:
        backbone_param_ids = {
            id(p) for p in model.backbone.features.parameters()
        }
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_param_ids:
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups = [
            {
                "params":       head_params,
                "lr":           lr,
                "weight_decay": weight_decay,
            },
            {
                "params":       backbone_params,
                "lr":           lr * backbone_lr_multiplier,
                "weight_decay": weight_decay,
            },
        ]
        logger.debug(
            "[optimizer] Differential LR: head=%d params @ %.2e, "
            "backbone=%d params @ %.2e",
            len(head_params), lr,
            len(backbone_params), lr * backbone_lr_multiplier,
        )
    else:
        param_groups = [
            {
                "params":       [p for p in model.parameters() if p.requires_grad],
                "lr":           lr,
                "weight_decay": weight_decay,
            }
        ]

    return param_groups


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    num_epochs: int = 30,
    warmup_steps: int = 500,
    min_lr: float = 1e-7,
    # ReduceLROnPlateau specific
    plateau_factor: float = 0.5,
    plateau_patience: int = 3,
    plateau_min_lr: float = 1e-7,
    plateau_mode: str = "max",
) -> tuple:
    """
    Build and return a learning rate scheduler.

    Args:
        optimizer:        The optimizer whose LR is being scheduled.
        name:             "cosine", "plateau", "cosine_warmup".
        num_epochs:       Total training epochs (for CosineAnnealingLR T_max).
        warmup_steps:     Number of STEPS (not epochs) for linear warmup.
                          Only used by "cosine_warmup".
        min_lr:           Minimum LR for cosine annealing.
        plateau_factor:   Factor to multiply LR by on plateau (for "plateau").
        plateau_patience: Epochs without improvement before LR reduction.
        plateau_mode:     "max" for AUC (higher = better), "min" for loss.
        plateau_min_lr:   Minimum LR floor for plateau scheduler.

    Returns:
        Tuple of (scheduler, requires_metric_step) where:
          scheduler:            The LR scheduler instance.
          requires_metric_step: True if scheduler.step() needs a metric arg
                                (ReduceLROnPlateau), False otherwise.

    Raises:
        ValueError: if name is not recognised.
    """
    name = name.lower().strip()

    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )
        logger.info(
            "[scheduler] CosineAnnealingLR  T_max=%d  eta_min=%.2e",
            num_epochs, min_lr,
        )
        return scheduler, False

    elif name == "cosine_warmup":
        # Phase 1: linear warmup from near-zero to initial LR
        # Phase 2: cosine annealing from initial LR to min_lr
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=_make_warmup_lambda(warmup_steps),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - 1),
            eta_min=min_lr,
        )
        # SequentialLR switches from warmup to cosine after warmup_steps
        # milestones=[1] means: switch after 1 call to warmup_scheduler.step()
        # Since we step per batch during warmup, warmup_steps steps occur
        # before handing off to the cosine scheduler.
        # Note: We handle the warmup→cosine switch manually in trainer.py
        # for clarity. Returning both schedulers in a dict.
        logger.info(
            "[scheduler] CosineAnnealingLR + linear warmup  "
            "warmup_steps=%d  T_max=%d  eta_min=%.2e",
            warmup_steps, num_epochs, min_lr,
        )
        return {
            "warmup":  warmup_scheduler,
            "cosine":  cosine_scheduler,
            "warmup_steps": warmup_steps,
        }, False

    elif name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=plateau_min_lr,
            verbose=False,  # we log manually
        )
        logger.info(
            "[scheduler] ReduceLROnPlateau  mode=%s  factor=%.2f  "
            "patience=%d  min_lr=%.2e",
            plateau_mode, plateau_factor, plateau_patience, plateau_min_lr,
        )
        return scheduler, True  # requires metric argument to .step()

    else:
        raise ValueError(
            f"Unknown scheduler: {name!r}. "
            f"Available: 'cosine', 'cosine_warmup', 'plateau'."
        )


def _make_warmup_lambda(warmup_steps: int):
    """
    Return a lambda for linear warmup over warmup_steps steps.

    LR at step t = (t / warmup_steps) * initial_LR  for t < warmup_steps
    LR at step t = initial_LR                        for t >= warmup_steps
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def get_current_lr(optimizer: Optimizer) -> float:
    """
    Return the current learning rate from the first parameter group.

    Used for logging. In differential-LR setups, this returns the head LR.
    """
    return optimizer.param_groups[0]["lr"]


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "build_optimizer",
    "build_scheduler",
    "get_current_lr",
]