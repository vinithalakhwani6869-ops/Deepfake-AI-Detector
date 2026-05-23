"""
models/efficientnet.py
──────────────────────
EfficientNet binary classifier for deepfake detection.

WHY EFFICIENTNET-B0 INITIALLY
──────────────────────────────
EfficientNet-B0 was chosen as the initial architecture for three concrete reasons:

1. Compound scaling efficiency.
   EfficientNet scales width, depth, and resolution jointly using a fixed
   coefficient. B0 sits at the Pareto frontier: it achieves higher accuracy
   per FLOP than ResNet-50 (comparable accuracy at ~5× fewer parameters).
   For a forensic classification task where the discriminative signals are
   subtle (frequency artifacts, blending seams), a parameter-efficient backbone
   avoids wasting capacity on low-level texture reconstruction.

2. Proven deepfake detection track record.
   B4/B7 variants dominated the DFDC Kaggle leaderboard (2020). B0 inherits
   the same architectural pattern at lower cost — a practical starting point
   before scaling to B4 once the training pipeline is validated.

3. Deployment compatibility.
   B0's 5.3M parameters and 224×224 input fit comfortably in 4GB GPU RAM with
   batch_size=64+. B4 at 19M parameters and 380×380 requires >8GB for the same
   batch size. B0 allows local development on consumer hardware while B4 runs
   on a production GPU server — same codebase, different config.

TRANSFER LEARNING DECISIONS
────────────────────────────
All EfficientNet weights available from torchvision were pretrained on ImageNet-1K.
The backbone (features extractor) learns generic visual representations:
  • Early layers: edges, gradients, textures
  • Middle layers: object parts, frequency patterns
  • Late layers: semantic concepts

For deepfake detection, the frequency pattern detectors in the middle layers
are directly useful — GAN-generated images leave systematic frequency artifacts
that align with what these layers are already sensitive to.

We therefore use a two-phase training strategy:
  Phase 1 (frozen backbone, free head):
    Trains only the new binary classifier head for 1–3 epochs.
    Reason: the random init on the new head would produce large gradients that
    would damage the pretrained backbone weights if everything is unfrozen from
    the start. Warming up the head first stabilises the gradient flow.

  Phase 2 (full fine-tuning, all layers):
    Unfreezes the backbone and trains everything with a lower learning rate.
    Reason: once the head is warm, the backbone can be carefully adapted to
    the deepfake distribution. Use LR ≈ 1e-4 (10× lower than head LR).

OVERFITTING PREVENTION
───────────────────────
Four mechanisms are active simultaneously:

1. Dropout (p=0.2 by default, configurable):
   Randomly zeros activations before the final linear layer.
   Prevents the head from memorising specific backbone feature combinations.

2. Stochastic depth (drop_connect_rate, handled internally by torchvision):
   EfficientNet uses stochastic depth — randomly skips entire MBConv blocks
   during training. Acts as a regulariser at the architecture level.

3. Data augmentation (JPEG simulation, ColorJitter — see data/transforms.py):
   The primary defence against overfitting on clean training data.

4. Weight decay via AdamW (handled in training/optimizers.py):
   L2 regularisation on all weight parameters. Prevents individual weights
   from growing large.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger(__name__)

# Number of output classes — always 2 for binary deepfake detection
# (index 0 = Real, index 1 = Fake, consistent with dataset.py and detector.py)
NUM_CLASSES: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENTNET-B0 BINARY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with a binary classification head for deepfake detection.

    Architecture:
        backbone: EfficientNet-B0 feature extractor (pretrained on ImageNet-1K)
        classifier: Sequential(Dropout(p), Linear(1280 → 2))

    The backbone is the standard torchvision EfficientNet-B0. The default
    head (Linear 1280 → 1000 for ImageNet) is replaced with a binary head.

    Output:
        Raw logits of shape (batch_size, 2).
        Do NOT apply softmax here — use it in inference only.
        BCEWithLogitsLoss and FocalLoss both expect raw logits.

    Args:
        pretrained:       Load ImageNet-1K pretrained backbone weights.
                          Always True for production. False only for ablations.
        dropout_rate:     Dropout probability before the final linear layer.
                          EfficientNet-B0 canonical value is 0.2.
                          Increase to 0.3–0.4 if overfitting is observed.
        freeze_backbone:  If True, the backbone parameters are frozen initially.
                          Use during Phase 1 (head warm-up). Unfreeze for Phase 2.
        drop_connect_rate: Stochastic depth rate for MBConv blocks.
                           torchvision uses 0.2 by default for B0.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        drop_connect_rate: float = 0.2,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes        = num_classes
        self.dropout_rate       = dropout_rate
        self.freeze_backbone    = freeze_backbone
        self.drop_connect_rate  = drop_connect_rate

        # ── Load backbone ─────────────────────────────────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None

        # torchvision's EfficientNet accepts stochastic_depth_prob at init time
        self.backbone = models.efficientnet_b0(
            weights=weights,
            stochastic_depth_prob=drop_connect_rate,
        )

        # ── Replace classifier head ───────────────────────────────────────────
        # backbone.classifier is the original Sequential([Dropout, Linear(1280→1000)])
        # We keep the Dropout (with our configurable rate) and replace the Linear.
        in_features: int = self.backbone.classifier[1].in_features  # always 1280 for B0

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "[efficientnet] Built EfficientNet-B0  pretrained=%s  "
            "num_classes=%d  dropout=%.2f  freeze_backbone=%s  in_features=%d",
            pretrained, num_classes, dropout_rate, freeze_backbone, in_features,
        )

        # ── Apply backbone freeze if requested ───────────────────────────────
        if freeze_backbone:
            self.freeze()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone + classifier head.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W).
               H = W = 224 for B0 (enforced by transforms.py).

        Returns:
            Raw logits of shape (batch_size, 2).
            Index 0 = Real, Index 1 = Fake.
        """
        return self.backbone(x)

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def freeze(self) -> None:
        """
        Freeze all backbone (feature extractor) parameters.

        Freezes: backbone.features (all MBConv blocks + stem)
        Leaves unfrozen: backbone.classifier (the binary head)

        When to use:
            Phase 1 training — train head only for 1–3 epochs to warm up
            the randomly initialised binary head before full fine-tuning.
            This prevents the large initial head gradients from corrupting
            the carefully pretrained backbone weights.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        frozen_params = sum(
            p.numel() for p in self.backbone.features.parameters()
        )
        logger.info(
            "[efficientnet] Backbone FROZEN — %d parameters frozen",
            frozen_params,
        )

    def unfreeze(self) -> None:
        """
        Unfreeze all backbone parameters for full fine-tuning.

        When to use:
            Phase 2 training — after the head is warmed up.
            Use a lower learning rate for the backbone (≈ LR / 10)
            to avoid catastrophic forgetting of ImageNet features.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True

        unfrozen_params = sum(
            p.numel() for p in self.backbone.features.parameters()
        )
        logger.info(
            "[efficientnet] Backbone UNFROZEN — %d parameters trainable",
            unfrozen_params,
        )

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Unfreeze only the last n MBConv blocks of the backbone.

        Useful for a gradual unfreeze strategy:
          • Unfreeze last 2 blocks first, train 1–2 epochs
          • Then unfreeze all blocks

        This reduces the risk of catastrophic forgetting vs unfreezing all
        at once, at the cost of a more complex training schedule.

        Args:
            n: Number of MBConv blocks to unfreeze from the end.
               EfficientNet-B0 has 9 feature blocks (indices 0–8).
        """
        blocks = list(self.backbone.features.children())
        total  = len(blocks)
        start  = max(0, total - n)

        # Freeze everything first
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Then selectively unfreeze the last n blocks
        unfrozen_params = 0
        for block in blocks[start:]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        logger.info(
            "[efficientnet] Unfroze last %d / %d blocks — %d parameters trainable",
            n, total, unfrozen_params,
        )

    # ── Inspection ────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    total - trainable,
        }

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"EfficientNetB0Classifier("
            f"dropout={self.dropout_rate}, "
            f"params_total={counts['total']:,}, "
            f"params_trainable={counts['trainable']:,})"
        )


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "EfficientNetB0Classifier",
    "NUM_CLASSES",
]