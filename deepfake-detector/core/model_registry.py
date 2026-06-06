"""
core/model_registry.py
───────────────────────
Centralised registry for model construction, weight loading, and path resolution.

PUBLIC API — two layers
────────────────────────
Layer 1: Detector-facing helpers (used by core/detector.py)
  build_model(model_name, num_classes, pretrained)  → nn.Module
  load_weights(model, checkpoint_path, device)       → nn.Module
  resolve_weights_path(weights_path)                 → Path | None

Layer 2: Training-facing factory (used by scripts/train.py)
  build(name, config)     → nn.Module  (config-driven, full ModelConfig)
  list_available()        → list[str]
  get_input_size(name)    → int
  register(name, builder) → None

Both layers share the same REGISTRY dict and the same builder functions,
so adding a new architecture in REGISTRY makes it available to both the
inference pipeline and the training pipeline automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Default weights location ───────────────────────────────────────────────────
_CORE_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS_PATH = _CORE_DIR / "model" / "deepfake_model.pth"

# Binary classification: index 0 = Real, index 1 = Fake
_DEFAULT_NUM_CLASSES = 2


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DETECTOR-FACING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_weights_path(
    weights_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve a weights file path to an absolute Path, or return None."""
    if weights_path is None:
        resolved = _DEFAULT_WEIGHTS_PATH.resolve()
        if resolved.exists():
            logger.info("[registry] resolve_weights_path: using default → %s", resolved)
        else:
            logger.warning("[registry] resolve_weights_path: default path does not exist: %s", resolved)
        return resolved

    if isinstance(weights_path, (str, Path)):
        resolved = Path(weights_path).resolve()
        if resolved.exists():
            logger.info("[registry] resolve_weights_path: explicit path → %s", resolved)
        else:
            logger.warning("[registry] resolve_weights_path: explicit path does not exist: %s", resolved)
        return resolved

    raise TypeError(f"weights_path must be None, str, or Path, got {type(weights_path).__name__!r}.")


def build_model(
    model_name: str,
    num_classes: int = _DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    freeze_backbone: bool = False,
    drop_connect_rate: float = 0.2,
) -> nn.Module:
    """Construct an nn.Module for the given architecture name."""
    if model_name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise ValueError(f"Unknown model architecture: {model_name!r}. Available: {available}.")

    config = ModelConfig(
        name              = model_name,
        pretrained        = pretrained,
        dropout_rate      = dropout_rate,
        freeze_backbone   = freeze_backbone,
        drop_connect_rate = drop_connect_rate,
        input_size        = _MODEL_INPUT_SIZES.get(model_name, 224),
    )

    logger.info("[registry] build_model: name=%r  num_classes=%d", model_name, num_classes)
    model = REGISTRY[model_name](config)
    return model


def load_weights(
    model:           nn.Module,
    checkpoint_path: Path,
    device:          Optional[torch.device] = None,
) -> nn.Module:
    """Load weights from a checkpoint file into an existing nn.Module."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    logger.info("[registry] load_weights: reading checkpoint → %s", checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read checkpoint file: {checkpoint_path}") from exc

    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    _load_state_dict_strict(model, state_dict, checkpoint_path)
    return model


def _extract_state_dict(checkpoint: object, source_path: Path) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint at {source_path} is not a dict")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Expected state_dict to be a dict")

    # Strip "model." prefix
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _load_state_dict_strict(model: nn.Module, state_dict: dict, source_path: Path) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as strict_exc:
        model.load_state_dict(state_dict, strict=False)
        raise RuntimeError(f"Architecture mismatch loading weights from: {source_path}\n{strict_exc}") from strict_exc


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — TRAINING-FACING FACTORY
# ══════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    def __init__(
        self,
        name:              str   = "efficientnet_b0",
        pretrained:        bool  = True,
        dropout_rate:      float = 0.2,
        freeze_backbone:   bool  = False,
        drop_connect_rate: float = 0.2,
        input_size:        int   = 224,
    ) -> None:
        self.name              = name
        self.pretrained        = pretrained
        self.dropout_rate      = dropout_rate
        self.freeze_backbone   = freeze_backbone
        self.drop_connect_rate = drop_connect_rate
        self.input_size        = input_size

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        filtered = {k: v for k, v in d.items() if hasattr(cls, k)} # simplified
        return cls(**filtered)


_MODEL_INPUT_SIZES: dict[str, int] = {
    "efficientnet_b0": 224,
    "efficientnet_b4": 380,
}


def _build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    from models.efficientnet import EfficientNetB0Classifier
    return EfficientNetB0Classifier(
        pretrained        = config.pretrained,
        dropout_rate      = config.dropout_rate,
        freeze_backbone   = config.freeze_backbone,
        drop_connect_rate = config.drop_connect_rate,
    )


def _build_efficientnet_b4(config: ModelConfig) -> nn.Module:
    from models.efficientnet import EfficientNetB4Classifier
    return EfficientNetB4Classifier(
        pretrained        = config.pretrained,
        dropout_rate      = config.dropout_rate,
        freeze_backbone   = config.freeze_backbone,
        drop_connect_rate = config.drop_connect_rate,
    )


REGISTRY: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet_b4": _build_efficientnet_b4,
}


def build(name: str, config: Optional[ModelConfig] = None) -> nn.Module:
    if name not in REGISTRY:
        raise ValueError(f"Unknown architecture: {name!r}")
    if config is None:
        config = ModelConfig(name=name)
    return REGISTRY[name](config)


def list_available() -> list[str]:
    return sorted(REGISTRY.keys())


def get_input_size(name: str) -> int:
    return _MODEL_INPUT_SIZES.get(name, 224)


def register(name: str, builder: Callable[[ModelConfig], nn.Module]) -> None:
    if name in REGISTRY:
        raise ValueError(f"Architecture {name!r} is already registered.")
    REGISTRY[name] = builder


__all__ = [
    "build_model",
    "load_weights",
    "resolve_weights_path",
    "build",
    "list_available",
    "get_input_size",
    "register",
    "ModelConfig",
    "REGISTRY",
]
