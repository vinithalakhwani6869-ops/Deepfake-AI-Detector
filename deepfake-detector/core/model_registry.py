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
  register(name, builder) → None

Both layers share the same REGISTRY dict and the same builder functions,
so adding a new architecture in REGISTRY makes it available to both the
inference pipeline and the training pipeline automatically.

WHY THESE THREE FUNCTIONS LIVE HERE AND NOT IN DETECTOR.PY
────────────────────────────────────────────────────────────
core/detector.py is responsible for the inference lifecycle:
  image → tensor → forward pass → label + confidence

It should not be responsible for:
  • Knowing which architectures exist (that is the registry's job)
  • Normalising checkpoint key formats (that is format-specific knowledge)
  • Resolving weights file paths (reused by evaluator, benchmark, export scripts)

Separating these responsibilities means:
  • detector.py can switch architectures with a single config change
  • load_weights() is tested independently from the inference loop
  • resolve_weights_path() can be called from evaluation scripts
    without constructing a full DeepfakeDetector

CHECKPOINT FORMAT COMPATIBILITY
────────────────────────────────
load_weights() transparently handles every checkpoint format produced by
the training pipeline and by third-party deepfake model releases:

  Format 1 — raw state dict (plain torch.save(model.state_dict(), path)):
    {"features.0.0.weight": tensor, ...}

  Format 2 — standard PyTorch training checkpoint:
    {"model_state_dict": {...}, "optimizer_state": {...}, "epoch": int, ...}

  Format 3 — timm / PyTorch Lightning:
    {"state_dict": {...}, "hyper_parameters": {...}, ...}

  Format 4 — Lightning LightningModule wrapper (model. prefix):
    {"model.features.0.0.weight": tensor, ...}
    Stripped to: {"features.0.0.weight": tensor, ...}

  Any combination of formats 2-4 is also handled (e.g. Lightning checkpoint
  with state_dict key AND model. prefixes in the nested dict).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Default weights location ───────────────────────────────────────────────────
# Both detector.py and model_registry.py live in core/.
# The model/ directory containing deepfake_model.pth is also inside core/.
# Path(__file__).resolve().parent  →  <project_root>/core
# / "model" / "deepfake_model.pth" →  <project_root>/core/model/deepfake_model.pth
# This matches the CUSTOM_WEIGHTS_PATH convention already used in detector.py.
_CORE_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS_PATH = _CORE_DIR / "model" / "deepfake_model.pth"

# Binary classification: index 0 = Real, index 1 = Fake
# Consistent with dataset.py (LABEL_REAL=0, LABEL_FAKE=1) and detector.py
_DEFAULT_NUM_CLASSES = 2


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DETECTOR-FACING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_weights_path(
    weights_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Resolve a weights file path to an absolute Path, or return None.

    Accepts:
        None  → uses the default location: core/model/deepfake_model.pth
        str   → coerced to Path, then resolved
        Path  → resolved to absolute path

    The returned Path is guaranteed to be absolute. Existence is checked
    and logged, but a non-existent path is returned rather than raising —
    the caller (detector.py) decides how to handle missing weights
    (fallback to ImageNet pretrained vs hard failure).

    Args:
        weights_path: Explicit path override. Pass None to use the default.

    Returns:
        Resolved absolute Path, or None if weights_path=None and the default
        location does not exist AND no fallback is appropriate.

    Raises:
        TypeError: if weights_path is not None, str, or Path.
    """
    if weights_path is None:
        resolved = _DEFAULT_WEIGHTS_PATH.resolve()
        if resolved.exists():
            logger.info(
                "[registry] resolve_weights_path: using default → %s", resolved
            )
        else:
            logger.warning(
                "[registry] resolve_weights_path: default path does not exist: %s  "
                "(caller should fall back to ImageNet pretrained weights)",
                resolved,
            )
        return resolved

    if isinstance(weights_path, (str, Path)):
        resolved = Path(weights_path).resolve()
        if resolved.exists():
            logger.info(
                "[registry] resolve_weights_path: explicit path → %s", resolved
            )
        else:
            logger.warning(
                "[registry] resolve_weights_path: explicit path does not exist: %s",
                resolved,
            )
        return resolved

    raise TypeError(
        f"weights_path must be None, str, or Path, got {type(weights_path).__name__!r}. "
        f"Example: resolve_weights_path(Path('core/model/deepfake_model.pth'))"
    )


def build_model(
    model_name: str,
    num_classes: int = _DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    freeze_backbone: bool = False,
    drop_connect_rate: float = 0.2,
) -> nn.Module:
    """
    Construct an nn.Module for the given architecture name.

    This is the detector-facing constructor. It builds the model architecture
    only — no weights are loaded here. Call load_weights() separately to
    load a checkpoint after calling build_model().

    Separation of build_model() and load_weights() allows:
        1. Architecture construction to be validated independently
        2. Weights loading to be retried without rebuilding the model
        3. The same model instance to be loaded with different weight files
           (e.g. in model comparison scripts)

    Args:
        model_name:        Registry key for the architecture.
                           Must be a key in REGISTRY.
                           e.g. "efficientnet_b0"
        num_classes:       Number of output classes. Always 2 for binary
                           deepfake detection (Real=0, Fake=1).
                           Exposed as a parameter for future multi-class extension.
        pretrained:        If True, initialise the backbone with ImageNet-1K
                           pretrained weights. The classification head is always
                           randomly initialised (it has num_classes outputs, not 1000).
                           Set False only for ablation experiments.
        dropout_rate:      Dropout probability before the final Linear layer.
                           EfficientNet-B0 canonical value: 0.2.
                           Increase to 0.3–0.4 if training loss is low but val AUC
                           is not improving (overfitting signal).
        freeze_backbone:   If True, freeze the backbone feature extractor at init.
                           Used for Phase 1 training (head warm-up only).
                           Call model.unfreeze() for Phase 2 full fine-tuning.
        drop_connect_rate: Stochastic depth drop probability for EfficientNet
                           MBConv blocks. torchvision B0 default: 0.2.

    Returns:
        nn.Module in training mode. The caller is responsible for:
            • model.to(device)  — move to correct device
            • model.eval()      — set inference mode before prediction
            • load_weights()    — load a checkpoint if available

    Raises:
        ValueError: if model_name is not registered in REGISTRY.
    """
    if model_name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise ValueError(
            f"Unknown model architecture: {model_name!r}. "
            f"Registered architectures: {available}. "
            f"To add a new architecture, create models/<name>.py and register "
            f"a builder function in core/model_registry.REGISTRY."
        )

    config = ModelConfig(
        name              = model_name,
        pretrained        = pretrained,
        dropout_rate      = dropout_rate,
        freeze_backbone   = freeze_backbone,
        drop_connect_rate = drop_connect_rate,
        input_size        = _MODEL_INPUT_SIZES.get(model_name, 224),
    )

    logger.info(
        "[registry] build_model: name=%r  num_classes=%d  pretrained=%s  "
        "dropout=%.2f  freeze=%s",
        model_name, num_classes, pretrained, dropout_rate, freeze_backbone,
    )

    model = REGISTRY[model_name](config)

    # Log parameter counts for immediate sanity check
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "[registry] build_model: built %s  params_total=%s  params_trainable=%s",
        type(model).__name__,
        f"{total:,}",
        f"{trainable:,}",
    )

    return model


def load_weights(
    model:           nn.Module,
    checkpoint_path: Path,
    device:          Optional[torch.device] = None,
) -> nn.Module:
    """
    Load weights from a checkpoint file into an existing nn.Module.

    Handles every checkpoint format produced by the training pipeline and by
    common third-party deepfake model releases (see module docstring for the
    full format compatibility list).

    The model is modified in-place AND returned for chaining:
        model = load_weights(build_model("efficientnet_b0"), weights_path, device)

    Weight loading uses strict=True. If strict loading fails, the function
    automatically runs strict=False and logs exactly which keys are missing
    or unexpected — actionable information for debugging architecture mismatches.

    Args:
        model:           nn.Module instance (from build_model()).
                         Must already be on the correct device, or pass device
                         to have map_location applied during torch.load.
        checkpoint_path: Absolute or relative path to the .pth checkpoint file.
                         Must exist. Use resolve_weights_path() to validate
                         existence before calling this function.
        device:          torch.device for map_location in torch.load.
                         If None, uses the device of the first model parameter.
                         Explicitly passing device avoids an extra parameter
                         scan and is recommended.

    Returns:
        The same nn.Module instance with weights loaded. Still in whatever
        mode it was in before (training or eval) — the caller must call
        model.eval() explicitly before inference.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
        RuntimeError:      if the checkpoint file cannot be read (corrupt, wrong
                           PyTorch version, unsupported format).
        RuntimeError:      if the state dict keys do not match the model
                           architecture (wrong EfficientNet variant, wrong
                           number of output classes, wrong head structure).
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Run resolve_weights_path() before calling load_weights() to "
            f"confirm the file exists."
        )

    # Determine device for map_location
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    logger.info(
        "[registry] load_weights: reading checkpoint → %s  device=%s",
        checkpoint_path, device,
    )

    # ── Read checkpoint from disk ─────────────────────────────────────────────
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            # weights_only=False is required for checkpoints that contain
            # non-tensor objects (optimiser state, epoch counters, config dicts,
            # Python scalars). PyTorch >= 2.6 changed the default to True, which
            # silently breaks any checkpoint saved with extra metadata.
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read checkpoint file: {checkpoint_path}\n"
            f"The file may be corrupt, truncated, or saved by an incompatible "
            f"PyTorch version.\n"
            f"Error: {type(exc).__name__}: {exc}"
        ) from exc

    # ── Normalise checkpoint to a flat state dict ─────────────────────────────
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)

    # ── Load with strict=True — fail loudly with diagnosis if mismatched ──────
    _load_state_dict_strict(model, state_dict, checkpoint_path)

    logger.info(
        "[registry] load_weights: ✓ weights loaded  file=%s",
        checkpoint_path.name,
    )

    return model


# ── Private helpers for load_weights ─────────────────────────────────────────

def _extract_state_dict(checkpoint: object, source_path: Path) -> dict:
    """
    Normalise any checkpoint format to a flat state dict.

    Handles formats 1–4 described in the module docstring.
    Strips the "model." key prefix added by Lightning LightningModule wrappers.

    Args:
        checkpoint:  The object returned by torch.load().
        source_path: Used only for error messages.

    Returns:
        Dict mapping parameter name → tensor, compatible with nn.Module.load_state_dict().

    Raises:
        RuntimeError: if checkpoint is not a dict.
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint at {source_path} is not a dict "
            f"(got {type(checkpoint).__name__}). "
            f"Expected: torch.save(model.state_dict(), path) or a dict containing "
            f"a 'model_state_dict' or 'state_dict' key.\n"
            f"If you saved the full model with torch.save(model, path), "
            f"you need to re-save with torch.save(model.state_dict(), path)."
        )

    # ── Unwrap nested checkpoint formats ──────────────────────────────────────
    if "model_state_dict" in checkpoint:
        # Format 2: standard PyTorch training checkpoint
        state_dict = checkpoint["model_state_dict"]
        fmt = "standard PyTorch {'model_state_dict': ...}"

    elif "state_dict" in checkpoint:
        # Format 3: timm or PyTorch Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        fmt = "timm/Lightning {'state_dict': ...}"

    else:
        # Format 1: the dict itself is the state dict
        state_dict = checkpoint
        fmt = "raw state dict"

    logger.info("[registry] Checkpoint format detected: %s", fmt)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Expected state_dict to be a dict, got {type(state_dict).__name__}. "
            f"Checkpoint may be malformed."
        )

    # ── Strip "model." prefix (Format 4: Lightning LightningModule wrapper) ───
    # When a model is wrapped in a LightningModule, all parameter names gain a
    # "model." prefix: "model.features.0.0.weight" → "features.0.0.weight".
    # This prefix must be stripped before loading into a bare nn.Module.
    stripped_count = sum(1 for k in state_dict if k.startswith("model."))
    if stripped_count > 0:
        state_dict = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        logger.info(
            "[registry] Stripped 'model.' prefix from %d / %d keys",
            stripped_count, len(state_dict),
        )

    logger.debug(
        "[registry] State dict: %d keys  first=%s  last=%s",
        len(state_dict),
        list(state_dict.keys())[:2],
        list(state_dict.keys())[-2:],
    )

    return state_dict


def _load_state_dict_strict(
    model: nn.Module,
    state_dict: dict,
    source_path: Path,
) -> None:
    """
    Load state_dict into model with strict=True.

    On failure, runs strict=False to collect the full list of missing and
    unexpected keys and includes them in the RuntimeError message.

    This two-pass approach makes architecture mismatches immediately actionable:
    the error message tells you exactly which layer names are wrong, rather than
    a generic "size mismatch" from PyTorch.

    Args:
        model:       The nn.Module to load into.
        state_dict:  Normalised flat state dict from _extract_state_dict().
        source_path: Checkpoint file path, used only for error messages.

    Raises:
        RuntimeError: if strict=True loading fails.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("[registry] load_state_dict strict=True — OK")

    except RuntimeError as strict_exc:
        # Collect detailed mismatch information using strict=False
        missing_keys:    list[str] = []
        unexpected_keys: list[str] = []

        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys    = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
        except Exception:
            pass  # strict=False also failed — the strict=True error is more useful

        # Build a diagnostic message that tells the user what to actually do
        diag_lines = [
            f"Architecture mismatch loading weights from: {source_path}",
            "",
            "This usually means one of:",
            "  • The checkpoint was trained on a different EfficientNet variant",
            "    (e.g. B4 checkpoint loaded into a B0 model)",
            "  • The number of output classes differs",
            "    (e.g. checkpoint has 1-class head, model expects 2-class head)",
            "  • The classifier head structure was changed between training and inference",
            "",
        ]

        if missing_keys:
            shown = missing_keys[:8]
            more  = len(missing_keys) - len(shown)
            diag_lines.append(
                f"Missing keys ({len(missing_keys)}): {shown}"
                + (f" ... and {more} more" if more > 0 else "")
            )

        if unexpected_keys:
            shown = unexpected_keys[:8]
            more  = len(unexpected_keys) - len(shown)
            diag_lines.append(
                f"Unexpected keys ({len(unexpected_keys)}): {shown}"
                + (f" ... and {more} more" if more > 0 else "")
            )

        diag_lines += [
            "",
            f"Original error: {strict_exc}",
        ]

        logger.error("[registry] load_state_dict strict=True FAILED:")
        for line in diag_lines:
            if line:
                logger.error("  %s", line)

        raise RuntimeError("\n".join(diag_lines)) from strict_exc


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — TRAINING-FACING FACTORY
# ══════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    """
    Typed configuration for model construction (training pipeline).

    Used by scripts/train.py and training/trainer.py.
    The detector-facing build_model() accepts individual keyword arguments
    rather than a ModelConfig — both ultimately call the same REGISTRY builder.

    Args:
        name:              Registry key identifying the architecture.
        pretrained:        Load ImageNet pretrained backbone weights.
        dropout_rate:      Dropout probability before classification head.
        freeze_backbone:   Freeze backbone at construction time.
        drop_connect_rate: Stochastic depth rate (EfficientNet-specific).
        input_size:        Spatial input resolution (height = width).
    """

    _VALID_FIELDS: frozenset[str] = frozenset({
        "name", "pretrained", "dropout_rate",
        "freeze_backbone", "drop_connect_rate", "input_size",
    })

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
        """
        Build from a plain dict (e.g. from OmegaConf / YAML).

        Unknown keys are silently ignored so full training config dicts can
        be passed without pre-filtering.
        """
        filtered = {k: v for k, v in d.items() if k in cls._VALID_FIELDS}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(name={self.name!r}, pretrained={self.pretrained}, "
            f"dropout={self.dropout_rate}, freeze={self.freeze_backbone})"
        )


# ── Input size per architecture ───────────────────────────────────────────────
# Used by build_model() to populate ModelConfig.input_size automatically.
# Add entries here when registering new architectures.
_MODEL_INPUT_SIZES: dict[str, int] = {
    "efficientnet_b0": 224,
    # "efficientnet_b4": 380,
    # "xception":        299,
    # "vit_b16":         224,
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each builder is a named function — not a lambda.
# Named functions are picklable, debuggable, and independently testable.

def _build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    """
    Build EfficientNet-B0 binary classifier.

    Uses EfficientNetB0Classifier from models/efficientnet.py.
    See that module for full architecture and transfer learning documentation.
    """
    from models.efficientnet import EfficientNetB0Classifier
    return EfficientNetB0Classifier(
        pretrained        = config.pretrained,
        dropout_rate      = config.dropout_rate,
        freeze_backbone   = config.freeze_backbone,
        drop_connect_rate = config.drop_connect_rate,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Keys: lowercase, underscore-separated. Stored in checkpoint files.
# Adding an architecture requires:
#   1. Create models/<name>.py
#   2. Write a _build_<name>(config: ModelConfig) -> nn.Module function above
#   3. Add "name": _build_<name> entry here
#   4. Add "name": input_size entry in _MODEL_INPUT_SIZES above

REGISTRY: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "efficientnet_b0": _build_efficientnet_b0,
    # "efficientnet_b4": _build_efficientnet_b4,
    # "xception":        _build_xception,
    # "vit_b16":         _build_vit_b16,
}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING-FACING FACTORY (preserved from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def build(name: str, config: Optional[ModelConfig] = None) -> nn.Module:
    """
    Construct an nn.Module using a ModelConfig (training pipeline entry point).

    Used by scripts/train.py. For inference use build_model() instead,
    which accepts individual keyword arguments rather than a ModelConfig.

    Args:
        name:   Architecture registry key. e.g. "efficientnet_b0"
        config: ModelConfig instance. If None, ModelConfig(name=name) defaults are used.

    Returns:
        nn.Module in training mode.

    Raises:
        ValueError: if name is not in REGISTRY.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown architecture: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )

    if config is None:
        config = ModelConfig(name=name)

    logger.info("[registry] build: name=%r  config=%s", name, config)
    model = REGISTRY[name](config)
    logger.info("[registry] build: constructed %s", type(model).__name__)
    return model


def list_available() -> list[str]:
    """Return sorted list of all registered architecture names."""
    return sorted(REGISTRY.keys())


def register(name: str, builder: Callable[[ModelConfig], nn.Module]) -> None:
    """
    Register a new architecture at runtime.

    For plugin-style extensions or experiment scripts that need to add a model
    without modifying this file directly.

    Args:
        name:    Unique lowercase registry key.
        builder: Callable(ModelConfig) → nn.Module.

    Raises:
        ValueError: if name is already registered.
    """
    if name in REGISTRY:
        raise ValueError(
            f"Architecture {name!r} is already registered. "
            f"Use a different name, or remove the existing entry first."
        )
    REGISTRY[name] = builder
    logger.info("[registry] Registered new architecture: %r", name)


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    # Layer 1: detector-facing helpers
    "build_model",
    "load_weights",
    "resolve_weights_path",
    # Layer 2: training-facing factory
    "build",
    "list_available",
    "register",
    # Config class
    "ModelConfig",
    # Registry dict (for introspection)
    "REGISTRY",
]