"""
detector.py — DeepFake Detector model logic

Architecture : EfficientNet-B0 (torchvision)
Input size   : 224 × 224  (model's own TRANSFORM handles final resize)
Classes      : index 0 = Real,  index 1 = Fake

Checkpoint formats supported (all handled transparently):
  ① torch.save(model.state_dict(), path)      raw state dict
  ② {"model_state_dict": ..., ...}            standard PyTorch checkpoint
  ③ {"state_dict": ..., ...}                  timm / PyTorch Lightning
  ④ keys prefixed with "model."               Lightning LightningModule wrapper

Note on resize:
  app.py already downscales incoming images to ≤ 512 × 512 before writing
  to the temp file. This TRANSFORM then resizes to 224 × 224 for the model.
  The two-stage approach ensures RAM safety AND correct model input shape.
"""

import logging
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, UnidentifiedImageError

# ── Logger ─────────────────────────────────────────────────────────────────────
# Using the module-level logger keeps all output consistent with the FastAPI
# logging config set up in app.py (same format, same level).
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR            = Path(__file__).resolve().parent
CUSTOM_WEIGHTS_PATH = BASE_DIR / "model" / "deepfake_model.pth"


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# EfficientNet-B0 canonical input: 224 × 224, ImageNet normalisation.
# This runs INSIDE the model pipeline after app.py has already capped the
# image at 512 × 512 — two independent resize stages for safety.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),   # ⑦ final resize for model input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet channel means
        std =[0.229, 0.224, 0.225],  # ImageNet channel stds
    ),
])


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════
class DeepfakeDetector:
    """
    Wraps an EfficientNet-B0 binary classifier for deepfake image detection.

    Usage:
        detector = DeepfakeDetector()          # once at server startup
        label, confidence = detector.predict("path/to/image.png")
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("[detector] ▶ Initialising on device: %s", self.device)
        logger.info("[detector] Weights path  : %s", CUSTOM_WEIGHTS_PATH)
        logger.info("[detector] Weights exist : %s", CUSTOM_WEIGHTS_PATH.exists())

        try:
            self.model = self._load_model()
            logger.info("[detector] ✓ Model loaded and in eval() mode")
        except Exception as exc:
            # Do NOT crash the server — set model to None so /health
            # can report the failure, and /detect returns a clean 500.
            logger.error("[detector] ✗ Model load FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            self.model = None

    # ── Architecture ───────────────────────────────────────────────────────────
    def _build_architecture(self) -> nn.Module:
        """
        Build an EfficientNet-B0 with a binary classification head.

        torchvision B0 head structure (default):
            classifier[0] = Dropout(p=0.2)
            classifier[1] = Linear(1280 → 1000)   ← replaced below

        We keep Dropout at p=0.2 — the canonical B0 value.
        Using p=0.4 (the B4 value) here is wrong and will cause
        weight mismatches if you fine-tune later.

        Output: 2 logits → softmax → [P(Real), P(Fake)]
                index 0 = Real,  index 1 = Fake
        """
        model       = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features   # always 1280 for B0

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),   # B0 canonical — do NOT change to 0.4
            nn.Linear(in_features, 2),          # binary head
        )

        logger.debug(
            "[detector] Architecture: EfficientNet-B0  in_features=%d  params=%s",
            in_features,
            f"{sum(p.numel() for p in model.parameters()):,}",
        )
        return model

    # ── Checkpoint normalisation ───────────────────────────────────────────────
    def _extract_state_dict(self, checkpoint: object) -> dict:
        """
        Accept any common checkpoint format and return a clean state dict
        whose keys match a bare torchvision EfficientNet-B0.

        Handles:
          ① raw OrderedDict of tensors
          ② {"model_state_dict": ...}   standard PyTorch training checkpoint
          ③ {"state_dict": ...}         timm / PyTorch Lightning
          ④ "model." key prefix         Lightning LightningModule wrapper

        Raises:
            RuntimeError — if the checkpoint is not a dict at all
        """
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Checkpoint is not a dict (got {type(checkpoint).__name__}). "
                f"Expected torch.save(model.state_dict(), path)."
            )

        # ── Unwrap nested checkpoint formats ──────────────────────────────────
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("[detector] Checkpoint format: {'model_state_dict': ...}")

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info("[detector] Checkpoint format: {'state_dict': ...} (timm / Lightning)")

        else:
            # Assume the dict itself is the flat state dict
            state_dict = checkpoint
            logger.info("[detector] Checkpoint format: raw state dict")

        # ── Strip "model." prefix added by Lightning LightningModule wrappers ─
        # e.g.  "model.features.0.0.weight"  →  "features.0.0.weight"
        stripped = sum(1 for k in state_dict if k.startswith("model."))
        if stripped:
            logger.info("[detector] Stripping 'model.' prefix from %d keys", stripped)

        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

        return cleaned

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self) -> nn.Module:
        """
        Load EfficientNet-B0 weights in priority order:
          1. Custom deepfake weights from CUSTOM_WEIGHTS_PATH
          2. ImageNet pretrained weights (fallback / demo mode)

        Returns an nn.Module in eval() mode on the correct device.

        Raises:
            RuntimeError — if the checkpoint file cannot be read or
                           if weights don't match the architecture
        """
        model = self._build_architecture()

        # ── No custom weights — fall back to ImageNet pretrained ──────────────
        if not CUSTOM_WEIGHTS_PATH.exists():
            logger.warning(
                "[detector] No weights at '%s'. "
                "Falling back to ImageNet pretrained weights. "
                "Predictions will NOT be reliable for deepfake detection "
                "until real deepfake-trained weights are supplied.",
                CUSTOM_WEIGHTS_PATH,
            )
            # Reload with ImageNet weights (classification head will be random)
            model       = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, 2),
            )
            model.to(self.device)
            model.eval()
            return model

        # ── Load checkpoint file ──────────────────────────────────────────────
        logger.info("[detector] Loading checkpoint: %s", CUSTOM_WEIGHTS_PATH)
        try:
            checkpoint = torch.load(
                CUSTOM_WEIGHTS_PATH,
                map_location=self.device,
                # weights_only=False is required for checkpoints that contain
                # non-tensor objects (optimiser state, epoch counters, config
                # dicts). PyTorch ≥ 2.6 changed the default to True, which
                # silently breaks such files.
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not read checkpoint: {CUSTOM_WEIGHTS_PATH}\n"
                f"The file may be corrupt or incompatible with this PyTorch version.\n"
                f"Error: {exc}"
            ) from exc

        # ── Normalise checkpoint to a flat state dict ─────────────────────────
        state_dict = self._extract_state_dict(checkpoint)

        # ── Load weights — strict=True first, diagnose on failure ─────────────
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("[detector] ✓ load_state_dict strict=True — OK")

        except RuntimeError as strict_exc:
            # Run strict=False to surface exactly which keys are wrong
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.error("[detector] strict=True failed — mismatch details:")
                logger.error("[detector]   missing keys    : %s", missing[:5])
                logger.error("[detector]   unexpected keys : %s", unexpected[:5])
            except Exception:
                pass  # strict=False also failed — original error is more useful

            raise RuntimeError(
                f"Architecture mismatch loading weights.\n"
                f"The checkpoint was likely trained on a different EfficientNet "
                f"variant or with a different classifier head.\n"
                f"Weights path : {CUSTOM_WEIGHTS_PATH}\n"
                f"Error        : {strict_exc}"
            ) from strict_exc

        model.to(self.device)
        model.eval()
        return model

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Open an image file and convert it to a model-ready tensor.

        app.py already:
          - verified the image with PIL verify()
          - applied EXIF orientation correction
          - converted to RGB
          - resized to ≤ 512 × 512

        We still apply EXIF transpose and RGB conversion here as a
        safety net in case this method is called outside the normal
        app.py pipeline (e.g. from a test script).

        Returns:
            Tensor of shape (1, 3, 224, 224)

        Raises:
            ValueError: if the image cannot be opened
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)   # safety net — already done by app.py
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Cannot open image '{image_path}'. "
                f"File may be corrupt or in an unsupported format. ({exc})"
            ) from exc

        tensor = TRANSFORM(img)        # (3, 224, 224)
        return tensor.unsqueeze(0)     # (1, 3, 224, 224)  — batch dimension

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Run deepfake detection on a single image file.

        Args:
            image_path: Absolute or relative path to the image file.
                        The file is expected to already be validated and
                        resized by app.py's _validate_and_prepare_image().

        Returns:
            (label, confidence)
              label      — "Real" or "Fake"
              confidence — probability of the predicted class × 100  (0.0–100.0)

        Raises:
            RuntimeError: model not loaded, or inference pipeline failed → HTTP 500
            ValueError:   image cannot be opened at this path             → HTTP 422
        """
        # ── Guard: model must be loaded ───────────────────────────────────────
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. "
                "Check the server startup logs for the weight loading error."
            )

        # ── Preprocess → tensor ───────────────────────────────────────────────
        tensor = self._preprocess(image_path).to(self.device)
        logger.debug(
            "[detector] Tensor: shape=%s  dtype=%s  device=%s",
            tuple(tensor.shape), tensor.dtype, tensor.device,
        )

        # ── Forward pass ──────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                logits = self.model(tensor)            # (1, 2) — raw logits
                probs  = torch.softmax(logits, dim=1)  # (1, 2) — probabilities
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU ran out of memory during inference. "
                "The image may still be too large despite pre-resizing."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Inference pipeline failed: {type(exc).__name__}: {exc}"
            ) from exc

        # ── Parse probabilities ───────────────────────────────────────────────
        real_prob: float = probs[0, 0].item()   # index 0 → Real
        fake_prob: float = probs[0, 1].item()   # index 1 → Fake

        # Assign verdict to whichever class has the higher probability
        if fake_prob >= real_prob:
            label      = "Fake"
            confidence = fake_prob * 100.0
        else:
            label      = "Real"
            confidence = real_prob * 100.0

        logger.info(
            "[detector] Prediction: %s  confidence=%.2f%%  "
            "P(real)=%.4f  P(fake)=%.4f  file=%r",
            label, confidence, real_prob, fake_prob, image_path,
        )

        return label, confidence