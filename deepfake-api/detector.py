"""
detector.py — DeepFake Detector model logic

Architecture : EfficientNet-B0 (torchvision)
Input size   : 224 x 224
Classes      : 0 = Real, 1 = Fake

Checkpoint formats supported:
  - torch.save(model.state_dict(), path)          raw state dict
  - {"model_state_dict": state_dict, ...}         standard PyTorch checkpoint
  - {"state_dict": state_dict, ...}               timm / Lightning checkpoint
  - keys prefixed with "model."                   Lightning LightningModule wrapper
"""

import logging
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent
CUSTOM_WEIGHTS_PATH = BASE_DIR / "model" / "deepfake_model.pth"

# ── Preprocessing pipeline ─────────────────────────────────────────────────────
# EfficientNet-B0 canonical input: 224 x 224, ImageNet normalisation
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


class DeepfakeDetector:
    """
    Wraps an EfficientNet-B0 binary classifier for deepfake detection.

    Usage:
        detector = DeepfakeDetector()
        label, confidence = detector.predict("path/to/image.jpg")
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DeepfakeDetector] device          = {self.device}")
        print(f"[DeepfakeDetector] weights path    = {CUSTOM_WEIGHTS_PATH}")
        print(f"[DeepfakeDetector] weights exist   = {CUSTOM_WEIGHTS_PATH.exists()}")

        try:
            self.model = self._load_model()
            print("[DeepfakeDetector] model loaded OK")
        except Exception as exc:
            print(f"[DeepfakeDetector] LOAD FAILED: {exc}")
            traceback.print_exc()
            self.model = None   # server stays alive; /health will report model_loaded=False

    # ── Architecture ───────────────────────────────────────────────────────────
    def _build_architecture(self) -> nn.Module:
        """
        EfficientNet-B0 with a binary head.

        torchvision B0 final feature channels : 1280
        Dropout rate                          : 0.2  (B0 canonical — NOT 0.4 which is B4)
        Output                                : 2 logits → softmax → [P(Real), P(Fake)]
        """
        model       = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features          # always 1280 for B0

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2),                         # 0 = Real, 1 = Fake
        )
        return model

    # ── Checkpoint normalisation ───────────────────────────────────────────────
    def _extract_state_dict(self, checkpoint: object) -> dict:
        """
        Accept any common checkpoint format and return a plain state dict
        whose keys match a bare torchvision EfficientNet-B0.

        Handles:
          1. Raw OrderedDict of tensors
          2. {"model_state_dict": ...}   standard PyTorch training loop
          3. {"state_dict": ...}         timm / PyTorch Lightning
          4. "model." key prefix         Lightning LightningModule wrapper
        """
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Checkpoint is not a dict (got {type(checkpoint).__name__}). "
                f"Expected torch.save(model.state_dict(), path)."
            )

        # ── Unwrap nested formats ──────────────────────────────────────────────
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("[DeepfakeDetector] checkpoint format: {'model_state_dict': ...}")

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("[DeepfakeDetector] checkpoint format: {'state_dict': ...} (timm/Lightning)")

        else:
            # Assume the dict itself is the state dict (most common for simple saves)
            state_dict = checkpoint
            print("[DeepfakeDetector] checkpoint format: raw state dict")

        # ── Strip "model." prefix added by Lightning wrappers ─────────────────
        # e.g. "model.features.0.0.weight" → "features.0.0.weight"
        stripped_count = sum(1 for k in state_dict if k.startswith("model."))
        if stripped_count:
            print(f"[DeepfakeDetector] stripping 'model.' prefix from {stripped_count} keys")

        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

        return cleaned

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self) -> nn.Module:
        model = self._build_architecture()

        # ── No weights file — fall back to ImageNet pretrained ─────────────────
        if not CUSTOM_WEIGHTS_PATH.exists():
            logger.warning(
                "No weights found at '%s'. "
                "Falling back to ImageNet pretrained weights. "
                "Deepfake predictions will NOT be reliable until real weights are supplied.",
                CUSTOM_WEIGHTS_PATH,
            )
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, 2),
            )
            model.to(self.device)
            model.eval()
            return model

        # ── Load the checkpoint file ───────────────────────────────────────────
        print(f"[DeepfakeDetector] loading checkpoint from {CUSTOM_WEIGHTS_PATH}")
        try:
            checkpoint = torch.load(
                CUSTOM_WEIGHTS_PATH,
                map_location=self.device,
                weights_only=False,     # required for checkpoints with non-tensor objects
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not read checkpoint file: {CUSTOM_WEIGHTS_PATH}\n"
                f"The file may be corrupt or incompatible with this PyTorch version.\n"
                f"Error: {exc}"
            ) from exc

        # ── Normalise to a plain state dict ────────────────────────────────────
        try:
            state_dict = self._extract_state_dict(checkpoint)
        except RuntimeError:
            raise

        # ── Load with strict=True, fall back to strict=False for diagnosis ─────
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[DeepfakeDetector] load_state_dict strict=True — OK")

        except RuntimeError as strict_exc:
            # Try strict=False so we can print exactly which keys are mismatched
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[DeepfakeDetector] strict=True failed.")
                print(f"[DeepfakeDetector] missing keys    : {missing[:5]}")
                print(f"[DeepfakeDetector] unexpected keys : {unexpected[:5]}")
            except Exception:
                pass

            raise RuntimeError(
                f"Architecture mismatch loading weights.\n"
                f"The checkpoint was likely trained on a different EfficientNet variant "
                f"or with a different classifier head.\n"
                f"Weights path : {CUSTOM_WEIGHTS_PATH}\n"
                f"Error        : {strict_exc}"
            ) from strict_exc

        model.to(self.device)
        model.eval()
        return model

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Open an image, correct EXIF orientation, and return a (1, 3, 224, 224) tensor.

        Raises:
            ValueError: if the image cannot be opened or decoded.
        """
        try:
            img = Image.open(image_path).convert("RGB")

            # Phone camera JPEGs store rotation in EXIF metadata.
            # PIL does NOT auto-rotate — exif_transpose() fixes this so the
            # model always sees an upright face.
            img = ImageOps.exif_transpose(img)

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Cannot open image '{image_path}'. "
                f"File may be corrupt or in an unsupported format. ({exc})"
            ) from exc

        tensor = TRANSFORM(img)        # (3, 224, 224)
        return tensor.unsqueeze(0)     # (1, 3, 224, 224)

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Run deepfake detection on a single image file.

        Args:
            image_path: Path to the saved temp image file.

        Returns:
            (label, confidence)
            label      — "Real" or "Fake"
            confidence — probability of the predicted class × 100  (0.0 – 100.0)

        Raises:
            ValueError:   unreadable / corrupt image     → HTTP 422
            RuntimeError: model not loaded or inference  → HTTP 500
        """
        if self.model is None:
            raise RuntimeError(
                "Model failed to load at startup. "
                "Check the server logs for the exact error from _load_model()."
            )

        # 1. Preprocess
        tensor = self._preprocess(image_path).to(self.device)

        # 2. Forward pass
        try:
            with torch.no_grad():
                logits = self.model(tensor)            # (1, 2)
                probs  = torch.softmax(logits, dim=1)  # (1, 2)
        except Exception as exc:
            raise RuntimeError(f"Inference failed: {exc}") from exc

        # 3. Parse output
        real_prob = probs[0, 0].item()   # index 0 → Real
        fake_prob = probs[0, 1].item()   # index 1 → Fake

        if fake_prob >= real_prob:
            label      = "Fake"
            confidence = fake_prob * 100.0
        else:
            label      = "Real"
            confidence = real_prob * 100.0

        print(
            f"[DeepfakeDetector] result={label}  "
            f"confidence={confidence:.2f}%  "
            f"real={real_prob:.4f}  fake={fake_prob:.4f}"
        )
        return label, confidence