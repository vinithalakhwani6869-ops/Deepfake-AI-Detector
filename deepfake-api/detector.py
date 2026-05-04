"""
detector.py — DeepFake Detector model logic

Architecture: EfficientNet-B4 (torchvision)
Weights:
  - If  model/deepfake_model.pth  exists  → loaded as custom deepfake weights
  - Otherwise                             → falls back to ImageNet pretrained
    weights and repurposes the final
    classifier as a binary head.

The model outputs a single logit for class 1 ("Fake").
Sigmoid(logit) > 0.5  → Fake
Sigmoid(logit) ≤ 0.5  → Real
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
CUSTOM_WEIGHTS_PATH = Path("model/deepfake_model.pth")

# ── Image preprocessing ────────────────────────────────────────────────────────
# EfficientNet-B4 canonical input: 380×380, ImageNet normalisation
TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


class DeepfakeDetector:
    """
    Wraps an EfficientNet-B4 binary classifier.

    Usage:
        detector = DeepfakeDetector()
        label, confidence = detector.predict("path/to/image.jpg")
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _build_architecture(self) -> nn.Module:
        """
        EfficientNet-B4 with the final fully-connected layer replaced by a
        binary head (2 outputs: Real / Fake).

        Drop-in swap: to use ResNet-50 instead, uncomment the ResNet block
        and comment out the EfficientNet block.
        """
        # ── EfficientNet-B4 ──
        model = models.efficientnet_b4(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 2),     # 0 = Real, 1 = Fake
        )

        # ── ResNet-50 (alternative — uncomment to switch) ──
        # model = models.resnet50(pretrained=False)
        # in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features, 2)

        return model

    def _load_model(self) -> nn.Module:
        """
        Load weights in priority order:
          1. Custom deepfake weights from CUSTOM_WEIGHTS_PATH
          2. ImageNet pretrained weights (useful baseline / demo mode)
        """
        model = self._build_architecture()

        if CUSTOM_WEIGHTS_PATH.exists():
            logger.info("Loading custom deepfake weights from %s", CUSTOM_WEIGHTS_PATH)
            try:
                state_dict = torch.load(
                    CUSTOM_WEIGHTS_PATH,
                    map_location=self.device,
                )
                # Support both raw state_dict and {"model_state_dict": ...} checkpoints
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict, strict=True)
                logger.info("Custom weights loaded successfully.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load custom weights from {CUSTOM_WEIGHTS_PATH}: {exc}"
                ) from exc
        else:
            logger.warning(
                "Custom weights not found at '%s'. "
                "Falling back to ImageNet pretrained weights. "
                "Predictions will NOT be reliable for deepfake detection "
                "until real weights are supplied.",
                CUSTOM_WEIGHTS_PATH,
            )
            # Reload with pretrained ImageNet weights
            model = models.efficientnet_b4(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 2),
            )
            # The binary head is randomly initialised in this fallback path —
            # fine-tune or replace with proper deepfake weights before deployment.

        model.to(self.device)
        model.eval()
        return model

    # ── Preprocessing ──────────────────────────────────────────────────────────
    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Open, validate, and transform an image into a model-ready tensor.

        Raises:
            ValueError: if the image cannot be opened or is corrupt.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Could not open image '{image_path}'. "
                f"File may be corrupt or in an unsupported format. ({exc})"
            ) from exc

        tensor = TRANSFORM(img)              # (3, 380, 380)
        return tensor.unsqueeze(0)           # (1, 3, 380, 380)

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Run deepfake detection on a single image.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            (label, confidence)
            label      — "Real" or "Fake"
            confidence — probability of the predicted class × 100  (0–100)

        Raises:
            ValueError:  unreadable / corrupt image
            RuntimeError: inference failure
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")

        # 1. Preprocess
        tensor = self._preprocess(image_path)
        tensor = tensor.to(self.device)

        # 2. Forward pass
        try:
            with torch.no_grad():
                logits = self.model(tensor)          # (1, 2)
                probs  = torch.softmax(logits, dim=1) # (1, 2)
        except Exception as exc:
            raise RuntimeError(f"Inference failed: {exc}") from exc

        # 3. Parse results
        # Index 0 → Real,  Index 1 → Fake
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()

        if fake_prob >= real_prob:
            label      = "Fake"
            confidence = fake_prob * 100.0
        else:
            label      = "Real"
            confidence = real_prob * 100.0

        logger.info(
            "Prediction: %s | Confidence: %.2f%% | File: %s",
            label, confidence, image_path,
        )
        return label, confidence