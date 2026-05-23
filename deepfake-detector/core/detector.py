"""
DeepfakeDetector — inference-only wrapper around the PyTorch model.

Migrated from deepfake-api/detector.py; uses model_registry, transforms, postprocessor.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from core.model_registry import build_model, load_weights, resolve_weights_path
from core.postprocessor import logits_to_prediction
from data.transforms import get_inference_transform

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Wraps an EfficientNet-B0 binary classifier for deepfake image detection.

    Usage:
        detector = DeepfakeDetector()
        label, confidence = detector.predict("path/to/image.png")
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        weights_path: Path | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.weights_path = resolve_weights_path(weights_path)
        self._transform = get_inference_transform()

        logger.info("[detector] Initialising on device: %s", self.device)
        logger.info("[detector] Model: %s", model_name)
        logger.info("[detector] Weights path: %s", self.weights_path)
        logger.info("[detector] Weights exist: %s", self.weights_path.exists())

        try:
            self.model = self._load_model()
            logger.info("[detector] Model loaded and in eval() mode")
        except Exception as exc:
            logger.error("[detector] Model load FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            self.model = None

    def _load_model(self) -> torch.nn.Module:
        model = build_model(
            self.model_name,
            num_classes=2,
            pretrained=False,
        )
        return load_weights(model, self.weights_path, device=self.device)

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """Open image file and return (1, 3, 224, 224) tensor."""
        try:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Cannot open image '{image_path}'. "
                f"File may be corrupt or in an unsupported format. ({exc})"
            ) from exc

        tensor = self._transform(img)
        return tensor.unsqueeze(0)

    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Run deepfake detection on a single image file.

        Returns:
            (label, confidence) — label is "Real"|"Fake", confidence is 0–100.
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. "
                "Check the server startup logs for the weight loading error."
            )

        tensor = self._preprocess(image_path).to(self.device)
        logger.debug(
            "[detector] Tensor shape=%s dtype=%s device=%s",
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

        try:
            with torch.no_grad():
                logits = self.model(tensor)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU ran out of memory during inference. "
                "The image may still be too large despite pre-resizing."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Inference pipeline failed: {type(exc).__name__}: {exc}"
            ) from exc

        label, confidence = logits_to_prediction(logits)
        logger.info(
            "[detector] Prediction: %s confidence=%.2f%% file=%r",
            label,
            confidence,
            image_path,
        )
        return label, confidence
