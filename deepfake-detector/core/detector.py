"""
DeepfakeDetector — inference-only wrapper around the PyTorch model.

Migrated from deepfake-api/detector.py; uses model_registry, transforms, postprocessor.
Production enhancements:
  - Lazy model loading with initialization guard
  - GPU/CPU device robustness with memory management
  - Memory-safe inference with OOM/device error handling
  - Structured logging with inference timing
  - Device synchronization for consistent memory cleanup
"""

from __future__ import annotations

import logging
import traceback
import gc
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from core.model_registry import build_model, load_weights, resolve_weights_path
from core.postprocessor import logits_to_prediction
from data.transforms import get_inference_transform

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Wraps an EfficientNet-B0 binary classifier for deepfake image detection.
    
    Production features:
      - Lazy model loading (only loads on first predict() call)
      - Automatic GPU/CPU device selection with fallback
      - Memory-safe inference with explicit device synchronization
      - Structured exception handling with rich logging

    Usage:
        detector = DeepfakeDetector()
        label, confidence = detector.predict("path/to/image.png")
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        weights_path: Path | None = None,
        lazy_load: bool = True,
    ) -> None:
        """
        Initialize detector with device selection and optional lazy loading.
        
        Args:
            model_name: Model identifier (default: efficientnet_b0).
            weights_path: Path to model weights. If None, auto-resolved.
            lazy_load: If True, defer model loading until first predict() call.
        """
        self.model_name = model_name
        self.weights_path = resolve_weights_path(weights_path)
        self.lazy_load = lazy_load
        self._model_loaded = False
        
        # Device selection with explicit fallback
        self.device = self._select_device()
        self._transform = get_inference_transform()
        
        logger.info(
            "[detector.init] Device: %s | Model: %s | Weights: %s | Lazy load: %s",
            self.device,
            model_name,
            self.weights_path.exists(),
            lazy_load,
        )
        
        self.model: Optional[torch.nn.Module] = None
        
        # If eager loading, attempt load now
        if not lazy_load:
            self._ensure_model_loaded()
    
    def _select_device(self) -> torch.device:
        """
        Select compute device with fallback chain.
        
        Priority: CUDA (if available) → CPU.
        Logs device properties for debugging.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "[detector] CUDA available: %s | GPU: %s | Capability: %s",
                torch.cuda.is_available(),
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_capability(0),
            )
            return device
        
        logger.warning("[detector] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    def _ensure_model_loaded(self) -> None:
        """
        Load model on-demand (lazy) or validate it's loaded.
        
        Called on first predict() if lazy_load=True.
        Raises RuntimeError if weights file is missing.
        """
        if self._model_loaded:
            return
        
        if not self.weights_path.exists():
            msg = (
                f"Model weights not found: {self.weights_path}. "
                "Ensure weights file exists before inference."
            )
            logger.error("[detector.load] %s", msg)
            raise RuntimeError(msg)
        
        try:
            logger.info("[detector.load] Loading model: %s → %s", self.model_name, self.device)
            self.model = self._load_model()
            self._model_loaded = True
            logger.info("[detector.load] Model loaded successfully, in eval() mode")
        except Exception as exc:
            logger.error(
                "[detector.load] Failed to load model: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            self.model = None
            self._model_loaded = False
            raise

    def _load_model(self) -> torch.nn.Module:
        """
        Build and load model weights from disk.
        
        Returns:
            Model in eval mode on the selected device.
        """
        model = build_model(
            self.model_name,
            num_classes=2,
            pretrained=False,
        )
        return load_weights(model, self.weights_path, device=self.device)

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Open image file and return (1, 3, 224, 224) tensor.
        
        Applies EXIF correction and RGB conversion deterministically.
        
        Args:
            image_path: Path to image file.
        
        Returns:
            Batched tensor ready for inference.
        
        Raises:
            ValueError: If image cannot be opened or is corrupt.
        """
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
        
        Memory-safe inference:
          - Lazy model loading on first call
          - Explicit GPU sync before cleanup
          - OOM error handling with device reset
          - Automatic garbage collection

        Args:
            image_path: Path to image file.
        
        Returns:
            (label, confidence) — label is "Real"|"Fake", confidence is 0–100.
        
        Raises:
            RuntimeError: If model is not loaded or inference fails.
            ValueError: If image preprocessing fails.
        """
        # Ensure model is loaded (deferred if lazy_load=True)
        if not self._model_loaded:
            self._ensure_model_loaded()
        
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. "
                "Check the server startup logs for the weight loading error."
            )

        tensor = self._preprocess(image_path).to(self.device)
        logger.debug(
            "[detector.infer] Tensor: shape=%s dtype=%s device=%s",
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

        try:
            with torch.no_grad():
                logits = self.model(tensor)
            
            # Explicit device sync before cleanup
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        
        except torch.cuda.OutOfMemoryError as exc:
            # Reset GPU on OOM
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error("[detector.infer] GPU OOM during inference")
            raise RuntimeError(
                "GPU ran out of memory during inference. "
                "The image may still be too large despite pre-resizing."
            ) from exc
        
        except Exception as exc:
            logger.error(
                "[detector.infer] Inference failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            raise RuntimeError(
                f"Inference pipeline failed: {type(exc).__name__}: {exc}"
            ) from exc
        
        finally:
            # Cleanup tensors from device
            del tensor
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

        label, confidence = logits_to_prediction(logits)
        logger.info(
            "[detector.infer] Prediction: %s (confidence=%.2f%%) file=%r",
            label,
            confidence,
            image_path,
        )
        return label, confidence
