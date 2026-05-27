"""
evaluation/evaluator.py
───────────────────────
Batch evaluation runner for trained deepfake classifiers.

WHY EVALUATION MUST MATCH INFERENCE PREPROCESSING EXACTLY
────────────────────────────────────────────────────────
The FastAPI pipeline applies two stages before the model sees a tensor:
  1. core.preprocessor.validate_and_prepare_image() — security validation,
     EXIF correction, RGB conversion, pre-inference resize (≤512 px)
  2. data.transforms.get_inference_transform() — resize to model input,
     ToTensor, ImageNet normalisation

If offline evaluation skips stage 1 and only uses torchvision transforms on
disk files, metrics will be optimistically biased (different pixel distribution)
and will not predict production behaviour. This module applies both stages on
every image, identical to core/detector.py.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from core.model_registry import build_model, load_weights, resolve_weights_path
from core.preprocessor import (
    DEFAULT_CONFIG,
    ImageValidationError,
    PreprocessorConfig,
    validate_and_prepare_image,
)
from data.dataset import DeepfakeDataset, LABEL_FAKE, LABEL_REAL
from data.transforms import get_inference_transform
from evaluation.metrics import MetricResult, compute_metrics, find_best_threshold

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test"]


@dataclass
class EvaluatorConfig:
    """Configuration for offline evaluation runs."""

    checkpoint_path: Path
    model_name: str = "efficientnet_b0"
    input_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    device: str = "auto"
    threshold: float = 0.5
    tune_threshold: bool = False
    threshold_metric: Literal["f1", "youden"] = "f1"
    seed: int = 42
    num_classes: int = 2
    preprocessor_config: PreprocessorConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    pin_memory: bool = True


class InferenceAlignedPreprocessor:
    """
    Replicates the production inference path for a single image file.

    Pipeline (same order as FastAPI + DeepfakeDetector):
        read bytes → validate_and_prepare_image → get_inference_transform
    """

    def __init__(
        self,
        input_size: int = 224,
        preprocessor_config: PreprocessorConfig = DEFAULT_CONFIG,
    ) -> None:
        self._preprocessor_config = preprocessor_config
        self._transform = get_inference_transform(input_size)

    def __call__(self, image_path: Path) -> torch.Tensor:
        file_bytes = image_path.read_bytes()
        img = validate_and_prepare_image(
            file_bytes,
            image_path.name,
            config=self._preprocessor_config,
        )
        return self._transform(img)


class _EvaluationPathDataset(Dataset):
    """
    Lightweight dataset over (path, label) pairs with API-aligned preprocessing.

    Does not modify data/dataset.py — reads paths from DeepfakeDataset.samples.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        preprocessor: InferenceAlignedPreprocessor,
    ) -> None:
        self._samples = samples
        self._preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        attempts = 0
        current = index

        while attempts < len(self._samples):
            path, label = self._samples[current]
            try:
                tensor = self._preprocessor(path)
                return tensor, label, str(path)
            except (ImageValidationError, UnidentifiedImageError, OSError, SyntaxError) as exc:
                logger.warning(
                    "[evaluator] Skipping unreadable image %s: %s", path, exc
                )
                current = (current + 1) % len(self._samples)
                attempts += 1

        raise RuntimeError(
            f"Could not preprocess any image after {len(self._samples)} attempts."
        )


def _collate_evaluation_batch(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    tensors = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return tensors, labels, paths


def set_deterministic_mode(seed: int) -> None:
    """Enable reproducible evaluation (no training randomness)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class SplitResult:
    """Raw predictions and metrics for one dataset split."""

    split: str
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray
    paths: list[str]
    metrics: MetricResult
    num_skipped: int = 0


class Evaluator:
    """
    Load a checkpoint and evaluate on train / val / test directory layouts.

    Uses DeepfakeDataset only to enumerate (path, label) pairs — preprocessing
    is performed by InferenceAlignedPreprocessor, not the dataset transform.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.preprocessor = InferenceAlignedPreprocessor(
            input_size=config.input_size,
            preprocessor_config=config.preprocessor_config,
        )
        self.model: Optional[torch.nn.Module] = None

    def load_model(self) -> torch.nn.Module:
        """Build architecture and load checkpoint weights."""
        checkpoint = Path(self.config.checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        model = build_model(
            self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=False,
        )
        model = load_weights(model, checkpoint, device=self.device)
        model.eval()
        self.model = model
        logger.info(
            "[evaluator] Model loaded: %s from %s on %s",
            self.config.model_name,
            checkpoint,
            self.device,
        )
        return model

    def _build_dataset(
        self,
        data_dir: Path,
        manifest_csv: Optional[Path] = None,
    ) -> _EvaluationPathDataset:
        """
        Enumerate samples via DeepfakeDataset (transform=None) then preprocess
        through the API-aligned pipeline in __getitem__.
        """
        catalog = DeepfakeDataset(
            root=data_dir.resolve(),
            transform=None,
            manifest_csv=manifest_csv.resolve() if manifest_csv else None,
        )
        return _EvaluationPathDataset(
            samples=catalog.samples,
            preprocessor=self.preprocessor,
        )

    @torch.inference_mode()
    def predict_dataloader(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run forward passes and collect fake-class probabilities."""
        if self.model is None:
            raise RuntimeError("Call load_model() before predict_dataloader().")

        all_labels: list[int] = []
        all_scores: list[float] = []
        all_paths: list[str] = []

        for batch_tensors, batch_labels, batch_paths in loader:
            batch_tensors = batch_tensors.to(self.device, non_blocking=True)
            logits = self.model(batch_tensors)
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, LABEL_FAKE].detach().cpu().numpy()

            all_labels.extend(batch_labels.numpy().tolist())
            all_scores.extend(fake_probs.tolist())
            all_paths.extend(batch_paths)

        y_true = np.asarray(all_labels, dtype=np.int64)
        y_score = np.asarray(all_scores, dtype=np.float64)
        return y_true, y_score, all_paths

    def evaluate_split(
        self,
        data_dir: Path,
        split: str,
        *,
        manifest_csv: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> SplitResult:
        """
        Evaluate one split directory (real/ + fake/ subfolders or CSV manifest).

        Args:
            data_dir:      Root of the split (e.g. data/val).
            split:         Name for reporting (train / val / test).
            manifest_csv:  Optional CSV manifest (FORMAT B).
            threshold:     Decision threshold on P(fake); defaults to config value.
        """
        if self.model is None:
            self.load_model()

        dataset = self._build_dataset(data_dir, manifest_csv)
        if len(dataset) == 0:
            raise ValueError(f"No samples found for split '{split}' in {data_dir}")

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
            drop_last=False,
            collate_fn=_collate_evaluation_batch,
        )

        logger.info(
            "[evaluator] Evaluating split=%r  dir=%s  samples=%d  batches=%d",
            split,
            data_dir,
            len(dataset),
            len(loader),
        )

        y_true, y_score, paths = self.predict_dataloader(loader)
        thr = self.config.threshold if threshold is None else threshold
        metrics = compute_metrics(y_true, y_score, threshold=thr)
        y_pred = (y_score >= thr).astype(np.int64)

        return SplitResult(
            split=split,
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
            paths=paths,
            metrics=metrics,
        )

    def run(
        self,
        splits: dict[str, Path],
        manifests: Optional[dict[str, Path]] = None,
    ) -> dict[str, SplitResult]:
        """
        Evaluate multiple splits. Optionally tune threshold on ``val`` first.

        Args:
            splits:    Mapping split name → directory path.
            manifests: Optional mapping split name → CSV manifest path.
        """
        set_deterministic_mode(self.config.seed)
        if self.model is None:
            self.load_model()

        manifests = manifests or {}
        results: dict[str, SplitResult] = {}
        tuned_threshold: Optional[float] = None

        # Threshold tuning on validation split before test evaluation
        if self.config.tune_threshold and "val" in splits:
            val_result = self.evaluate_split(
                splits["val"],
                "val",
                manifest_csv=manifests.get("val"),
            )
            tuned_threshold, _ = find_best_threshold(
                val_result.y_true,
                val_result.y_score,
                metric=self.config.threshold_metric,
            )
            logger.info(
                "[evaluator] Tuned threshold on val: %.4f (%s)",
                tuned_threshold,
                self.config.threshold_metric,
            )
            val_result = self.evaluate_split(
                splits["val"],
                "val",
                manifest_csv=manifests.get("val"),
                threshold=tuned_threshold,
            )
            results["val"] = val_result

        for name, data_dir in splits.items():
            if name == "val" and name in results:
                continue
            thr = tuned_threshold if tuned_threshold is not None else None
            results[name] = self.evaluate_split(
                data_dir,
                name,
                manifest_csv=manifests.get(name),
                threshold=thr,
            )

        return results


def default_checkpoint_path(explicit: Optional[Path] = None) -> Path:
    """Resolve checkpoint — explicit path or registry search."""
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    resolved = resolve_weights_path()
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint or place weights in "
            "checkpoints/deepfake_model.pth"
        )
    return resolved


__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "InferenceAlignedPreprocessor",
    "SplitResult",
    "set_deterministic_mode",
    "resolve_device",
    "default_checkpoint_path",
]
