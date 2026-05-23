"""
data/transforms.py
──────────────────
Three separate transform pipelines for the three distinct phases:
  • train_transforms()     — augmented, for training splits
  • val_transforms()       — deterministic, for validation and test splits
  • inference_transforms() — deterministic, for production API inference

DEEPFAKE-SPECIFIC AUGMENTATION RATIONALE
─────────────────────────────────────────
Deepfake detection is fundamentally different from general image classification.
The signals the model learns are subtle forensic artifacts:
  • GAN-generated frequency artifacts (grid patterns in DCT domain)
  • Blending seam inconsistencies at face boundaries
  • Unnatural sharpness or smoothness at face edges
  • Color statistics that differ from camera sensor noise
  • Inconsistent JPEG compression block artifacts

This means augmentations that DESTROY these signals must be avoided,
even if they improve generalisation in standard vision benchmarks.

Augmentations used and why
──────────────────────────
✔ RandomHorizontalFlip(p=0.5)
    Faces appear mirrored in both real and fake datasets equally.
    Does not destroy any forensic artifacts. Standard and safe.

✔ ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
    Very mild. GAN generators introduce systematic color deviations that
    are real detection signals. Heavy color jitter would wash these out.
    Mild jitter improves generalisation across different camera sensors.

✔ RandomGrayscale(p=0.02)
    Very rare (2%). Some forensic detectors work on luminance only.
    A tiny probability forces the model not to over-rely on color.

✔ GaussianBlur(kernel_size, sigma=(0.1, 1.0))
    Deepfakes often have unnaturally sharp or unnaturally blurred
    face regions. Mild blur augmentation prevents the model from using
    raw pixel sharpness as the only signal, improving robustness.

✔ RandomApply([JPEG compression simulation], p=0.5)
    THE MOST IMPORTANT AUGMENTATION for deepfake detection.
    Real-world images are almost always re-compressed (social media,
    messaging apps, email). JPEG compression partially destroys DCT
    artifacts in both real and fake images. Without this augmentation,
    models trained on uncompressed data fail catastrophically on
    compressed inputs — a well-documented failure mode in DFDC results.
    We simulate JPEG compression using PIL's save/reload with quality 70-95.

✗ RandomCrop / CenterCrop (AVOIDED)
    Face boundary regions contain critical blending artifacts.
    Cropping away portions of the face removes the most discriminative
    forensic evidence. We use Resize only — never crop in training.

✗ RandomErasing / Cutout / CutMix (AVOIDED)
    Masks pixel regions that may contain key forgery artifacts.
    The model must see the whole face to detect blending seams.

✗ Elastic distortion / GridDistortion (AVOIDED)
    Destroys facial geometry consistency, which is a primary signal
    distinguishing real from GAN-generated faces.

✗ Heavy brightness/contrast (AVOIDED)
    Color and brightness statistics carry forensic signal.
    Changes beyond ±10% risk washing out generative model signatures.

✗ Mixup / CutMix label mixing (AVOIDED)
    Binary calibrated probabilities matter for the confidence score.
    Soft labels degrade probability calibration on the output head.
"""

from __future__ import annotations

import io
import random
from typing import Callable

import numpy as np
from PIL import Image
from torchvision import transforms


# ── ImageNet statistics ────────────────────────────────────────────────────────
# Required for all EfficientNet variants — they use ImageNet-pretrained backbones.
# These values must be IDENTICAL across train / val / inference pipelines.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# ── Default model input size ───────────────────────────────────────────────────
# EfficientNet-B0 canonical input: 224 × 224
# EfficientNet-B4 canonical input: 380 × 380
# Passed as a parameter so this file supports multiple architectures.
DEFAULT_INPUT_SIZE: int = 224


# ══════════════════════════════════════════════════════════════════════════════
# JPEG COMPRESSION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class RandomJPEGCompression:
    """
    Simulate JPEG re-compression by encoding to JPEG in memory and decoding back.

    This is the single most important augmentation for deepfake detection.

    Real-world images shared via social media, messaging apps, or email are
    almost always re-compressed. JPEG compression introduces DCT block artefacts
    that partially obscure GAN-generated frequency patterns.

    Without this augmentation, models trained on raw/lossless images will see
    a distribution shift at inference time (compressed images look different in
    the frequency domain) and will underperform significantly.

    Args:
        quality_range: (min_quality, max_quality) — tuple of JPEG quality values.
                       Higher = less compression. Range 70–95 mimics social media.
        p:             Probability of applying compression. 0.5 is recommended.
    """

    def __init__(
        self,
        quality_range: tuple[int, int] = (70, 95),
        p: float = 0.5,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if quality_range[0] > quality_range[1]:
            raise ValueError("quality_range[0] must be <= quality_range[1]")

        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_range[0], self.quality_range[1])

        # Encode to JPEG bytes in memory, then decode back to PIL Image.
        # This is identical to what happens when an image is saved and re-opened
        # by a browser, messaging app, or social media platform.
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()  # .copy() detaches from the BytesIO buffer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, p={self.p})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def train_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Augmented transform pipeline for the training split.

    Augmentations are ordered deliberately:
      1. PIL-space augmentations first (before tensor conversion)
      2. JPEG simulation last in PIL space (must happen before ToTensor)
      3. ToTensor
      4. Normalise

    Args:
        input_size: Spatial resolution to resize to (height = width).
                    224 for EfficientNet-B0, 380 for B4.

    Returns:
        A composed torchvision transform callable.
    """
    # GaussianBlur kernel size must be odd and <= image size.
    # At 224px, kernel=5 gives a subtle blur without destroying edge artefacts.
    blur_kernel = _odd_kernel(max(3, input_size // 45))

    return transforms.Compose([
        # ── Spatial: resize to model input size ───────────────────────────────
        # We use LANCZOS (high-quality anti-aliased downsampling) to preserve
        # fine-grained frequency artefacts that are lost with bilinear/bicubic.
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),

        # ── Geometric: horizontal flip only ───────────────────────────────────
        # Safe for deepfake detection — see module docstring.
        transforms.RandomHorizontalFlip(p=0.5),

        # ── Photometric: very mild color jitter ───────────────────────────────
        # Improves sensor/camera generalisation without destroying color signals.
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,          # very tight hue range — GAN color bias is a signal
        ),

        # ── Photometric: rare grayscale ────────────────────────────────────────
        # Forces the model not to rely solely on color. Very low probability.
        transforms.RandomGrayscale(p=0.02),

        # ── Blur: mild Gaussian blur ───────────────────────────────────────────
        # Deepfakes often have unnatural sharpness. Mild blur prevents the model
        # from using raw sharpness as its only discriminator.
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 1.0))],
            p=0.3,
        ),

        # ── Compression: JPEG simulation ──────────────────────────────────────
        # THE critical augmentation. Must happen in PIL space before ToTensor.
        RandomJPEGCompression(quality_range=(70, 95), p=0.5),

        # ── Tensor conversion ─────────────────────────────────────────────────
        transforms.ToTensor(),

        # ── Normalise with ImageNet statistics ────────────────────────────────
        # Required because the EfficientNet backbone was pretrained on ImageNet.
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION / TEST TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def val_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Deterministic transform pipeline for validation and test splits.

    No augmentation whatsoever — evaluation must be reproducible.
    The spatial and photometric operations are IDENTICAL to inference_transforms()
    to ensure no train/val distribution gap exists in the preprocessing chain.

    Args:
        input_size: Must match the value used in train_transforms().

    Returns:
        A composed torchvision transform callable.
    """
    return transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def inference_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Deterministic transform pipeline for production API inference.

    Identical to val_transforms(). Defined separately so that:
      1. The API (core/preprocessor.py) imports from this module directly,
         making the preprocessing contract explicit and auditable.
      2. Future inference-specific changes (e.g. TTA — test-time augmentation)
         can be applied here without touching validation preprocessing.

    Args:
        input_size: Must match the value used during training.

    Returns:
        A composed torchvision transform callable.
    """
    return transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Production inference transform — alias for ``inference_transforms()``.

    Used by ``core.detector.DeepfakeDetector`` at API startup. Keeps the
    original migration name stable while ``inference_transforms()`` is the
    canonical name used by ``data.datamodule`` and training docs.
    """
    return inference_transforms(input_size=input_size)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _odd_kernel(size: int) -> int:
    """
    Return the nearest odd integer >= size.
    GaussianBlur requires an odd kernel size.
    """
    return size if size % 2 == 1 else size + 1


def denormalise(tensor: "torch.Tensor") -> "torch.Tensor":
    """
    Reverse ImageNet normalisation for visualisation purposes.

    Args:
        tensor: Normalised tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        Tensor with pixel values in [0, 1].
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 4:
        # Batch dimension present: (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std  = std.view(1, 3, 1, 1)
    else:
        # Single image: (C, H, W)
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "train_transforms",
    "val_transforms",
    "inference_transforms",
    "get_inference_transform",
    "RandomJPEGCompression",
    "denormalise",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_INPUT_SIZE",
]