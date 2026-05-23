"""
data/dataset.py
───────────────
Robust PyTorch Dataset for binary deepfake image classification.

Supports two dataset layouts:

  FORMAT A — Class-subfolder layout (default, mirrors torchvision.datasets.ImageFolder):
      root/
        real/  *.jpg *.png *.webp
        fake/  *.jpg *.png *.webp

  FORMAT B — CSV manifest layout (for large-scale datasets with external splits):
      root/
        images/   *.jpg ...
      manifest.csv   columns: path (relative to root), label (0 or 1)

Label convention (consistent with detector.py):
  0 → Real
  1 → Fake

Design decisions
────────────────
• Images are NOT loaded into RAM on __init__. Only file paths are collected.
  This allows datasets of any size (FaceForensics++, DFDC have millions of frames).

• Corrupted images are SKIPPED gracefully with a warning log, not a crash.
  Training runs of hours/days must not die because of one bad file.

• PIL is used for all image I/O (consistent with the inference pipeline
  in core/preprocessor.py — same decode path means same distribution).

• pathlib.Path is used exclusively — no os.path anywhere.

• Strong type hints throughout for IDE support and mypy compliance.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Label index mapping — must match detector.py (index 0 = Real, index 1 = Fake)
LABEL_REAL: int = 0
LABEL_FAKE: int = 1

# Class name → label index
CLASS_TO_IDX: dict[str, int] = {
    "real": LABEL_REAL,
    "fake": LABEL_FAKE,
}

# Supported image extensions (lowercase)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp",
})

# Type alias for a dataset sample: (tensor, label)
Sample = tuple[torch.Tensor, int]


# ══════════════════════════════════════════════════════════════════════════════
# DEEPFAKE DATASET
# ══════════════════════════════════════════════════════════════════════════════

class DeepfakeDataset(Dataset[Sample]):
    """
    PyTorch Dataset for binary deepfake image classification.

    Supports class-subfolder layout and CSV manifest layout.
    Images are loaded lazily (one per __getitem__ call) — never held in RAM.

    Args:
        root:          Root directory of the dataset split (e.g. Path("data/train")).
                       For FORMAT A, this directory must contain "real/" and "fake/" subdirs.
                       For FORMAT B, this is the root directory referenced by the CSV.
        transform:     A callable that accepts a PIL Image and returns a torch.Tensor.
                       Use train_transforms(), val_transforms(), or inference_transforms()
                       from data/transforms.py.
        manifest_csv:  Optional path to a CSV file for FORMAT B layout.
                       If None, FORMAT A (class-subfolder) is used.
                       CSV must have columns: "path" (relative to root), "label" (0 or 1).
        allow_empty:   If False (default), raise ValueError when no valid images are found.
                       Set True only in tests where an empty dataset is intentional.

    Example (FORMAT A):
        dataset = DeepfakeDataset(
            root=Path("datasets/ff++/train"),
            transform=train_transforms(input_size=224),
        )

    Example (FORMAT B):
        dataset = DeepfakeDataset(
            root=Path("datasets/dfdc"),
            transform=val_transforms(input_size=224),
            manifest_csv=Path("datasets/dfdc/val.csv"),
        )
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        manifest_csv: Optional[Path] = None,
        allow_empty: bool = False,
    ) -> None:
        if not root.is_dir():
            raise FileNotFoundError(
                f"Dataset root directory does not exist: {root}"
            )

        self.root         = root
        self.transform    = transform
        self.manifest_csv = manifest_csv

        # Populate samples list: list of (absolute_path, label_int) tuples
        if manifest_csv is not None:
            self._samples: list[tuple[Path, int]] = self._load_from_csv(manifest_csv)
        else:
            self._samples = self._load_from_subfolders(root)

        if not self._samples and not allow_empty:
            raise ValueError(
                f"No valid images found in '{root}'. "
                f"Expected subfolders 'real/' and 'fake/' with "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))} files, "
                f"or a manifest CSV with valid paths."
            )

        # Build index → class name for external inspection (e.g. sampler)
        self.idx_to_class: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

        # Precompute label array as a plain Python list for O(1) access in sampler
        self._labels: list[int] = [label for _, label in self._samples]

        logger.info(
            "[dataset] Loaded %d samples from '%s'  (real=%d  fake=%d)",
            len(self._samples),
            root,
            self._labels.count(LABEL_REAL),
            self._labels.count(LABEL_FAKE),
        )

    # ── Format A: class-subfolder loading ────────────────────────────────────

    def _load_from_subfolders(self, root: Path) -> list[tuple[Path, int]]:
        """
        Discover images from class-named subdirectories under root.

        Expected structure:
            root/
              real/  (maps to label 0)
              fake/  (maps to label 1)

        Unknown subdirectory names are skipped with a warning.
        """
        samples: list[tuple[Path, int]] = []

        for class_name, label in CLASS_TO_IDX.items():
            class_dir = root / class_name
            if not class_dir.is_dir():
                logger.warning(
                    "[dataset] Class directory not found (skipping): %s", class_dir
                )
                continue

            found = 0
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append((path, label))
                    found += 1

            logger.debug(
                "[dataset] Class '%s' (label=%d): %d images in %s",
                class_name, label, found, class_dir,
            )

        return samples

    # ── Format B: CSV manifest loading ───────────────────────────────────────

    def _load_from_csv(self, csv_path: Path) -> list[tuple[Path, int]]:
        """
        Load image paths and labels from a CSV manifest file.

        Expected CSV columns:
          path:  relative path from self.root to the image file
          label: integer label (0 = real, 1 = fake)

        Rows referencing non-existent files are skipped with a warning.
        Rows with invalid labels raise ValueError.
        """
        if not csv_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

        samples: list[tuple[Path, int]] = []
        skipped = 0

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            # Validate that required columns are present
            if reader.fieldnames is None or not {"path", "label"}.issubset(
                set(reader.fieldnames)
            ):
                raise ValueError(
                    f"CSV '{csv_path}' must have 'path' and 'label' columns. "
                    f"Found: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    label = int(row["label"])
                except ValueError:
                    raise ValueError(
                        f"CSV row {row_num}: label '{row['label']}' is not a valid integer. "
                        f"Expected 0 (real) or 1 (fake)."
                    )

                if label not in (LABEL_REAL, LABEL_FAKE):
                    raise ValueError(
                        f"CSV row {row_num}: label {label} is not valid. "
                        f"Expected 0 (real) or 1 (fake)."
                    )

                img_path = self.root / row["path"].strip()
                if not img_path.is_file():
                    logger.warning(
                        "[dataset] CSV row %d: file not found (skipping): %s",
                        row_num, img_path,
                    )
                    skipped += 1
                    continue

                if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    logger.warning(
                        "[dataset] CSV row %d: unsupported extension '%s' (skipping): %s",
                        row_num, img_path.suffix, img_path,
                    )
                    skipped += 1
                    continue

                samples.append((img_path, label))

        if skipped:
            logger.warning("[dataset] CSV loading: skipped %d rows", skipped)

        return samples

    # ── PyTorch Dataset interface ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        """
        Load and return the image and label at the given index.

        If the image is corrupted or unreadable, the item is SKIPPED and the
        next valid item is returned. This prevents training runs from crashing
        due to isolated bad files in large datasets.

        A warning is logged for every skipped image so corrupt files can be
        identified and removed during dataset maintenance.

        Args:
            index: Integer index in [0, len(self) - 1].

        Returns:
            Tuple of (tensor, label) where tensor shape depends on the transform.
            With standard transforms: (torch.Tensor[3, H, W], int).
        """
        # Try up to len(self) times to find a loadable image.
        # In practice, datasets should have < 0.01% corrupt images, so
        # this loop almost never iterates more than once.
        attempts = 0
        current_index = index

        while attempts < len(self._samples):
            img_path, label = self._samples[current_index]

            try:
                img = self._load_image(img_path)
            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                logger.warning(
                    "[dataset] Skipping corrupt image (index=%d): %s — %s",
                    current_index, img_path, exc,
                )
                # Advance to the next index, wrapping around at the end
                current_index = (current_index + 1) % len(self._samples)
                attempts += 1
                continue

            # Apply the transform pipeline (resize, augment, normalise, to tensor)
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                # Fallback: return a plain float tensor without normalisation.
                # This is useful for debugging but should not be used in training.
                from torchvision.transforms.functional import to_tensor
                tensor = to_tensor(img)

            return tensor, label

        # If every image in the dataset is corrupt, raise a clear error.
        raise RuntimeError(
            f"Could not load any valid image after {len(self._samples)} attempts. "
            f"Dataset at '{self.root}' may be entirely corrupt."
        )

    # ── Image loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        """
        Open and validate a single image file.

        Steps:
          1. Open with PIL (lazy decode — header only at this point)
          2. Call load() to force full pixel decode (catches truncated files)
          3. Apply EXIF orientation correction
          4. Convert to RGB (handles RGBA, palette, grayscale, CMYK)

        Raises:
            UnidentifiedImageError: file is not a recognised image format
            OSError:                file is missing, unreadable, or truncated
            SyntaxError:            file is corrupt mid-stream (PIL raises this
                                    for some truncated JPEG files)
        """
        img = Image.open(path)

        # Force PIL to fully decode the image now.
        # Without this, truncated files pass PIL.open() silently and only
        # fail later (inside the transform pipeline, with a confusing error).
        img.load()

        # Correct EXIF rotation metadata.
        # Phone camera images store the rotation in EXIF but PIL does NOT
        # auto-rotate. Without this, portrait photos are sideways during training.
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)

        # Convert to RGB.
        # Dataset images may be RGBA (PNG with alpha), palette (GIF-style P mode),
        # grayscale (L), or CMYK. The model always expects 3-channel RGB input.
        img = img.convert("RGB")

        return img

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def labels(self) -> list[int]:
        """
        Return all labels as a flat list.

        Used by BalancedSampler to compute class weights without iterating
        through __getitem__ (which would load every image — expensive).
        """
        return self._labels

    @property
    def class_counts(self) -> dict[str, int]:
        """Return a mapping of class name → sample count."""
        real_count = self._labels.count(LABEL_REAL)
        fake_count = self._labels.count(LABEL_FAKE)
        return {
            "real": real_count,
            "fake": fake_count,
        }

    @property
    def samples(self) -> list[tuple[Path, int]]:
        """Return the full list of (path, label) tuples. Read-only view."""
        return list(self._samples)

    def __repr__(self) -> str:
        counts = self.class_counts
        return (
            f"{self.__class__.__name__}("
            f"root='{self.root}', "
            f"n_samples={len(self)}, "
            f"real={counts['real']}, "
            f"fake={counts['fake']})"
        )


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "DeepfakeDataset",
    "LABEL_REAL",
    "LABEL_FAKE",
    "CLASS_TO_IDX",
    "SUPPORTED_EXTENSIONS",
    "Sample",
]