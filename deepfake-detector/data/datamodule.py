"""
data/datamodule.py
──────────────────
DataLoader factory for the deepfake detection training and evaluation pipeline.

Responsibilities
────────────────
• Construct train, validation, and test DataLoaders from configured paths.
• Apply the correct transform pipeline to each split.
• Attach a BalancedSampler to the training loader if the dataset is imbalanced.
• Expose GPU-friendly DataLoader settings (pin_memory, persistent_workers).
• Provide a deterministic validation/test loader (no shuffle, no augmentation).
• Keep all configuration injectable — no hardcoded paths.

DataLoader configuration decisions
────────────────────────────────────
pin_memory=True:
    Pins CPU memory, allowing faster host→GPU transfers via DMA.
    Should be True whenever training on GPU. Safe to set always — ignored on CPU.

persistent_workers=True:
    Worker processes persist across batches rather than being respawned.
    Eliminates the per-epoch worker spawn overhead (significant on Windows
    where fork() is not available and spawn is slow).
    Requires num_workers > 0.

prefetch_factor=2 (default):
    Each worker pre-fetches this many batches ahead. Keeps GPU fed.
    Increase to 4 if GPU is idle between batches (data bottleneck).

drop_last=True (training only):
    The last batch may be smaller than batch_size (incomplete batch).
    Incomplete batches can cause issues with BatchNorm statistics during
    training. We drop them. Val/test use drop_last=False so no samples
    are missed during evaluation.

shuffle=True (training only):
    Shuffles sample order each epoch to prevent the model memorising
    temporal or filesystem ordering patterns.
    IMPORTANT: shuffle=False when using BalancedSampler — they conflict.
    The DataModule handles this automatically based on whether balancing
    is enabled.

num_workers:
    Set to 0 on Windows when debugging (CUDA + multiprocessing requires
    `if __name__ == "__main__":` guards). Production Linux: 4–8 workers.
    The DataModule accepts this as a parameter — never hardcoded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import DeepfakeDataset
from data.sampler import BalancedSampler, build_sampler
from data.transforms import (
    DEFAULT_INPUT_SIZE,
    inference_transforms,
    train_transforms,
    val_transforms,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATAMODULE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class DataConfig:
    """
    Typed configuration container for the DataModule.

    All values have sensible defaults. Override via constructor arguments
    or by passing a config dict from Hydra/OmegaConf.

    Attributes:
        train_dir:          Path to the training split root.
        val_dir:            Path to the validation split root.
        test_dir:           Optional path to the test split root.
        train_manifest:     Optional CSV manifest for training split.
        val_manifest:       Optional CSV manifest for validation split.
        test_manifest:      Optional CSV manifest for test split.
        input_size:         Model input resolution (224 for B0, 380 for B4).
        batch_size:         Samples per batch.
        num_workers:        DataLoader worker processes.
        pin_memory:         Pin CPU memory for faster GPU transfers.
        persistent_workers: Keep worker processes alive between epochs.
        prefetch_factor:    Batches to prefetch per worker.
        use_balanced_sampler: Enable class-balanced sampling for training.
        imbalance_threshold:  Ratio above which BALANCED strategy is chosen.
        seed:               Random seed for reproducible train loader shuffling.
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        test_dir: Optional[Path] = None,
        train_manifest: Optional[Path] = None,
        val_manifest: Optional[Path] = None,
        test_manifest: Optional[Path] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        use_balanced_sampler: bool = True,
        imbalance_threshold: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.train_dir           = Path(train_dir)
        self.val_dir             = Path(val_dir)
        self.test_dir            = Path(test_dir) if test_dir is not None else None
        self.train_manifest      = Path(train_manifest) if train_manifest else None
        self.val_manifest        = Path(val_manifest) if val_manifest else None
        self.test_manifest       = Path(test_manifest) if test_manifest else None
        self.input_size          = input_size
        self.batch_size          = batch_size
        self.num_workers         = num_workers
        self.pin_memory          = pin_memory
        self.persistent_workers  = persistent_workers and (num_workers > 0)
        self.prefetch_factor     = prefetch_factor if num_workers > 0 else None
        self.use_balanced_sampler = use_balanced_sampler
        self.imbalance_threshold = imbalance_threshold
        self.seed                = seed

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        """
        Build a DataConfig from a plain dictionary (e.g. from OmegaConf).

        Path-typed fields are converted automatically.
        """
        path_fields = {
            "train_dir", "val_dir", "test_dir",
            "train_manifest", "val_manifest", "test_manifest",
        }
        converted = {
            k: (Path(v) if k in path_fields and v is not None else v)
            for k, v in d.items()
        }
        return cls(**converted)

    def __repr__(self) -> str:
        return (
            f"DataConfig("
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, "
            f"input_size={self.input_size}, "
            f"balanced={self.use_balanced_sampler})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DATAMODULE
# ══════════════════════════════════════════════════════════════════════════════

class DeepfakeDataModule:
    """
    DataLoader factory for the deepfake detection pipeline.

    Creates train, val, and (optionally) test DataLoaders with correct
    transforms, sampling, and GPU-optimised DataLoader settings.

    Usage:
        config = DataConfig(
            train_dir=Path("data/train"),
            val_dir=Path("data/val"),
            batch_size=32,
            num_workers=4,
        )
        dm = DeepfakeDataModule(config)
        dm.setup()

        for batch_tensors, batch_labels in dm.train_dataloader():
            ...

    Args:
        config: A DataConfig instance with all loader settings.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

        # Datasets — populated in setup()
        self._train_dataset: Optional[DeepfakeDataset] = None
        self._val_dataset:   Optional[DeepfakeDataset] = None
        self._test_dataset:  Optional[DeepfakeDataset] = None

        # Sampler — populated in setup() if use_balanced_sampler is True
        self._train_sampler: Optional[BalancedSampler] = None

        # Reproducibility seed for training DataLoader
        self._generator = torch.Generator()
        self._generator.manual_seed(config.seed)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate datasets for the requested stage.

        Args:
            stage: One of "fit", "validate", "test", or None (all stages).
                   Mirrors PyTorch Lightning's DataModule.setup() convention.
                   None builds all available datasets.
        """
        build_train = stage in (None, "fit")
        build_val   = stage in (None, "fit", "validate")
        build_test  = stage in (None, "test") and self.config.test_dir is not None

        if build_train:
            self._train_dataset = DeepfakeDataset(
                root=self.config.train_dir,
                transform=train_transforms(self.config.input_size),
                manifest_csv=self.config.train_manifest,
            )
            logger.info("[datamodule] Train dataset: %s", self._train_dataset)

            # Build sampler if requested
            if self.config.use_balanced_sampler:
                self._train_sampler = build_sampler(
                    labels=self._train_dataset.labels,
                    imbalance_threshold=self.config.imbalance_threshold,
                    generator=self._generator,
                )
                logger.info(
                    "[datamodule] Train sampler: %s", self._train_sampler
                )

        if build_val:
            self._val_dataset = DeepfakeDataset(
                root=self.config.val_dir,
                transform=val_transforms(self.config.input_size),
                manifest_csv=self.config.val_manifest,
            )
            logger.info("[datamodule] Val dataset: %s", self._val_dataset)

        if build_test and self.config.test_dir is not None:
            self._test_dataset = DeepfakeDataset(
                root=self.config.test_dir,
                transform=val_transforms(self.config.input_size),
                manifest_csv=self.config.test_manifest,
            )
            logger.info("[datamodule] Test dataset: %s", self._test_dataset)

    # ── DataLoader factories ──────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        """
        Return the training DataLoader.

        Configuration:
          • shuffle=True when no sampler (random ordering per epoch)
          • shuffle=False when BalancedSampler is used (they conflict)
          • drop_last=True — prevents incomplete final batch
          • pin_memory=True — faster host→GPU transfers
          • persistent_workers=True — avoids per-epoch worker respawn
        """
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader().")

        # shuffle and sampler are mutually exclusive in PyTorch DataLoader
        use_sampler = self._train_sampler is not None
        shuffle     = not use_sampler

        loader = DataLoader(
            dataset            = self._train_dataset,
            batch_size         = self.config.batch_size,
            shuffle            = shuffle,
            sampler            = self._train_sampler,
            num_workers        = self.config.num_workers,
            pin_memory         = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor    = self.config.prefetch_factor,
            drop_last          = True,   # avoid incomplete final batch
            generator          = self._generator if not use_sampler else None,
            worker_init_fn     = _worker_init_fn,
        )

        logger.info(
            "[datamodule] Train loader: %d samples  %d batches  "
            "sampler=%s  shuffle=%s",
            len(self._train_dataset),
            len(loader),
            "BalancedSampler" if use_sampler else "None",
            shuffle,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation DataLoader.

        Configuration:
          • shuffle=False — deterministic evaluation (same order every run)
          • drop_last=False — evaluate on ALL samples
          • No sampler — natural class distribution for metric computation
        """
        if self._val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader().")

        loader = DataLoader(
            dataset            = self._val_dataset,
            batch_size         = self.config.batch_size,
            shuffle            = False,   # MUST be deterministic for evaluation
            sampler            = None,    # natural distribution — no rebalancing
            num_workers        = self.config.num_workers,
            pin_memory         = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor    = self.config.prefetch_factor,
            drop_last          = False,   # evaluate every sample
            worker_init_fn     = _worker_init_fn,
        )

        logger.info(
            "[datamodule] Val loader: %d samples  %d batches",
            len(self._val_dataset),
            len(loader),
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """
        Return the test DataLoader.

        Identical settings to val_dataloader() — deterministic,
        no rebalancing, every sample evaluated exactly once.
        """
        if self._test_dataset is None:
            raise RuntimeError(
                "Test dataset not available. "
                "Either setup(stage='test') was not called or test_dir is None."
            )

        loader = DataLoader(
            dataset            = self._test_dataset,
            batch_size         = self.config.batch_size,
            shuffle            = False,
            sampler            = None,
            num_workers        = self.config.num_workers,
            pin_memory         = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor    = self.config.prefetch_factor,
            drop_last          = False,
            worker_init_fn     = _worker_init_fn,
        )

        logger.info(
            "[datamodule] Test loader: %d samples  %d batches",
            len(self._test_dataset),
            len(loader),
        )
        return loader

    def inference_dataloader(
        self,
        dataset_root: Path,
        manifest_csv: Optional[Path] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Build a DataLoader for a new (unlabelled or labelled) inference set.

        Uses inference_transforms() — identical to val_transforms() but
        semantically distinct (see transforms.py docstring).

        Args:
            dataset_root: Root directory of the inference dataset.
            manifest_csv: Optional CSV manifest.
            batch_size:   Override the config batch size for inference.
                          Useful when running on a CPU server where smaller
                          batches avoid memory issues.

        Returns:
            A deterministic DataLoader suitable for production batch inference.
        """
        infer_dataset = DeepfakeDataset(
            root=dataset_root,
            transform=inference_transforms(self.config.input_size),
            manifest_csv=manifest_csv,
        )

        effective_batch = batch_size if batch_size is not None else self.config.batch_size

        return DataLoader(
            dataset            = infer_dataset,
            batch_size         = effective_batch,
            shuffle            = False,
            num_workers        = self.config.num_workers,
            pin_memory         = self.config.pin_memory,
            persistent_workers = self.config.persistent_workers,
            prefetch_factor    = self.config.prefetch_factor,
            drop_last          = False,
            worker_init_fn     = _worker_init_fn,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def train_dataset(self) -> DeepfakeDataset:
        if self._train_dataset is None:
            raise RuntimeError("setup() has not been called yet.")
        return self._train_dataset

    @property
    def val_dataset(self) -> DeepfakeDataset:
        if self._val_dataset is None:
            raise RuntimeError("setup() has not been called yet.")
        return self._val_dataset

    @property
    def test_dataset(self) -> Optional[DeepfakeDataset]:
        return self._test_dataset

    def __repr__(self) -> str:
        return (
            f"DeepfakeDataModule("
            f"config={self.config!r}, "
            f"train_ready={self._train_dataset is not None}, "
            f"val_ready={self._val_dataset is not None}, "
            f"test_ready={self._test_dataset is not None})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# WORKER INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _worker_init_fn(worker_id: int) -> None:
    """
    Initialise each DataLoader worker with a unique random seed.

    Without this, all workers share the same RNG state and will produce
    identical augmentation sequences, effectively multiplying duplicates
    rather than introducing diversity.

    The seed formula (base_seed + worker_id) ensures:
      • Each worker has a unique seed
      • Seeds are deterministic across runs given the same base seed
      • Seeds vary between workers in the same run
    """
    base_seed = torch.initial_seed() % (2 ** 32)
    worker_seed = base_seed + worker_id

    import random
    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # torch's per-worker seed is already set by PyTorch internally;
    # we set random and numpy seeds to match it.


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "DeepfakeDataModule",
    "DataConfig",
]