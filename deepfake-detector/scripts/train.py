"""
scripts/train.py
────────────────
Production training entrypoint for the deepfake detection system.

Usage examples
──────────────
Minimal (all defaults):
    python scripts/train.py \
        --train-dir data/train \
        --val-dir   data/val

With custom config:
    python scripts/train.py \
        --train-dir   data/train \
        --val-dir     data/val \
        --epochs      30 \
        --batch-size  32 \
        --lr          3e-4 \
        --loss        focal \
        --scheduler   cosine \
        --checkpoint-dir checkpoints/run_001 \
        --phase1-epochs  2

Resume interrupted training:
    python scripts/train.py \
        --train-dir   data/train \
        --val-dir     data/val \
        --resume      checkpoints/run_001/last.pth

Freeze backbone for head-only training (Phase 1 manually):
    python scripts/train.py \
        --train-dir       data/train \
        --val-dir         data/val \
        --freeze-backbone \
        --phase1-epochs   0 \
        --epochs          5

Disable AMP (CPU or debugging):
    python scripts/train.py \
        --train-dir data/train \
        --val-dir   data/val \
        --no-amp

Expected dataset structure (default FORMAT A):
    data/
      train/
        real/  *.jpg *.png *.webp
        fake/  *.jpg *.png *.webp
      val/
        real/
        fake/

Or pass --train-manifest / --val-manifest for CSV FORMAT B.

DETERMINISTIC SEEDS
────────────────────
--seed sets:
  • torch.manual_seed
  • torch.cuda.manual_seed_all
  • Python random.seed
  • numpy random seed
  • torch.backends.cudnn.deterministic = True
  • torch.backends.cudnn.benchmark = False  (set True with --no-deterministic for speed)

Fully deterministic training is ~5–10% slower due to disabled cuDNN heuristics.
For production training where reproducibility matters, use --seed.
For maximum speed, add --no-deterministic (results will still be consistent
across runs with the same seed, but cuDNN may use non-deterministic algorithms).

GPU DETECTION
──────────────
Device is selected automatically:
  1. CUDA GPU if available
  2. Apple MPS if available (M1/M2 Mac)
  3. CPU fallback

Use --device cpu/cuda/mps to override.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on sys.path regardless of working directory ────────
# This allows: python scripts/train.py  from the project root
# AND:         python train.py  from the scripts/ directory
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from core import model_registry
from core.model_registry import ModelConfig
from data.datamodule import DataConfig, DeepfakeDataModule
from training.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from training.losses import build_loss
from training.optimizers import build_optimizer, build_scheduler
from training.trainer import Trainer, TrainingConfig


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging(log_level: str, log_file: Path | None = None) -> None:
    """
    Configure root logger with console + optional file output.

    Format is consistent with app.py so logs from training and serving
    look identical in aggregation dashboards.
    """
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    level   = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any existing basicConfig from imported modules
    )


logger = logging.getLogger("train")


# ══════════════════════════════════════════════════════════════════════════════
# SEED + DETERMINISM
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducible training.

    Args:
        seed:          Integer seed value. Use the same seed across runs
                       to reproduce results exactly.
        deterministic: If True, forces cuDNN to use deterministic algorithms.
                       ~5–10% speed penalty but guarantees bit-identical results.
                       Set False for maximum throughput when exact reproducibility
                       is not required.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # CUBLAS_WORKSPACE_CONFIG required for deterministic CUDA ops in PyTorch >= 1.8
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # older PyTorch — no-op
    else:
        # benchmark=True lets cuDNN profile and select the fastest conv algorithm.
        # This produces faster training but non-deterministic results across runs.
        torch.backends.cudnn.benchmark = True

    logger.info(
        "[seed] seed=%d  deterministic=%s  benchmark=%s",
        seed,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_device(requested: str | None) -> torch.device:
    """
    Select the best available compute device.

    Priority when requested=None:
      1. CUDA (NVIDIA GPU)
      2. MPS  (Apple Silicon)
      3. CPU  (fallback)

    Args:
        requested: "cuda", "mps", "cpu", or None for auto-detect.

    Returns:
        torch.device instance.
    """
    if requested is not None:
        device = torch.device(requested)
        logger.info("[device] Forced: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        logger.info(
            "[device] CUDA: %s  VRAM: %.1f GB  CUDA cap: %d.%d",
            props.name,
            props.total_memory / 1024 ** 3,
            props.major, props.minor,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("[device] Apple MPS selected")
    else:
        device = torch.device("cpu")
        logger.warning(
            "[device] No GPU found — training on CPU. "
            "This will be slow for large datasets."
        )

    return device


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train the DeepFake Detector AI model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument("--train-dir",      type=Path, required=True,
                      help="Root directory of training split (contains real/ and fake/ subdirs)")
    data.add_argument("--val-dir",        type=Path, required=True,
                      help="Root directory of validation split")
    data.add_argument("--test-dir",       type=Path, default=None,
                      help="Optional test split directory")
    data.add_argument("--train-manifest", type=Path, default=None,
                      help="Optional CSV manifest for training split (FORMAT B)")
    data.add_argument("--val-manifest",   type=Path, default=None,
                      help="Optional CSV manifest for validation split (FORMAT B)")
    data.add_argument("--num-workers",    type=int,  default=4,
                      help="DataLoader worker processes (0 for debug on Windows)")
    data.add_argument("--no-balanced-sampler", action="store_true",
                      help="Disable class-balanced sampling (use natural distribution)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = p.add_argument_group("Model")
    model.add_argument("--model-name",       type=str,   default="efficientnet_b0",
                       choices=model_registry.list_available(),
                       help="Architecture to train (must be registered in model_registry)")
    model.add_argument("--dropout",          type=float, default=0.2,
                       help="Dropout rate before the classification head")
    model.add_argument("--no-pretrained",    action="store_true",
                       help="Skip ImageNet pretrained weights (ablation only)")
    model.add_argument("--freeze-backbone",  action="store_true",
                       help="Freeze backbone at init (useful for manual Phase 1 control)")
    model.add_argument("--drop-connect-rate",type=float, default=0.2,
                       help="Stochastic depth rate for EfficientNet MBConv blocks")

    # ── Training ──────────────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--epochs",           type=int,   default=30,
                       help="Maximum training epochs")
    train.add_argument("--batch-size",       type=int,   default=32,
                       help="Samples per batch")
    train.add_argument("--phase1-epochs",    type=int,   default=2,
                       help="Epochs with frozen backbone (0 = skip Phase 1)")
    train.add_argument("--grad-clip",        type=float, default=1.0,
                       help="Max gradient norm for clipping (0 = disabled)")
    train.add_argument("--no-amp",           action="store_true",
                       help="Disable mixed precision training (force float32)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = p.add_argument_group("Optimizer")
    opt.add_argument("--optimizer",          type=str,   default="adamw",
                     choices=["adamw", "sgd"],
                     help="Optimizer algorithm")
    opt.add_argument("--lr",                 type=float, default=3e-4,
                     help="Learning rate for the classifier head")
    opt.add_argument("--weight-decay",       type=float, default=1e-4,
                     help="L2 regularisation coefficient")
    opt.add_argument("--backbone-lr-mult",   type=float, default=0.1,
                     help="LR multiplier for backbone in Phase 2 (backbone LR = lr × mult)")
    opt.add_argument("--scheduler",          type=str,   default="cosine",
                     choices=["cosine", "cosine_warmup", "plateau"],
                     help="LR scheduler")
    opt.add_argument("--warmup-steps",       type=int,   default=500,
                     help="Warmup steps for cosine_warmup scheduler")
    opt.add_argument("--min-lr",             type=float, default=1e-7,
                     help="Minimum LR floor for cosine annealing")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss = p.add_argument_group("Loss")
    loss.add_argument("--loss",              type=str,   default="bce",
                      choices=["bce", "focal", "label_smoothing"],
                      help="Loss function")
    loss.add_argument("--focal-gamma",       type=float, default=2.0,
                      help="Focal loss gamma (focusing parameter)")
    loss.add_argument("--focal-alpha",       type=float, default=0.25,
                      help="Focal loss alpha (class weight for positive/Fake class)")
    loss.add_argument("--label-smoothing",   type=float, default=0.1,
                      help="Label smoothing factor for label_smoothing loss")
    loss.add_argument("--pos-weight",        type=float, default=None,
                      help="BCE positive class weight (n_real/n_fake for imbalanced data)")

    # ── Early stopping ────────────────────────────────────────────────────────
    es = p.add_argument_group("Early Stopping")
    es.add_argument("--patience",            type=int,   default=7,
                    help="Epochs without val_auc improvement before stopping")
    es.add_argument("--min-delta",           type=float, default=0.001,
                    help="Minimum improvement to reset patience counter")

    # ── Checkpointing + output ─────────────────────────────────────────────────
    ckpt = p.add_argument_group("Checkpointing")
    ckpt.add_argument("--checkpoint-dir",    type=Path,
                      default=Path("checkpoints") / datetime.now().strftime("%Y%m%d_%H%M%S"),
                      help="Directory to save checkpoints and logs")
    ckpt.add_argument("--resume",            type=Path,  default=None,
                      help="Path to last.pth checkpoint to resume training from")

    # ── Reproducibility + environment ─────────────────────────────────────────
    env = p.add_argument_group("Environment")
    env.add_argument("--seed",               type=int,   default=42,
                     help="Random seed for reproducibility")
    env.add_argument("--no-deterministic",   action="store_true",
                     help="Enable cuDNN benchmark mode (faster but non-deterministic)")
    env.add_argument("--device",             type=str,   default=None,
                     help="Force device: cuda / mps / cpu (auto-detect if omitted)")
    env.add_argument("--log-level",          type=str,   default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging verbosity")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    """
    Full training pipeline entry point.

    Can be called programmatically (pass argv=[...]) or from the command line.
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file = Path(args.checkpoint_dir) / "train.log"
    _setup_logging(args.log_level, log_file=log_file)

    logger.info("=" * 65)
    logger.info("DeepFake Detector AI — Training Run")
    logger.info("=" * 65)
    logger.info("Arguments: %s", vars(args))

    # ── Seed ──────────────────────────────────────────────────────────────────
    _set_seed(args.seed, deterministic=not args.no_deterministic)

    # ── Device ────────────────────────────────────────────────────────────────
    device = _select_device(args.device)

    # ── Validate paths ────────────────────────────────────────────────────────
    for name, path in [("--train-dir", args.train_dir), ("--val-dir", args.val_dir)]:
        if not path.is_dir():
            logger.error("%s does not exist or is not a directory: %s", name, path)
            sys.exit(1)

    if args.resume is not None and not args.resume.is_file():
        logger.error("--resume path does not exist: %s", args.resume)
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    input_size = model_registry.get_input_size(args.model_name)

    model_config = ModelConfig(
        name              = args.model_name,
        pretrained        = not args.no_pretrained,
        dropout_rate      = args.dropout,
        freeze_backbone   = args.freeze_backbone or (args.phase1_epochs > 0),
        drop_connect_rate = args.drop_connect_rate,
        input_size        = input_size,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data_config = DataConfig(
        train_dir            = args.train_dir,
        val_dir              = args.val_dir,
        test_dir             = args.test_dir,
        train_manifest       = args.train_manifest,
        val_manifest         = args.val_manifest,
        input_size           = input_size,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
        pin_memory           = device.type != "cpu",
        persistent_workers   = args.num_workers > 0,
        use_balanced_sampler = not args.no_balanced_sampler,
        seed                 = args.seed,
    )

    logger.info("[main] Building model  config=%s", model_config)
    model = model_registry.build(args.model_name, model_config)

    if hasattr(model, "count_parameters"):
        counts = model.count_parameters()
        logger.info(
            "[main] Parameters — total=%s  trainable=%s  frozen=%s",
            f"{counts['total']:,}",
            f"{counts['trainable']:,}",
            f"{counts['frozen']:,}",
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = build_loss(
        name       = args.loss,
        gamma      = args.focal_gamma,
        alpha      = args.focal_alpha,
        pos_weight = args.pos_weight,
        smoothing  = args.label_smoothing,
    )
    logger.info("[main] Loss: %s", loss_fn)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    optimizer = build_optimizer(
        model                  = model,
        name                   = args.optimizer,
        lr                     = args.lr,
        weight_decay           = args.weight_decay,
        backbone_lr_multiplier = args.backbone_lr_mult,
    )
    logger.info("[main] Optimizer: %s", optimizer.__class__.__name__)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler, scheduler_needs_metric = build_scheduler(
        optimizer    = optimizer,
        name         = args.scheduler,
        num_epochs   = args.epochs,
        warmup_steps = args.warmup_steps,
        min_lr       = args.min_lr,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        checkpoint_dir = checkpoint_dir,
        monitor        = "val_auc",
        mode           = "max",
        save_last      = True,
        verbose        = True,
    )

    early_stop_cb = EarlyStopping(
        monitor   = "val_auc",
        patience  = args.patience,
        min_delta = args.min_delta,
        mode      = "max",
        verbose   = True,
    )

    lr_monitor = LearningRateMonitor(log_every_n_epochs=1)

    # ── Training config ───────────────────────────────────────────────────────
    training_config = TrainingConfig(
        num_epochs        = args.epochs,
        grad_clip_norm    = grad_clip,
        use_amp           = not args.no_amp,
        log_every_n_steps = 50,
        phase1_epochs     = args.phase1_epochs,
        model_name        = args.model_name,
    )

    # Extra context stored in every checkpoint for full reproducibility
    extra_config = {
        "data":      vars(data_config),
        "model":     vars(model_config),
        "optimizer": {
            "name":                 args.optimizer,
            "lr":                   args.lr,
            "weight_decay":         args.weight_decay,
            "backbone_lr_mult":     args.backbone_lr_mult,
        },
        "loss":      {
            "name":                 args.loss,
            "focal_gamma":          args.focal_gamma,
            "focal_alpha":          args.focal_alpha,
            "label_smoothing":      args.label_smoothing,
        },
        "seed":      args.seed,
    }

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model                  = model,
        optimizer              = optimizer,
        loss_fn                = loss_fn,
        scheduler              = scheduler,
        scheduler_needs_metric = scheduler_needs_metric,
        checkpoint_cb          = checkpoint_cb,
        early_stop_cb          = early_stop_cb,
        lr_monitor             = lr_monitor,
        config                 = training_config,
        device                 = device,
        extra_config           = extra_config,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume is not None:
        logger.info("[main] Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("[main] Starting training run — outputs: %s", checkpoint_dir)

    try:
        history = trainer.fit(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("[main] Training interrupted by user. Last checkpoint saved.")
        sys.exit(0)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_epoch = checkpoint_cb.best_epoch
    best_auc   = checkpoint_cb.best_value

    logger.info("=" * 65)
    logger.info("Training complete")
    logger.info("  Best val_auc : %.4f  (epoch %d)", best_auc, best_epoch)
    logger.info("  Best ckpt   : %s",    checkpoint_cb.best_path)
    logger.info("  Last ckpt   : %s",    checkpoint_cb.last_path)
    logger.info("  Log file    : %s",    log_file)
    logger.info("=" * 65)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Required on Windows with multiprocessing DataLoader workers.
    # Without this guard, worker processes re-execute the training script,
    # causing duplicate model builds, dataset loads, and training runs.
    main()