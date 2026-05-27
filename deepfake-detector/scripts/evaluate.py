#!/usr/bin/env python3
"""
scripts/evaluate.py
───────────────────
CLI entry point for offline deepfake model evaluation.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/deepfake_model.pth \\
        --val-dir data/val \\
        --output-dir logs/eval_run_001

    python scripts/evaluate.py \\
        --checkpoint checkpoints/deepfake_model.pth \\
        --val-dir data/val \\
        --test-dir data/test \\
        --tune-threshold \\
        --benchmark \\
        --output-dir logs/eval_run_001

Loads optional YAML config (--config) for paths and hyperparameters.
Saves metrics.json and evaluation plots under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.benchmark import run_inference_benchmark
from evaluation.evaluator import (
    Evaluator,
    EvaluatorConfig,
    default_checkpoint_path,
    set_deterministic_mode,
)
from evaluation.visualiser import save_evaluation_plots

logger = logging.getLogger("evaluate")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file if PyYAML is installed."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to load config {path}. "
            "Install with: pip install pyyaml"
        ) from exc

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI args with optional YAML config (CLI takes precedence)."""
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml_config(Path(args.config))

    def pick(key: str, arg_value: Any, default: Any = None) -> Any:
        if arg_value is not None:
            return arg_value
        return cfg.get(key, default)

    return {
        "checkpoint": pick("checkpoint", args.checkpoint),
        "model_name": pick("model_name", args.model_name, "efficientnet_b0"),
        "input_size": int(pick("input_size", args.input_size, 224)),
        "batch_size": int(pick("batch_size", args.batch_size, 32)),
        "num_workers": int(pick("num_workers", args.num_workers, 0)),
        "device": pick("device", args.device, "auto"),
        "threshold": float(pick("threshold", args.threshold, 0.5)),
        "seed": int(pick("seed", args.seed, 42)),
        "train_dir": pick("train_dir", args.train_dir),
        "val_dir": pick("val_dir", args.val_dir),
        "test_dir": pick("test_dir", args.test_dir),
        "train_manifest": pick("train_manifest", args.train_manifest),
        "val_manifest": pick("val_manifest", args.val_manifest),
        "test_manifest": pick("test_manifest", args.test_manifest),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained deepfake detector checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, help="Optional YAML config file")
    p.add_argument("--checkpoint", type=Path, help="Path to .pth checkpoint")
    p.add_argument("--model-name", default="efficientnet_b0", help="Architecture registry key")
    p.add_argument("--input-size", type=int, default=224, help="Model input resolution")
    p.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--threshold", type=float, default=0.5, help="P(fake) decision threshold")
    p.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune threshold on val split (requires --val-dir)",
    )
    p.add_argument(
        "--threshold-metric",
        choices=["f1", "youden"],
        default="f1",
        help="Metric to maximise when tuning threshold",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--train-dir", type=Path, help="Train split root (real/ + fake/)")
    p.add_argument("--val-dir", type=Path, help="Validation split root")
    p.add_argument("--test-dir", type=Path, help="Test split root")
    p.add_argument("--train-manifest", type=Path, help="CSV manifest for train split")
    p.add_argument("--val-manifest", type=Path, help="CSV manifest for val split")
    p.add_argument("--test-manifest", type=Path, help="CSV manifest for test split")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics.json and plots (default: logs/eval_<timestamp>)",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference latency / throughput benchmark on val-dir",
    )
    p.add_argument(
        "--benchmark-images",
        type=int,
        default=100,
        help="Max images for benchmark profiling",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving ROC / confusion matrix / distribution plots",
    )
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p


def _print_summary(results: dict) -> None:
    """Print a structured human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTOR — EVALUATION SUMMARY")
    print("=" * 60)

    for split_name, split_data in results.get("splits", {}).items():
        m = split_data["metrics"]
        print(f"\n  [{split_name.upper()}]  n={m['num_samples']}  threshold={m['threshold']:.4f}")
        print(f"    accuracy   : {m['accuracy']:.4f}")
        print(f"    precision  : {m['precision']:.4f}")
        print(f"    recall     : {m['recall']:.4f}")
        print(f"    f1         : {m['f1']:.4f}")
        print(f"    roc_auc    : {m['roc_auc']:.4f}")
        print(f"    confusion  : {m['confusion_matrix']}")

    if "benchmark" in results:
        print("\n  [BENCHMARK]")
        for device_key, bench in results["benchmark"].items():
            print(
                f"    {device_key}: "
                f"{bench['latency_mean_ms']:.2f} ms/img  "
                f"{bench['throughput_images_per_sec']:.1f} img/s  "
                f"peak_mem={bench['peak_memory_mb']:.1f} MB"
            )

    print("\n" + "=" * 60 + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = _merge_config(args)

    if not any([cfg.get("train_dir"), cfg.get("val_dir"), cfg.get("test_dir")]):
        parser.error("At least one of --train-dir, --val-dir, or --test-dir is required.")

    if args.tune_threshold and cfg.get("val_dir") is None:
        parser.error("--tune-threshold requires --val-dir.")

    set_deterministic_mode(cfg["seed"])

    checkpoint = default_checkpoint_path(
        Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "logs" / f"eval_{stamp}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_config = EvaluatorConfig(
        checkpoint_path=checkpoint,
        model_name=cfg["model_name"],
        input_size=cfg["input_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        device=cfg["device"],
        threshold=cfg["threshold"],
        tune_threshold=args.tune_threshold,
        threshold_metric=args.threshold_metric,
        seed=cfg["seed"],
    )

    evaluator = Evaluator(eval_config)
    evaluator.load_model()

    splits: dict[str, Path] = {}
    manifests: dict[str, Path] = {}

    for name, dir_key, manifest_key in (
        ("train", "train_dir", "train_manifest"),
        ("val", "val_dir", "val_manifest"),
        ("test", "test_dir", "test_manifest"),
    ):
        data_dir = cfg.get(dir_key)
        if data_dir is not None:
            splits[name] = Path(data_dir).resolve()
            manifest = cfg.get(manifest_key)
            if manifest is not None:
                manifests[name] = Path(manifest).resolve()

    split_results = evaluator.run(splits, manifests=manifests or None)

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "model_name": cfg["model_name"],
        "input_size": cfg["input_size"],
        "seed": cfg["seed"],
        "device": str(evaluator.device),
        "splits": {},
        "plots": {},
    }

    plots_dir = output_dir / "plots"
    for split_name, result in split_results.items():
        report["splits"][split_name] = {
            "data_dir": str(splits[split_name]),
            "metrics": result.metrics.to_dict(),
        }

        if not args.no_plots:
            try:
                saved = save_evaluation_plots(
                    result.y_true,
                    result.y_score,
                    result.y_pred,
                    plots_dir,
                    split=split_name,
                )
                report["plots"][split_name] = {k: str(v) for k, v in saved.items()}
            except ImportError as exc:
                logger.warning("Skipping plots: %s", exc)

    if args.benchmark:
        bench_dir = cfg.get("val_dir") or cfg.get("test_dir") or cfg.get("train_dir")
        if bench_dir is None:
            parser.error("--benchmark requires at least one data directory.")
        bench_manifest = manifests.get("val") or manifests.get("test")
        bench_report = run_inference_benchmark(
            evaluator.model,
            Path(bench_dir).resolve(),
            manifest_csv=bench_manifest,
            max_images=args.benchmark_images,
            batch_size=1,
            input_size=cfg["input_size"],
        )
        report["benchmark"] = bench_report.to_dict()

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Metrics saved → %s", metrics_path)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
