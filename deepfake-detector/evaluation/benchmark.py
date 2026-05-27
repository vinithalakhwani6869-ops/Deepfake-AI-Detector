"""
evaluation/benchmark.py
───────────────────────
Inference-only performance profiling — no training benchmarks.

Measures:
  • Per-image latency (ms)
  • Throughput (images / second)
  • Peak memory usage during inference
  • CPU vs GPU comparison (when CUDA is available)
"""

from __future__ import annotations

import logging
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from evaluation.evaluator import InferenceAlignedPreprocessor, resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceBenchmarkResult:
    """Benchmark statistics for one device."""

    device: str
    num_images: int
    batch_size: int
    warmup_batches: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_std_ms: float
    throughput_images_per_sec: float
    peak_memory_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkReport:
    """CPU and optional GPU benchmark results."""

    cpu: DeviceBenchmarkResult
    gpu: Optional[DeviceBenchmarkResult] = None

    def to_dict(self) -> dict:
        data = {"cpu": self.cpu.to_dict()}
        if self.gpu is not None:
            data["gpu"] = self.gpu.to_dict()
        return data


def _collect_sample_paths(
    data_dir: Path,
    *,
    max_images: int,
    manifest_csv: Optional[Path] = None,
) -> list[Path]:
    """Collect up to ``max_images`` image paths from a split directory."""
    from data.dataset import DeepfakeDataset

    catalog = DeepfakeDataset(
        root=data_dir.resolve(),
        transform=None,
        manifest_csv=manifest_csv.resolve() if manifest_csv else None,
    )
    paths = [path for path, _ in catalog.samples]
    if not paths:
        raise ValueError(f"No images found under {data_dir}")
    return paths[:max_images]


def _prepare_batch_tensors(
    paths: list[Path],
    preprocessor: InferenceAlignedPreprocessor,
    device: torch.device,
) -> torch.Tensor:
    tensors = [preprocessor(path) for path in paths]
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


@torch.inference_mode()
def _benchmark_device(
    model: nn.Module,
    paths: list[Path],
    preprocessor: InferenceAlignedPreprocessor,
    device: torch.device,
    *,
    batch_size: int,
    warmup_batches: int,
) -> DeviceBenchmarkResult:
    """Run timed inference on ``device`` and collect latency / memory stats."""
    model = model.to(device)
    model.eval()

    per_image_ms: list[float] = []
    peak_memory_mb = 0.0

    tracemalloc.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    batches = [
        paths[i : i + batch_size] for i in range(0, len(paths), batch_size)
    ]

    # Warmup — excludes CUDA kernel compilation and allocator warmup
    for batch_paths in batches[:warmup_batches]:
        batch = _prepare_batch_tensors(batch_paths, preprocessor, device)
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    total_images = 0
    total_wall_sec = 0.0

    for batch_paths in batches:
        batch = _prepare_batch_tensors(batch_paths, preprocessor, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        _ = model(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - t0
        batch_n = len(batch_paths)
        total_wall_sec += elapsed
        total_images += batch_n
        per_image_ms.extend([(elapsed / batch_n) * 1000.0] * batch_n)

    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_traced / (1024 * 1024)

    if device.type == "cuda":
        peak_memory_mb = max(
            peak_memory_mb,
            torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        )

    throughput = total_images / total_wall_sec if total_wall_sec > 0 else 0.0

    return DeviceBenchmarkResult(
        device=str(device),
        num_images=total_images,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        latency_mean_ms=float(statistics.mean(per_image_ms)),
        latency_median_ms=float(statistics.median(per_image_ms)),
        latency_p95_ms=float(np.percentile(per_image_ms, 95)),
        latency_std_ms=float(statistics.pstdev(per_image_ms)) if len(per_image_ms) > 1 else 0.0,
        throughput_images_per_sec=float(throughput),
        peak_memory_mb=float(peak_memory_mb),
    )


def run_inference_benchmark(
    model: nn.Module,
    data_dir: Path,
    *,
    manifest_csv: Optional[Path] = None,
    max_images: int = 100,
    batch_size: int = 1,
    warmup_batches: int = 2,
    input_size: int = 224,
    compare_gpu: bool = True,
) -> BenchmarkReport:
    """
    Profile inference latency, throughput, and memory on CPU and optionally GPU.

    Uses the same ``InferenceAlignedPreprocessor`` as production evaluation so
  benchmark numbers reflect real serving preprocessing cost.

    Args:
        model:         Loaded model in eval mode (weights already applied).
        data_dir:      Directory with real/ and fake/ subfolders.
        max_images:    Cap images profiled (for fast CLI runs).
        batch_size:    Batch size for timed forward passes.
        warmup_batches: Batches excluded from timing statistics.
        compare_gpu:   If True and CUDA is available, also benchmark on GPU.
    """
    paths = _collect_sample_paths(data_dir, max_images=max_images, manifest_csv=manifest_csv)
    preprocessor = InferenceAlignedPreprocessor(input_size=input_size)

    cpu_device = torch.device("cpu")
    cpu_model = _copy_model_to_device(model, cpu_device)
    cpu_result = _benchmark_device(
        cpu_model,
        paths,
        preprocessor,
        cpu_device,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
    )
    logger.info(
        "[benchmark] CPU: %.2f ms/img  %.1f img/s  peak_mem=%.1f MB",
        cpu_result.latency_mean_ms,
        cpu_result.throughput_images_per_sec,
        cpu_result.peak_memory_mb,
    )

    gpu_result: Optional[DeviceBenchmarkResult] = None
    if compare_gpu and torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_model = _copy_model_to_device(model, gpu_device)
        gpu_result = _benchmark_device(
            gpu_model,
            paths,
            preprocessor,
            gpu_device,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
        )
        logger.info(
            "[benchmark] GPU: %.2f ms/img  %.1f img/s  peak_mem=%.1f MB",
            gpu_result.latency_mean_ms,
            gpu_result.throughput_images_per_sec,
            gpu_result.peak_memory_mb,
        )

    return BenchmarkReport(cpu=cpu_result, gpu=gpu_result)


def _copy_model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Return a model copy on the target device for isolated benchmarking."""
    clone = model
    return clone.to(device)


__all__ = [
    "DeviceBenchmarkResult",
    "BenchmarkReport",
    "run_inference_benchmark",
]
