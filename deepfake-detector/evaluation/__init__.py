"""
Offline evaluation subsystem — never imported by the FastAPI API layer.

Import submodules directly to avoid loading torch when only metrics are needed:
    from evaluation.metrics import compute_metrics
    from evaluation.evaluator import Evaluator
"""

__all__ = [
    "BenchmarkReport",
    "Evaluator",
    "EvaluatorConfig",
    "MetricResult",
    "SplitResult",
    "compute_metrics",
    "run_inference_benchmark",
]


def __getattr__(name: str):
    if name in ("MetricResult", "compute_metrics"):
        from evaluation.metrics import MetricResult, compute_metrics
        return {"MetricResult": MetricResult, "compute_metrics": compute_metrics}[name]
    if name in ("Evaluator", "EvaluatorConfig", "SplitResult"):
        from evaluation.evaluator import Evaluator, EvaluatorConfig, SplitResult
        return {
            "Evaluator": Evaluator,
            "EvaluatorConfig": EvaluatorConfig,
            "SplitResult": SplitResult,
        }[name]
    if name in ("BenchmarkReport", "run_inference_benchmark"):
        from evaluation.benchmark import BenchmarkReport, run_inference_benchmark
        return {
            "BenchmarkReport": BenchmarkReport,
            "run_inference_benchmark": run_inference_benchmark,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
