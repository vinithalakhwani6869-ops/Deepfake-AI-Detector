"""Core business logic — no FastAPI imports."""

__all__ = ["DeepfakeDetector"]


def __getattr__(name: str):
    """Lazy import so preprocessor/postprocessor work without torch loaded."""
    if name == "DeepfakeDetector":
        from core.detector import DeepfakeDetector
        return DeepfakeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
