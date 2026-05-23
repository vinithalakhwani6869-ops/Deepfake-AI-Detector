"""GET / and GET /health — unchanged from original API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_detector
from api.schemas.response import HealthStatus, RootStatus
from core.detector import DeepfakeDetector

router = APIRouter(tags=["health"])


@router.get("/", response_model=RootStatus)
def root() -> RootStatus:
    return RootStatus(
        status="ok",
        message="DeepFake Detector API is running.",
    )


@router.get("/health", response_model=HealthStatus)
def health(detector: DeepfakeDetector = Depends(get_detector)) -> HealthStatus:
    return HealthStatus(
        status="ok",
        model_loaded=detector.model is not None,
        device=str(detector.device),
    )
