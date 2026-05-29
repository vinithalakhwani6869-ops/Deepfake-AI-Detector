"""
GET / and GET /health — health check endpoints.

Production enhancements:
  - Root endpoint returns basic status
  - Health endpoint reports model load status and device type
  - Lazy loading compatible (doesn't force model load)
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends

from api.dependencies import get_detector
from api.schemas.response import HealthStatus, RootStatus
from core.detector import DeepfakeDetector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/", response_model=RootStatus)
def root() -> RootStatus:
    """
    Root endpoint — basic service status.
    
    Returns:
        RootStatus with status and message.
    """
    logger.debug("[health] GET / — root endpoint")
    return RootStatus(
        status="ok",
        message="DeepFake Detector API is running.",
    )


@router.get("/health", response_model=HealthStatus)
def health(detector: DeepfakeDetector = Depends(get_detector)) -> HealthStatus:
    """
    Health check endpoint — reports model and device status.
    
    Lazy loading compatible: if detector was created with lazy_load=True,
    this endpoint returns the load status without forcing initialization.
    
    Args:
        detector: DeepfakeDetector singleton from dependency injection.
    
    Returns:
        HealthStatus with model_loaded and device info.
    """
    logger.debug(
        "[health] GET /health — model_loaded=%s device=%s",
        detector.model is not None,
        detector.device,
    )
    return HealthStatus(
        status="ok",
        model_loaded=detector.model is not None,
        device=str(detector.device),
    )
