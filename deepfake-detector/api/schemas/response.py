"""Pydantic response models — frontend contract preserved."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RootStatus(BaseModel):
    status: str
    message: str


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    device: str


class DetectionResult(BaseModel):
    """Exact JSON shape expected by the existing frontend."""

    result: str = Field(..., description='"Real" or "Fake"')
    confidence: float = Field(..., ge=0.0, le=100.0)
    filename: str
