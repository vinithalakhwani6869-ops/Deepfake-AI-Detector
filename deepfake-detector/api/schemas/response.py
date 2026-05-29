"""
Pydantic response models — frontend contract preserved.

Production enhancements:
  - Explicit field constraints (confidence 0–100)
  - Clear docstrings
  - Type hints for IDE support
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RootStatus(BaseModel):
    """
    Root endpoint response.
    
    Attributes:
        status: Service status ("ok").
        message: Human-readable status message.
    """

    status: str
    message: str


class HealthStatus(BaseModel):
    """
    Health check endpoint response.
    
    Attributes:
        status: Service status ("ok").
        model_loaded: Whether the model has been loaded.
        device: Compute device ("cuda:0", "cpu", etc.).
    """

    status: str
    model_loaded: bool
    device: str


class DetectionResult(BaseModel):
    """
    Detection endpoint response — exact JSON shape expected by frontend.
    
    Attributes:
        result: Classification result ("Real" or "Fake").
        confidence: Confidence score (0–100%).
        filename: Original filename of uploaded image.
    """

    result: str = Field(..., description='"Real" or "Fake"')
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence score as percentage (0–100)",
    )
    filename: str
