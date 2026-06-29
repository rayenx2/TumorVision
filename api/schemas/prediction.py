from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import Field

from api.schemas.common import BaseSchema


class ClassProbability(BaseSchema):
    class_name: Literal["Normal", "Cyst", "Tumor", "Stone"] = Field(
        ...,
        description="Class label associated with this probability.",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability for the class.",
    )


class PredictionResponse(BaseSchema):
    prediction_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique prediction identifier.",
    )
    predicted_class: Literal["Normal", "Cyst", "Tumor", "Stone"] = Field(
        ...,
        description="Predicted kidney image class.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class.",
    )
    probabilities: list[ClassProbability] = Field(
        ...,
        description="Per-class prediction probabilities.",
    )
    uncertainty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model uncertainty score for the prediction.",
    )
    is_uncertain: bool = Field(
        ...,
        description="Whether the prediction should be treated as uncertain.",
    )
    gradcam_base64: str = Field(
        ...,
        description="Grad-CAM visualization encoded as a base64 data URI.",
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference duration in milliseconds.",
    )
    model_version: str = Field(
        ...,
        description="Model version used for inference.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the prediction was generated.",
    )
    task_id: str | None = Field(
        default=None,
        description="Celery task ID for async PDF report generation, if requested.",
    )


class PredictionMetadata(BaseSchema):
    prediction_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique prediction identifier.",
    )
    predicted_class: Literal["Normal", "Cyst", "Tumor", "Stone"] = Field(
        ...,
        description="Predicted kidney image class.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class.",
    )
    uncertainty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model uncertainty score for the prediction.",
    )
    is_uncertain: bool = Field(
        ...,
        description="Whether the prediction should be treated as uncertain.",
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference duration in milliseconds.",
    )
    model_version: str = Field(
        ...,
        description="Model version used for inference.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the prediction was generated.",
    )
