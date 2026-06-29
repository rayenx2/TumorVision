from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Shared Pydantic schema configuration."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )


class ErrorDetail(BaseSchema):
    code: str = Field(
        ...,
        examples=["VALIDATION_ERROR"],
        description="Machine-readable error code.",
    )
    message: str = Field(
        ...,
        examples=["Invalid request payload."],
        description="Human-readable error message.",
    )
    field: str | None = Field(
        default=None,
        examples=["image"],
        description="Request field related to the error, if applicable.",
    )


class ErrorResponse(BaseSchema):
    success: bool = Field(default=False)
    error: ErrorDetail
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the error response was generated.",
    )
    path: str | None = Field(
        default=None,
        examples=["/api/v1/predict"],
        description="Request path where the error occurred.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request identifier for tracing.",
    )


class HealthResponse(BaseSchema):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy",
        examples=["healthy"],
    )
    app_name: str
    app_version: str
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        examples=["development"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the health response was generated.",
    )


class ModelInfoResponse(BaseSchema):
    model_repo: str = Field(
        ...,
        description="Hugging Face model repository identifier.",
    )
    model_filename: str = Field(
        ...,
        description="Model artifact filename.",
    )
    model_version: str = Field(
        ...,
        description="Model version currently configured for inference.",
    )
    image_size: int = Field(
        ...,
        ge=1,
        description="Expected square input image size in pixels.",
    )
    class_names: list[str] = Field(
        ...,
        description="Model output class names in index order.",
    )
    uncertainty_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold used to flag uncertain predictions.",
    )
    is_loaded: bool = Field(
        ...,
        description="Whether the model is currently loaded in memory.",
    )
