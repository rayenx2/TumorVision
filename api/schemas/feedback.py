from typing import Literal

from pydantic import Field

from api.schemas.common import BaseSchema
from api.schemas.prediction import PredictionMetadata


class FeedbackRequest(BaseSchema):
    prediction_id: str = Field(
        ...,
        description="Prediction identifier associated with the feedback.",
    )
    correct_class: Literal["Normal", "Cyst", "Tumor", "Stone"] = Field(
        ...,
        description="Radiologist-confirmed correct class.",
    )
    comment: str | None = Field(
        default=None,
        description="Optional feedback comment.",
    )
    radiologist_name: str | None = Field(
        default=None,
        description="Optional radiologist name.",
    )


class HistoryItem(PredictionMetadata):
    feedback_received: bool = Field(
        default=False,
        description="Whether feedback has been received for this prediction.",
    )
    correct_class: Literal["Normal", "Cyst", "Tumor", "Stone"] | None = Field(
        default=None,
        description="Correct class provided through feedback, if available.",
    )


class HistoryResponse(BaseSchema):
    items: list[HistoryItem] = Field(
        ...,
        description="Prediction history items.",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of history items available.",
    )
    limit: int = Field(
        ...,
        ge=1,
        description="Maximum number of items returned.",
    )
    offset: int = Field(
        ...,
        ge=0,
        description="Number of items skipped before this page.",
    )
