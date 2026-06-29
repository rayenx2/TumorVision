from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_storage_service
from api.schemas.feedback import FeedbackRequest, HistoryResponse
from api.services.storage_service import StorageService
from src.utils.logger import api_logger as logger

router = APIRouter(prefix="/records", tags=["Records"])


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: FeedbackRequest,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, str]:
    """Submit radiologist feedback for a prediction."""
    saved = storage_service.save_feedback(feedback)

    if not saved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with prediction_id '{feedback.prediction_id}' not found.",
        )

    logger.info("Feedback saved for prediction_id=%s", feedback.prediction_id)
    return {"message": "Feedback saved successfully"}


@router.get(
    "/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    storage_service: StorageService = Depends(get_storage_service),
) -> HistoryResponse:
    """Get paginated prediction history."""
    items, total = storage_service.list_predictions(limit=limit, offset=offset)

    return HistoryResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )
