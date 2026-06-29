from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from api.config import Settings
from api.dependencies import get_prediction_service, get_settings, get_storage_service
from api.schemas.prediction import PredictionMetadata, PredictionResponse
from api.services.prediction_service import PredictionService
from api.services.storage_service import StorageService
from api.utils.image_utils import cleanup_temp_file, save_temp_file, validate_upload
from api.workers.celery_app import celery_app
from api.workers.tasks import generate_report_task
from src.utils.logger import api_logger as logger

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_image(
    file: UploadFile = File(...),
    generate_report: bool = Query(
        default=False,
        description="If true, trigger async PDF report generation.",
    ),
    settings: Settings = Depends(get_settings),
    prediction_service: PredictionService = Depends(get_prediction_service),
    storage_service: StorageService = Depends(get_storage_service),
) -> PredictionResponse:
    """Validate an uploaded image, run prediction, and persist metadata."""
    temp_path: Path | None = None

    try:
        contents = await validate_upload(file=file, settings=settings)
        temp_path = save_temp_file(contents=contents, settings=settings)
        prediction = prediction_service.predict(temp_path)

        storage_service.save_prediction(
            PredictionMetadata(
                prediction_id=prediction.prediction_id,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
                uncertainty_score=prediction.uncertainty_score,
                is_uncertain=prediction.is_uncertain,
                inference_time_ms=prediction.inference_time_ms,
                model_version=prediction.model_version,
                timestamp=prediction.timestamp,
            )
        )

        if generate_report:
            task = generate_report_task.delay(
                image_path=str(temp_path),
                prediction_id=prediction.prediction_id,
            )
            prediction.task_id = task.id
            logger.info(
                "Report generation task queued: task_id=%s, prediction_id=%s",
                task.id,
                prediction.prediction_id,
            )

        return prediction
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc
    finally:
        if temp_path is not None and not generate_report:
            cleanup_temp_file(temp_path)


@router.post(
    "/{prediction_id}/report",
    status_code=status.HTTP_200_OK,
)
async def generate_report_from_existing(
    prediction_id: str,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """Trigger async PDF report generation from an existing prediction."""
    temp_path: Path | None = None
    try:
        contents = await validate_upload(file=file, settings=settings)
        temp_path = save_temp_file(contents=contents, settings=settings)

        task = generate_report_task.delay(
            image_path=str(temp_path),
            prediction_id=prediction_id,
            run_inference=False,
        )
        logger.info(
            "Report generation task queued (no inference): task_id=%s, prediction_id=%s",
            task.id,
            prediction_id,
        )

        return {"task_id": task.id}
    except HTTPException:
        if temp_path:
            cleanup_temp_file(temp_path)
        raise
    except Exception as exc:
        if temp_path:
            cleanup_temp_file(temp_path)
        logger.exception("Failed to queue report generation for %s", prediction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed",
        ) from exc


@router.get(
    "/report/{task_id}",
    status_code=status.HTTP_200_OK,
)
async def get_report(task_id: str):
    """Check report task status. Return PDF file if ready, else JSON status."""
    task_result = AsyncResult(task_id, app=celery_app)
    state = task_result.state

    if state == "PENDING":
        return {"task_id": task_id, "status": "pending"}

    if state == "STARTED":
        return {"task_id": task_id, "status": "in_progress"}

    if state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(task_result.info)}

    if state == "SUCCESS":
        result = task_result.result
        report_path = Path(result["report_path"])

        if not report_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report file not found on disk",
            )

        return FileResponse(
            path=report_path,
            media_type="application/pdf",
            filename=report_path.name,
        )

    return {"task_id": task_id, "status": state.lower()}
