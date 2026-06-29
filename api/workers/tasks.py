from pathlib import Path

from api.dependencies import get_settings
from api.services.report_service import ReportService
from api.utils.image_utils import cleanup_temp_file
from api.workers.celery_app import celery_app
from src.utils.logger import api_logger as logger


@celery_app.task(name="api.workers.tasks.health_check_task")
def health_check_task() -> dict[str, str]:
    """Simple task to verify Celery-Redis connection."""
    logger.info("Health check task executed")
    return {"status": "ok", "message": "Celery worker is alive"}


@celery_app.task(
    name="api.workers.tasks.generate_report_task",
    bind=True,
)
def generate_report_task(
    self, image_path: str, prediction_id: str, run_inference: bool = True
) -> dict[str, str]:
    """Generate PDF report asynchronously and cleanup temp image."""
    logger.info("Report task started for prediction %s", prediction_id)

    try:
        settings = get_settings()
        report_service = ReportService(settings=settings)
        report_path = report_service.generate_report(
            image_path=image_path,
            prediction_id=prediction_id,
            run_inference=run_inference,
        )

        logger.info("Report task completed: %s", report_path)
        return {
            "prediction_id": prediction_id,
            "report_path": report_path,
        }

    except Exception:
        logger.exception("Report task failed for prediction %s", prediction_id)
        raise

    finally:
        cleanup_temp_file(Path(image_path))
