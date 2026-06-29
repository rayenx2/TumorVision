from celery import Celery

from api.dependencies import get_settings

settings = get_settings()

celery_app = Celery(
    "kidney_tumor_identification_system",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["api.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    result_expires=60 * 60 * 24,
    worker_prefetch_multiplier=1,
)

__all__ = ["celery_app"]
