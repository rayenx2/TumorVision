from fastapi import APIRouter, Depends, status

from api.config import Settings
from api.dependencies import get_model_service, get_settings
from api.schemas.common import HealthResponse, ModelInfoResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return the API health status and application metadata."""
    environment = "development" if settings.debug else "production"

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=environment,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def model_info(
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Return model configuration and load status."""
    return ModelInfoResponse(
        model_repo=settings.hf_model_repo,
        model_filename=settings.hf_model_filename,
        model_version=settings.app_version,
        image_size=settings.model_image_size,
        class_names=settings.class_names,
        uncertainty_threshold=settings.uncertainty_threshold,
        is_loaded=model_service.is_loaded,
    )
