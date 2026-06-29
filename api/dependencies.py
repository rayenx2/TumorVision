from functools import lru_cache

from api.config import Settings
from api.services.model_service import ModelService
from api.services.prediction_service import PredictionService
from api.services.storage_service import StorageService


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Return the cached model service instance."""
    return ModelService(settings=get_settings())


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Return the cached prediction service instance."""
    return PredictionService(settings=get_settings())


@lru_cache(maxsize=1)
def get_storage_service():
    settings = get_settings()

    if settings.supabase_url and settings.supabase_key:
        from api.services.supabase_storage_service import SupabaseStorageService

        return SupabaseStorageService(
            url=settings.supabase_url,
            key=settings.supabase_key,
        )
    return StorageService(db_path=settings.database_path)
