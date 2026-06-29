from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "on", "dev", "development", "debug"}
FALSY_VALUES = {"0", "false", "f", "no", "n", "off", "prod", "production", "release"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="TumorVision")
    app_version: str = Field(default="0.1.0")
    api_prefix: str = Field(default="/api/v1")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    debug: bool = Field(default=False)
    hf_model_repo: str = Field(default="Himel000/kidney-tumor-efficientnetb4")
    hf_model_filename: str = Field(default="model.keras")
    model_image_size: int = Field(default=380)
    class_names: list[str] = Field(
        default=["Cyst", "Normal", "Stone", "Tumor"],
        description="Model output class order: Cyst=0, Normal=1, Stone=2, Tumor=3.",
    )
    uncertainty_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    database_path: str = Field(default="data/predictions.db")
    max_upload_size_mb: int = Field(default=10, ge=1)
    allowed_image_types: list[str] = Field(default=["image/jpeg", "image/png", "image/jpg"])
    temp_dir: str = Field(default="temp/uploads")
    redis_url: str = Field(default="redis://localhost:6379/0")
    supabase_url: str = Field(default="")
    supabase_key: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        if isinstance(value, str):
            return value.upper()

        return value

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_mode(cls, value: bool | str) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized_value = value.strip().lower()

            if normalized_value in TRUTHY_VALUES:
                return True

            if normalized_value in FALSY_VALUES:
                return False

            raise ValueError(
                "debug must be a boolean value or one of: "
                f"{sorted(TRUTHY_VALUES | FALSY_VALUES)}"
            )

        raise ValueError("debug must be a boolean value")

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

        if len(value) > 1 and value.endswith("/"):
            return value.rstrip("/")

        return value
