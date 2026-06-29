import io
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from PIL import Image

from api.config import Settings
from src.utils.logger import api_logger as logger


async def validate_upload(file: UploadFile, settings: Settings) -> bytes:
    """Validate uploaded image: content-type, size, format. Return raw bytes."""
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type",
        )

    contents = await file.read()
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    if len(contents) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large",
        )

    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted image",
        ) from exc

    return contents


def save_temp_file(contents: bytes, settings: Settings) -> Path:
    """Save uploaded bytes to temp file, return path."""
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / f"{uuid4().hex}.jpg"
    temp_path.write_bytes(contents)
    logger.info("Saved temp upload: %s", temp_path)

    return temp_path


def cleanup_temp_file(path: Path) -> None:
    """Delete temp file safely."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to delete temp upload: %s", path, exc_info=True)
