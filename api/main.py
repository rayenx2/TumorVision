# 1. Imports
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_model_service, get_settings
from api.middleware.error_handlers import register_exception_handlers
from api.routers.health import router as health_router
from api.routers.prediction import router as prediction_router
from api.routers.records import router as records_router
from src.utils.logger import api_logger as logger

# 2. Settings
settings = get_settings()


# 3. Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    model_service = get_model_service()
    try:
        model_service.load()
    except Exception:
        logger.exception("Model load failed during startup")

    yield
    logger.info("Shutting down %s", settings.app_name)


# 4. App
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    description="""
## Kidney Tumor Identification API

Backend service for kidney tumor image analysis workflows.

### Features

- Health monitoring
- Versioned API routes
- Type-safe configuration
""",
)

register_exception_handlers(app)


# 5. CORS
# TODO: Replace permissive production default with explicit frontend origins.
allow_origins = ["*"] if settings.debug else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 6. Routers + Root
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(prediction_router, prefix=settings.api_prefix)
app.include_router(records_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Return a minimal root response for uptime checks."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "status": "healthy",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": f"{settings.api_prefix}/health",
    }
