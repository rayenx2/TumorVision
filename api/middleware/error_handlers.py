from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.exceptions import APIException
from api.schemas.common import ErrorDetail, ErrorResponse
from src.utils.logger import api_logger as logger


def _build_error_response(
    *,
    request: Request,
    code: str,
    message: str,
    field: str | None = None,
) -> dict:
    """Build a JSON-serializable API error response."""
    response = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
        ),
        path=request.url.path,
        request_id=request.headers.get("x-request-id"),
    )
    return response.model_dump(mode="json")


async def api_exception_handler(
    request: Request,
    exc: APIException,
) -> JSONResponse:
    """Handle custom API exceptions."""
    logger.warning(
        "API exception: code=%s status=%s path=%s message=%s",
        exc.error_code,
        exc.status_code,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(
            request=request,
            code=exc.error_code,
            message=exc.message,
            field=exc.field,
        ),
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle FastAPI and Starlette HTTP exceptions."""
    message = str(exc.detail) if exc.detail else "HTTP request failed."
    error_code = "HTTP_ERROR"

    if exc.status_code == status.HTTP_404_NOT_FOUND:
        error_code = "NOT_FOUND"
    elif exc.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        error_code = "METHOD_NOT_ALLOWED"

    logger.warning(
        "HTTP exception: code=%s status=%s path=%s message=%s",
        error_code,
        exc.status_code,
        request.url.path,
        message,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(
            request=request,
            code=error_code,
            message=message,
        ),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors."""
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    location = first_error.get("loc", [])
    field = ".".join(str(item) for item in location) if location else None
    message = first_error.get("msg", "Invalid request payload.")

    logger.warning(
        "Validation error: path=%s field=%s message=%s",
        request.url.path,
        field,
        message,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_build_error_response(
            request=request,
            code="VALIDATION_ERROR",
            message=str(message),
            field=field,
        ),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected server errors."""
    logger.exception("Unhandled exception at %s", request.url.path)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_build_error_response(
            request=request,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error.",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register API exception handlers on the FastAPI app."""
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
