from fastapi import status


class APIException(Exception):
    """Base exception for API-layer failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "API_ERROR"
    message: str = "An unexpected API error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        field: str | None = None,
    ):
        self.message = message or self.message
        self.status_code = status_code or self.status_code
        self.error_code = error_code or self.error_code
        self.field = field
        super().__init__(self.message)

    def to_error_detail(self) -> dict[str, str | None]:
        """Return a shape compatible with api.schemas.common.ErrorDetail."""
        return {
            "code": self.error_code,
            "message": self.message,
            "field": self.field,
        }


class ValidationAPIException(APIException):
    """Raised when request validation fails beyond Pydantic validation."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "VALIDATION_ERROR"
    message = "Invalid request."


class UploadValidationException(ValidationAPIException):
    """Raised when an uploaded file fails API validation."""

    error_code = "UPLOAD_VALIDATION_ERROR"
    message = "Invalid uploaded file."


class PredictionAPIException(APIException):
    """Raised when prediction execution fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "PREDICTION_ERROR"
    message = "Prediction failed."


class ReportGenerationException(APIException):
    """Raised when PDF report generation fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "REPORT_GENERATION_ERROR"
    message = "Report generation failed."


class ResourceNotFoundException(APIException):
    """Raised when a requested API resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "RESOURCE_NOT_FOUND"
    message = "Requested resource was not found."


class TaskNotFoundException(ResourceNotFoundException):
    """Raised when a background task cannot be found."""

    error_code = "TASK_NOT_FOUND"
    message = "Task was not found."


class ExternalServiceException(APIException):
    """Raised when an external service dependency fails."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "EXTERNAL_SERVICE_ERROR"
    message = "External service request failed."
