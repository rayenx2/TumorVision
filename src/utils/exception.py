import sys

from src.utils.logger import logger


def _get_error_message(error: Exception, error_detail: sys) -> str:
    """
    Extracts filename, line number, and message from an exception.
    This gives you exactly where in the code the error happened.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f"Error in [{file_name}] " f"at line [{line_number}]: " f"{str(error)}"


class KidneyTumorException(Exception):
    """
    Custom exception for the Kidney Tumor Identification System.

    Usage:
        try:
            risky_operation()
        except Exception as e:
            raise KidneyTumorException(e, sys)
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        self.error_message = _get_error_message(error_message, error_detail)
        super().__init__(self.error_message)
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message


class DataIngestionError(KidneyTumorException):
    """Raised when data download or extraction fails."""

    pass


class DataValidationError(KidneyTumorException):
    """Raised when data validation checks fail."""

    pass


class ModelTrainingError(KidneyTumorException):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(KidneyTumorException):
    """Raised when model evaluation fails or does not meet threshold."""

    pass


class PredictionError(KidneyTumorException):
    """Raised when inference or Grad-CAM generation fails."""

    pass
