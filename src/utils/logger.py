# 1. Imports
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 2. Constants
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMPED_LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
API_LOG_FILE = LOG_DIR / "api.log"

MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# 3. Helper
def get_logger(
    name: str,
    log_file: Path | None = None,
    use_rotating: bool = False,
) -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    target_file = log_file if log_file is not None else TIMESTAMPED_LOG_FILE

    if use_rotating:
        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=MAX_LOG_FILE_SIZE,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(target_file, encoding="utf-8")

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 4. Module-level exports
logger = get_logger("kidney_tumor")
api_logger = get_logger(
    "kidney_tumor.api",
    log_file=API_LOG_FILE,
    use_rotating=True,
)
