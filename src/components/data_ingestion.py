import sys
import zipfile
from pathlib import Path

from src.entity.config_entity import DataIngestionConfig
from src.utils.common import create_directories
from src.utils.exception import DataIngestionError
from src.utils.logger import logger


class DataIngestion:
    """
    Downloads the kidney CT dataset from Kaggle and extracts it into the
    configured artifact directory.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        """
        Download the configured Kaggle dataset as a zip file.

        Requires Kaggle credentials to be available through either:
        - KAGGLE_USERNAME and KAGGLE_KEY environment variables
        - kaggle.json in the default Kaggle config directory
        """
        try:
            if self.config.local_data_file.exists():
                logger.info(
                    "Dataset archive already exists at %s. Skipping download.",
                    self.config.local_data_file,
                )
                return self.config.local_data_file

            create_directories([self.config.root_dir, self.config.local_data_file.parent])

            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError as exc:
                raise ImportError(
                    "The 'kaggle' package is required for data ingestion. "
                    "Install it with: pip install kaggle"
                ) from exc

            logger.info("Authenticating Kaggle API.")
            api = KaggleApi()
            api.authenticate()

            logger.info("Downloading Kaggle dataset: %s", self.config.kaggle_dataset)
            api.dataset_download_files(
                self.config.kaggle_dataset,
                path=str(self.config.local_data_file.parent),
                unzip=False,
                quiet=False,
            )

            downloaded_zip = self._find_downloaded_archive()
            if downloaded_zip != self.config.local_data_file:
                downloaded_zip.replace(self.config.local_data_file)

            self._validate_zip_file(self.config.local_data_file)
            logger.info("Dataset downloaded to %s", self.config.local_data_file)
            return self.config.local_data_file

        except Exception as e:
            raise DataIngestionError(e, sys)

    def extract_zip_file(self) -> Path:
        """
        Extract the downloaded dataset archive into the configured unzip path.
        """
        try:
            if not self.config.local_data_file.exists():
                raise FileNotFoundError(f"Dataset archive not found: {self.config.local_data_file}")

            self._validate_zip_file(self.config.local_data_file)
            create_directories([self.config.unzip_dir])

            if any(self.config.unzip_dir.iterdir()):
                logger.info(
                    "Dataset directory already contains files at %s. Skipping extraction.",
                    self.config.unzip_dir,
                )
                return self.config.unzip_dir

            logger.info(
                "Extracting %s to %s",
                self.config.local_data_file,
                self.config.unzip_dir,
            )
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)

            logger.info("Dataset extracted to %s", self.config.unzip_dir)
            return self.config.unzip_dir

        except Exception as e:
            raise DataIngestionError(e, sys)

    def run(self) -> Path:
        """
        Execute the full data ingestion step.
        """
        self.download_file()
        return self.extract_zip_file()

    def _find_downloaded_archive(self) -> Path:
        expected_zip = self.config.local_data_file
        if expected_zip.exists():
            return expected_zip

        dataset_slug = self.config.kaggle_dataset.split("/")[-1]
        candidates = sorted(
            self.config.local_data_file.parent.glob("*.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for candidate in candidates:
            if candidate.stem == dataset_slug:
                return candidate

        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            f"Kaggle download completed but no zip archive was found in "
            f"{self.config.local_data_file.parent}"
        )

    @staticmethod
    def _validate_zip_file(zip_path: Path) -> None:
        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(f"Invalid zip archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            corrupt_file = zip_ref.testzip()

        if corrupt_file is not None:
            raise zipfile.BadZipFile(f"Corrupt file found inside {zip_path}: {corrupt_file}")
