import sys
from pathlib import Path
from typing import Any

from src.entity.config_entity import DataValidationConfig
from src.utils.exception import DataValidationError
from src.utils.logger import logger


class DataValidation:
    """
    Validates the extracted CT kidney dataset before transformation/training.

    The current DataValidationConfig does not carry an explicit data directory,
    so this component can either receive one at runtime or discover the folder
    that contains all configured class directories.
    """

    DEFAULT_DATA_ROOT = Path("artifacts/data_ingestion")

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.valid_classes = [str(class_name) for class_name in config.valid_classes]
        self.allowed_extensions = {
            (extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}")
            for extension in config.allowed_extensions
        }

    def validate_all_files_exist(self, data_dir: Path | str | None = None) -> bool:
        """
        Backward-compatible entry point used by many training templates.
        Returns True when the dataset passes every configured validation check.
        """
        report = self.validate_dataset(data_dir=data_dir)
        return bool(report["validation_status"])

    def validate_dataset(self, data_dir: Path | str | None = None) -> dict[str, Any]:
        """
        Validate dataset shape, class folders, image counts, and file extensions.

        Args:
            data_dir: Optional directory containing the class folders. If omitted,
                the component searches common ingestion artifact locations.

        Returns:
            A report dictionary. The same information is also written to
            config.status_file for pipeline/DVC visibility.
        """
        try:
            dataset_dir = self._resolve_dataset_dir(data_dir)
            logger.info("Validating dataset at %s", dataset_dir)

            class_dirs = {
                path.name: path
                for path in dataset_dir.iterdir()
                if path.is_dir() and path.name in self.valid_classes
            }
            present_classes = set(class_dirs)
            expected_classes = set(self.valid_classes)

            missing_classes = sorted(expected_classes - present_classes)
            unexpected_classes = sorted(
                path.name
                for path in dataset_dir.iterdir()
                if path.is_dir() and path.name not in expected_classes
            )

            class_counts: dict[str, int] = {}
            invalid_files: list[str] = []
            classes_below_minimum: dict[str, int] = {}

            for class_name in self.valid_classes:
                class_dir = class_dirs.get(class_name)
                if class_dir is None:
                    class_counts[class_name] = 0
                    continue

                image_count = 0
                for file_path in class_dir.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if file_path.suffix.lower() in self.allowed_extensions:
                        image_count += 1
                    else:
                        invalid_files.append(str(file_path))

                class_counts[class_name] = image_count
                if image_count < self.config.min_images_per_class:
                    classes_below_minimum[class_name] = image_count

            validation_status = not (
                missing_classes or unexpected_classes or invalid_files or classes_below_minimum
            )

            report: dict[str, Any] = {
                "validation_status": validation_status,
                "dataset_dir": str(dataset_dir),
                "valid_classes": self.valid_classes,
                "allowed_extensions": sorted(self.allowed_extensions),
                "min_images_per_class": self.config.min_images_per_class,
                "class_counts": class_counts,
                "missing_classes": missing_classes,
                "unexpected_classes": unexpected_classes,
                "classes_below_minimum": classes_below_minimum,
                "invalid_files": invalid_files,
            }

            self._write_status(report)
            logger.info("Data validation status: %s", validation_status)
            return report

        except Exception as e:
            raise DataValidationError(e, sys)

    def run(self, data_dir: Path | str | None = None) -> bool:
        """
        Execute the data validation stage and return the validation status.
        """
        return self.validate_all_files_exist(data_dir=data_dir)

    def _resolve_dataset_dir(self, data_dir: Path | str | None = None) -> Path:
        if data_dir is not None:
            candidate = Path(data_dir)
        else:
            candidate = Path(getattr(self.config, "data_dir", self.DEFAULT_DATA_ROOT))

        if not candidate.exists():
            raise FileNotFoundError(f"Dataset directory not found: {candidate}")

        if self._contains_class_dirs(candidate):
            return candidate

        for path in candidate.rglob("*"):
            if path.is_dir() and self._contains_class_dirs(path):
                return path

        raise FileNotFoundError(
            "Could not find a dataset directory containing the expected class folders: "
            f"{', '.join(self.valid_classes)} under {candidate}"
        )

    def _contains_class_dirs(self, path: Path) -> bool:
        children = {child.name for child in path.iterdir() if child.is_dir()}
        return set(self.valid_classes).issubset(children)

    def _write_status(self, report: dict[str, Any]) -> None:
        self.config.status_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"Validation status: {report['validation_status']}",
            f"Dataset directory: {report['dataset_dir']}",
            f"Expected classes: {', '.join(report['valid_classes'])}",
            f"Allowed extensions: {', '.join(report['allowed_extensions'])}",
            f"Minimum images per class: {report['min_images_per_class']}",
            "Class counts:",
        ]

        lines.extend(
            f"  - {class_name}: {count}" for class_name, count in report["class_counts"].items()
        )

        if report["missing_classes"]:
            lines.append(f"Missing classes: {', '.join(report['missing_classes'])}")
        if report["unexpected_classes"]:
            lines.append(f"Unexpected classes: {', '.join(report['unexpected_classes'])}")
        if report["classes_below_minimum"]:
            below_minimum = ", ".join(
                f"{class_name} ({count})"
                for class_name, count in report["classes_below_minimum"].items()
            )
            lines.append(f"Classes below minimum: {below_minimum}")
        if report["invalid_files"]:
            lines.append("Invalid files:")
            lines.extend(f"  - {file_path}" for file_path in report["invalid_files"])

        self.config.status_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
