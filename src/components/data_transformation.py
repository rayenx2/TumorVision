import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from src.entity.config_entity import DataTransformationConfig
from src.utils.common import create_directories
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger


class DataTransformation:
    """
    Prepare the validated image dataset for model training.

    This component discovers the class-directory dataset produced by ingestion,
    creates deterministic train/validation/test splits, and can optionally build
    TensorFlow image datasets from those split folders.
    """

    DEFAULT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    RANDOM_SEED = 42

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.data_dir = Path(config.data_dir)
        self.image_size = tuple(int(value) for value in config.image_size[:2])
        self.split_dirs = {
            "train": self.root_dir / "train",
            "val": self.root_dir / "val",
            "test": self.root_dir / "test",
        }

        self._validate_split_config()

    def run(self) -> dict[str, Any]:
        """
        Execute the full transformation step.
        """
        return self.create_train_val_test_split()

    def create_train_val_test_split(self) -> dict[str, Any]:
        """
        Copy source images into train, validation, and test directories.

        Returns:
            A summary containing the discovered source directory, class names,
            output split directories, and per-class counts.
        """
        try:
            dataset_dir = self._resolve_dataset_dir(self.data_dir)
            class_dirs = self._get_class_dirs(dataset_dir)

            create_directories([self.root_dir, *self.split_dirs.values()])
            for split_dir in self.split_dirs.values():
                for class_name in class_dirs:
                    (split_dir / class_name).mkdir(parents=True, exist_ok=True)

            summary: dict[str, Any] = {
                "source_dir": str(dataset_dir),
                "root_dir": str(self.root_dir),
                "image_size": list(self.image_size),
                "batch_size": self.config.batch_size,
                "splits": {
                    "train": self.config.train_split,
                    "val": self.config.val_split,
                    "test": self.config.test_split,
                },
                "split_dirs": {
                    split_name: str(split_dir) for split_name, split_dir in self.split_dirs.items()
                },
                "class_counts": {},
                "total_images": 0,
            }

            for class_name, class_dir in class_dirs.items():
                image_paths = self._list_image_files(class_dir)
                train_files, val_files, test_files = self._split_files(image_paths)

                split_files = {
                    "train": train_files,
                    "val": val_files,
                    "test": test_files,
                }

                summary["class_counts"][class_name] = {
                    split_name: len(files) for split_name, files in split_files.items()
                }
                summary["total_images"] += len(image_paths)

                for split_name, files in split_files.items():
                    output_class_dir = self.split_dirs[split_name] / class_name
                    for file_path in files:
                        shutil.copy2(file_path, output_class_dir / file_path.name)

            self._write_summary(summary)
            logger.info(
                "Data transformation completed. Prepared %s images in %s",
                summary["total_images"],
                self.root_dir,
            )
            return summary

        except Exception as e:
            raise KidneyTumorException(e, sys)

    def get_train_val_test_datasets(self):
        """
        Build TensorFlow datasets from the split directories.

        Returns:
            Tuple of (train_ds, val_ds, test_ds) preprocessed for EfficientNet.
        """
        try:
            if not all(path.exists() for path in self.split_dirs.values()):
                self.create_train_val_test_split()

            return (
                self._load_tf_dataset(self.split_dirs["train"], shuffle=True),
                self._load_tf_dataset(self.split_dirs["val"], shuffle=False),
                self._load_tf_dataset(self.split_dirs["test"], shuffle=False),
            )
        except Exception as e:
            raise KidneyTumorException(e, sys)

    def _load_tf_dataset(self, directory: Path, shuffle: bool):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to create image datasets. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            seed=self.RANDOM_SEED,
        )

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        return dataset.map(
            lambda images, labels: (
                preprocess_input(tf.cast(images, tf.float32)),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    def _resolve_dataset_dir(self, data_dir: Path) -> Path:
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        if self._has_image_class_dirs(data_dir):
            return data_dir

        candidates = [
            path
            for path in data_dir.rglob("*")
            if path.is_dir() and self._has_image_class_dirs(path)
        ]

        if not candidates:
            raise FileNotFoundError(
                "Could not find a dataset directory containing class folders "
                f"with images under {data_dir}"
            )

        return max(candidates, key=self._count_images_in_child_dirs)

    def _has_image_class_dirs(self, path: Path) -> bool:
        image_class_dirs = [
            child for child in path.iterdir() if child.is_dir() and self._list_image_files(child)
        ]
        return len(image_class_dirs) >= 2

    def _get_class_dirs(self, dataset_dir: Path) -> dict[str, Path]:
        class_dirs = {
            path.name: path
            for path in sorted(dataset_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_dir() and self._list_image_files(path)
        }

        if not class_dirs:
            raise FileNotFoundError(f"No class directories with images found in {dataset_dir}")

        return class_dirs

    def _list_image_files(self, directory: Path) -> list[Path]:
        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in self.DEFAULT_ALLOWED_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )

    def _split_files(self, image_paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        if not image_paths:
            return [], [], []

        shuffled_paths = image_paths.copy()
        random.Random(self.RANDOM_SEED).shuffle(shuffled_paths)

        train_count, val_count, test_count = self._calculate_split_counts(len(shuffled_paths))
        train_end = train_count
        val_end = train_count + val_count

        return (
            shuffled_paths[:train_end],
            shuffled_paths[train_end:val_end],
            shuffled_paths[val_end : val_end + test_count],
        )

    def _calculate_split_counts(self, total: int) -> tuple[int, int, int]:
        split_values = [
            self.config.train_split,
            self.config.val_split,
            self.config.test_split,
        ]
        raw_counts = [total * split for split in split_values]
        counts = [int(count) for count in raw_counts]
        remaining = total - sum(counts)

        remainders = sorted(
            enumerate(raw_counts),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for index, _ in remainders[:remaining]:
            counts[index] += 1

        return counts[0], counts[1], counts[2]

    def _count_images_in_child_dirs(self, path: Path) -> int:
        return sum(len(self._list_image_files(child)) for child in path.iterdir() if child.is_dir())

    def _validate_split_config(self) -> None:
        splits = [
            self.config.train_split,
            self.config.val_split,
            self.config.test_split,
        ]

        if any(split < 0 for split in splits):
            raise ValueError("Train, validation, and test splits must be non-negative.")

        if not abs(sum(splits) - 1.0) <= 1e-6:
            raise ValueError(
                "Train, validation, and test splits must add up to 1.0. " f"Received: {sum(splits)}"
            )

    def _write_summary(self, summary: dict[str, Any]) -> None:
        summary_path = self.root_dir / "split_summary.json"
        summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
