import json
import os
import sys
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.utils.exception import KidneyTumorException
from src.utils.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox (dot-access dict).

    Usage:
        config = read_yaml(Path("config/config.yaml"))
        print(config.data_ingestion.root_dir)

    Args:
        path_to_yaml: Path to the YAML file

    Returns:
        ConfigBox with YAML contents

    Raises:
        KidneyTumorException: If file is empty or not found
    """
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)

        if content is None:
            raise BoxValueError(f"YAML file is empty: {path_to_yaml}")

        logger.info(f"YAML file loaded: {path_to_yaml}")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")
    except Exception as e:
        raise KidneyTumorException(e, sys)


@ensure_annotations
def create_directories(paths: list, verbose: bool = True):
    """
    Creates a list of directories if they don't exist.

    Usage:
        create_directories([Path("artifacts/data"), Path("reports")])

    Args:
        paths: List of directory paths to create
        verbose: Whether to log each creation
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dict as a JSON file with indentation.

    Usage:
        save_json(Path("reports/metrics.json"), {"auc": 0.95})
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        raise KidneyTumorException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns a ConfigBox (dot-access dict).

    Usage:
        metrics = load_json(Path("reports/metrics.json"))
        print(metrics.auc)
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON loaded: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise KidneyTumorException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in KB.

    Usage:
        print(get_size(Path("artifacts/model.h5")))  # "~ 528 KB"
    """
    size_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_kb} KB"
