import sys
from pathlib import Path

import pandas as pd

from src.components.feature_extractor import FeatureExtractor
from src.utils.logger import logger

# Add the project root to python path to import src modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def generate_reference_features(
    train_data_dir: str = "artifacts/data_transformation/train",
    output_csv: str = "data/reference_features.csv",
):
    """
    Extract features from the training dataset to build the reference distribution
    for data drift detection.
    """
    train_dir_path = project_root / train_data_dir
    output_path = project_root / output_csv

    if not train_dir_path.exists():
        logger.error(f"Training data directory not found: {train_dir_path}")
        logger.info("Please run the data transformation pipeline first.")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    features_list = []

    # Get all image files
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        image_paths.extend(list(train_dir_path.rglob(f"*{ext}")))
        image_paths.extend(list(train_dir_path.rglob(f"*{ext.upper()}")))

    # Deduplicate
    image_paths = list(set(image_paths))

    if not image_paths:
        logger.error(f"No images found in {train_dir_path}")
        return

    logger.info(f"Found {len(image_paths)} images in {train_dir_path}")
    logger.info("Extracting features. This might take a few minutes...")

    for i, img_path in enumerate(image_paths):
        try:
            features = extractor.extract_features(img_path)
            features_list.append(features)
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1} / {len(image_paths)} images")

    if not features_list:
        logger.error("No features were extracted.")
        return

    df = pd.DataFrame(features_list)
    df.to_csv(output_path, index=False)

    logger.info(f"Successfully generated reference features at {output_path}")
    logger.info(f"Shape of reference data: {df.shape}")


if __name__ == "__main__":
    generate_reference_features()
