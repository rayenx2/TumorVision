from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np


class FeatureExtractor:
    """
    A component for extracting numerical features from medical images.
    """

    def __init__(self):
        pass

    def extract_features(self, image_source: Union[str, Path, np.ndarray]) -> Dict[str, float]:
        """
        Extracts 5 statistical features from an image.

        Args:
            image_source: Path to the image or a numpy array representing the image.

        Returns:
            A dictionary containing 5 features:
            - mean_intensity
            - std_intensity
            - entropy
            - sharpness
            - contrast
        """
        # Load or process the image into grayscale numpy array
        if isinstance(image_source, (str, Path)):
            image = cv2.imread(str(image_source), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image from {image_source}")
        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 3:
                image = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
            elif len(image_source.shape) == 2:
                image = image_source
            else:
                raise ValueError("numpy array image must be 2D or 3D")
        else:
            raise TypeError("image_source must be a file path or a numpy array")

        # 1. Mean Intensity
        mean_intensity = float(np.mean(image))

        # 2. Standard Deviation of Intensity
        std_intensity = float(np.std(image))

        # 3. Entropy (measure of randomness/information content)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-7)  # Add epsilon to prevent division by zero
        non_zero_hist = hist[hist > 0]
        entropy = float(-np.sum(non_zero_hist * np.log2(non_zero_hist)))

        # 4. Sharpness (Variance of the Laplacian)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        # 5. Contrast (RMS Contrast)
        # Normalized standard deviation (std / mean)
        contrast = float(std_intensity / mean_intensity) if mean_intensity > 0 else 0.0

        return {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "entropy": entropy,
            "sharpness": sharpness,
            "contrast": contrast,
        }
