import base64
import csv
import time
from pathlib import Path

from api.config import Settings
from api.schemas.prediction import ClassProbability, PredictionResponse
from src.components.feature_extractor import FeatureExtractor
from src.config.configuration import ConfigurationManager
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.logger import api_logger as logger


class PredictionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config_manager = ConfigurationManager()
        self.pipeline = PredictionPipeline(self.config_manager)
        logger.info("PredictionService initialized")
        self.feature_extractor = FeatureExtractor()
        self.current_features_path = Path("data/current_features.csv")
        self.current_features_path.parent.mkdir(parents=True, exist_ok=True)

    def predict(self, image_path: Path) -> PredictionResponse:
        start_time = time.perf_counter()
        result = self.pipeline.predict(image_path, generate_report=False)
        try:
            features = self.feature_extractor.extract_features(image_path)
            self._append_features(features)
        except Exception as e:
            logger.warning("Feature logging failed: %s", e)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        gradcam_base64 = self._encode_gradcam(Path(result.overlay_path))
        probabilities = self._build_probabilities(result.probabilities)

        return PredictionResponse(
            prediction_id=result.case_id,
            predicted_class=result.predicted_class,
            confidence=result.mc_dropout_confidence,
            probabilities=probabilities,
            uncertainty_score=result.uncertainty_score,
            is_uncertain=result.is_uncertain,
            gradcam_base64=gradcam_base64,
            inference_time_ms=inference_time_ms,
            model_version=self.settings.app_version,
        )

    def _encode_gradcam(self, overlay_path: Path) -> str:
        contents = overlay_path.read_bytes()
        encoded = base64.b64encode(contents).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _build_probabilities(self, probs_dict: dict) -> list[ClassProbability]:
        probabilities = [
            ClassProbability(class_name=class_name, probability=float(probability))
            for class_name, probability in probs_dict.items()
        ]
        return sorted(probabilities, key=lambda item: item.probability, reverse=True)

    def _append_features(self, features: dict) -> None:
        file_exists = self.current_features_path.exists()
        with open(self.current_features_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)
