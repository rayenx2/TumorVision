import json
from pathlib import Path

import numpy as np

from api.config import Settings
from src.components.gradcam import GradCAMResult
from src.components.report_generator import ReportInput
from src.components.uncertainty import UncertaintyResult
from src.config.configuration import ConfigurationManager
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.logger import api_logger as logger


class ReportService:
    """Generate PDF reports for completed prediction requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.config_manager = ConfigurationManager()
        self.pipeline = PredictionPipeline(self.config_manager)
        logger.info("ReportService initialized")

    def generate_report(
        self, image_path: str, prediction_id: str, run_inference: bool = True
    ) -> str:
        """Generate a PDF report and return its file path."""
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        if run_inference:
            result = self.pipeline.predict(
                image_file,
                case_id=prediction_id,
                generate_report=True,
            )
            report_path = str(result.report_path)
        else:
            # Bypass inference by loading the saved prediction result
            json_path = Path("artifacts/predictions") / f"{prediction_id}.json"
            if not json_path.exists():
                raise FileNotFoundError(f"Prediction result JSON not found: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            gradcam_result = GradCAMResult(
                predicted_class=data["predicted_class"],
                predicted_index=data["predicted_index"],
                confidence=data["single_shot_confidence"],
                probabilities=data["probabilities"],
                heatmap=np.array([]),
                overlay=np.array([]),
                overlay_path=data["overlay_path"],
            )

            uncertainty_result = UncertaintyResult(
                predicted_class=data["predicted_class"],
                predicted_index=data["predicted_index"],
                confidence=data["mc_dropout_confidence"],
                uncertainty_score=data["uncertainty_score"],
                predictive_entropy=data["predictive_entropy"],
                expected_entropy=data["expected_entropy"],
                mutual_information=data["mutual_information"],
                probability_margin=data["probability_margin"],
                is_uncertain=data["is_uncertain"],
                flags=data["flags"],
                probabilities=data["probabilities"],
                probability_std=data["probability_std"],
                iterations=data["mc_iterations"],
            )

            report_input = ReportInput(
                image_path=image_file,
                gradcam_result=gradcam_result,
                uncertainty_result=uncertainty_result,
                case_id=prediction_id,
                patient_id=None,
            )
            report_path = str(self.pipeline.report_generator.generate(report_input))

        logger.info("PDF report generated for prediction %s", prediction_id)
        return report_path
