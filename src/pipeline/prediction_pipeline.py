import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.components.gradcam import GradCAM, GradCAMResult
from src.components.report_generator import ReportGenerator, ReportInput
from src.components.uncertainty import UncertaintyEstimator, UncertaintyResult
from src.config.configuration import ConfigurationManager
from src.utils.exception import PredictionError
from src.utils.logger import logger


@dataclass
class PredictionResult:
    """Unified result from the prediction pipeline.

    Contains all data needed for API responses, UI display, and PDF reports.
    """

    # Image Info
    image_path: Path
    case_id: str
    timestamp: str

    # Prediction from MC Dropout, the primary source
    predicted_class: str
    predicted_index: int

    # Confidence values from both inference paths
    single_shot_confidence: float
    mc_dropout_confidence: float

    # Per-class probabilities
    probabilities: dict
    probability_std: dict

    # Uncertainty metrics
    uncertainty_score: float
    mutual_information: float
    predictive_entropy: float
    expected_entropy: float
    probability_margin: float

    # Flags
    is_uncertain: bool
    is_low_confidence: bool
    flags: list = field(default_factory=list)

    # Artifacts
    overlay_path: Path | None = None
    report_path: Path | None = None

    # Metadata
    mc_iterations: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict for API responses."""
        return self._serialize_paths(asdict(self))

    @classmethod
    def _serialize_paths(cls, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: cls._serialize_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._serialize_paths(item) for item in value]
        return value


class PredictionPipeline:
    """End-to-end prediction pipeline.

    Usage:
        config_mgr = ConfigurationManager()
        pipeline = PredictionPipeline(config_mgr)
        result = pipeline.predict("path/to/image.jpg", generate_report=True)
        print(result.predicted_class)
    """

    def __init__(self, config_manager: ConfigurationManager):
        """Initialize pipeline with all sub-configs.

        Components are lazy-initialized on first use to keep startup fast.
        """
        self.config_manager = config_manager
        self.gradcam_config = config_manager.get_gradcam_config()
        self.uncertainty_config = config_manager.get_uncertainty_config()
        self.report_config = config_manager.get_report_config()

        self._gradcam: GradCAM | None = None
        self._uncertainty: UncertaintyEstimator | None = None
        self._report_generator: ReportGenerator | None = None

        logger.info("PredictionPipeline initialized (components lazy-loaded)")

    @property
    def gradcam(self) -> GradCAM:
        """Lazy-load GradCAM component."""
        if self._gradcam is None:
            self._gradcam = GradCAM(config=self.gradcam_config)
            logger.info("GradCAM component initialized")
        return self._gradcam

    @property
    def uncertainty(self) -> UncertaintyEstimator:
        """Lazy-load UncertaintyEstimator component."""
        if self._uncertainty is None:
            self._uncertainty = UncertaintyEstimator(config=self.uncertainty_config)
            logger.info("UncertaintyEstimator component initialized")
        return self._uncertainty

    @property
    def report_generator(self) -> ReportGenerator:
        """Lazy-load ReportGenerator component."""
        if self._report_generator is None:
            self._report_generator = ReportGenerator(config=self.report_config)
            logger.info("ReportGenerator component initialized")
        return self._report_generator

    def predict(
        self,
        image_path: str | Path,
        case_id: str | None = None,
        patient_id: str | None = None,
        generate_report: bool = False,
    ) -> PredictionResult:
        """Run full prediction pipeline on a single image.

        Args:
            image_path: Path to the CT scan image.
            case_id: Optional case identifier. Generated if omitted.
            patient_id: Optional patient identifier for the PDF report.
            generate_report: If True, generate a PDF report.

        Returns:
            PredictionResult with prediction data and artifact paths.
        """
        try:
            case_id = case_id or self._generate_case_id()
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            overlay_path = Path(self.gradcam_config.root_dir) / f"case_{case_id}_overlay.png"
            timestamp = datetime.now().isoformat(timespec="seconds")
            logger.info("Starting prediction for case %s", case_id)

            gradcam_result = self.gradcam.explain(image_path, output_path=overlay_path)

            self.uncertainty.model = self.gradcam.model
            self.uncertainty._inference_model = None
            uncertainty_result = self.uncertainty.estimate(image_path)

            report_path = None
            if generate_report:
                report_input = ReportInput(
                    image_path=image_path,
                    gradcam_result=gradcam_result,
                    uncertainty_result=uncertainty_result,
                    case_id=case_id,
                    patient_id=patient_id,
                )
                report_path = self.report_generator.generate(report_input)

            result = self._build_prediction_result(
                image_path=image_path,
                case_id=case_id,
                timestamp=timestamp,
                gradcam_result=gradcam_result,
                uncertainty_result=uncertainty_result,
                overlay_path=Path(gradcam_result.overlay_path or overlay_path),
                report_path=report_path,
            )

            # Save the prediction result to disk for later
            # report generation without re-running inference

            import json

            predictions_dir = Path("artifacts/predictions")
            predictions_dir.mkdir(parents=True, exist_ok=True)
            prediction_json_path = predictions_dir / f"{case_id}.json"
            prediction_json_path.write_text(
                json.dumps(result.to_dict(), indent=4), encoding="utf-8"
            )

            logger.info("Prediction completed for case %s", case_id)
            return result

        except Exception as e:
            raise PredictionError(e, sys)

    def _generate_case_id(self) -> str:
        """Generate a unique case ID: KCT-YYYYMMDD-XXXX."""
        date_str = datetime.now().strftime("%Y%m%d")
        random_part = uuid4().hex[:4].upper()
        return f"KCT-{date_str}-{random_part}"

    def _build_prediction_result(
        self,
        image_path: Path,
        case_id: str,
        timestamp: str,
        gradcam_result: GradCAMResult,
        uncertainty_result: UncertaintyResult,
        overlay_path: Path,
        report_path: Path | None,
    ) -> PredictionResult:
        """Combine component outputs into one unified PredictionResult."""
        flags = list(uncertainty_result.flags or [])

        return PredictionResult(
            image_path=image_path,
            case_id=case_id,
            timestamp=timestamp,
            predicted_class=uncertainty_result.predicted_class,
            predicted_index=int(uncertainty_result.predicted_index),
            single_shot_confidence=float(gradcam_result.confidence),
            mc_dropout_confidence=float(uncertainty_result.confidence),
            probabilities=dict(uncertainty_result.probabilities),
            probability_std=dict(uncertainty_result.probability_std),
            uncertainty_score=float(uncertainty_result.uncertainty_score),
            mutual_information=float(uncertainty_result.mutual_information),
            predictive_entropy=float(uncertainty_result.predictive_entropy),
            expected_entropy=float(uncertainty_result.expected_entropy),
            probability_margin=float(uncertainty_result.probability_margin),
            is_uncertain=bool("high_uncertainty" in flags),
            is_low_confidence=bool("low_confidence" in flags),
            flags=flags,
            overlay_path=overlay_path,
            report_path=report_path,
            mc_iterations=int(uncertainty_result.iterations),
        )
