import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.utils.logger import logger


@dataclass
class DriftReport:
    drift_detected: bool
    drift_score: float
    drifted_features: list
    timestamp: str
    report_path: str


class DriftDetector:
    """
    Detects data drift between reference (training) data
    and current (production) data using Evidently AI.

    For CT scan images, we compare extracted features:
    - Mean pixel intensity
    - Standard deviation
    - Contrast
    - Entropy
    - Sharpness
    """

    def __init__(
        self,
        reference_data_path: str,
        current_data_path: str,
        drift_threshold: float = 0.6,
        report_output_dir: str = "reports/drift",
    ):
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.drift_threshold = drift_threshold
        self.report_output_dir = Path(report_output_dir)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

    def _load_feature_stats(self, path: Path) -> pd.DataFrame:
        """
        Load pre-extracted image feature statistics.
        These are extracted during prediction and saved as CSV.
        Columns: mean_intensity, std_intensity, contrast, entropy, sharpness
        """
        if not path.exists():
            raise FileNotFoundError(f"Feature stats file not found: {path}")
        return pd.read_csv(path)

    def run(self) -> DriftReport:
        """Run drift detection and return a structured report."""
        logger.info("Starting drift detection...")

        reference_df = self._load_feature_stats(self.reference_data_path)
        current_df = self._load_feature_stats(self.current_data_path)

        logger.info(f"Reference samples: {len(reference_df)} | Current samples: {len(current_df)}")

        # Define which columns to monitor for drift
        data_definition = DataDefinition(
            numerical_columns=[
                "mean_intensity",
                "std_intensity",
                "contrast",
                "entropy",
                "sharpness",
            ]
        )

        reference_dataset = Dataset.from_pandas(
            reference_df,
            data_definition=data_definition,
        )

        current_dataset = Dataset.from_pandas(
            current_df,
            data_definition=data_definition,
        )

        # Run Evidently drift report
        report = Report([DataDriftPreset()])
        result = report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.report_output_dir / f"drift_report_{timestamp}.html"
        result.save_html(str(html_path))
        logger.info(f"Drift HTML report saved: {html_path}")

        # Parse drift score and drifted features
        drift_score, drifted_features = self._parse_results(result)

        drift_detected = drift_score >= self.drift_threshold and len(current_df) >= 100

        logger.info(f"Drift score: {drift_score:.4f} | Threshold: {self.drift_threshold}")
        logger.info(f"Drift detected: {drift_detected}")
        if drifted_features:
            logger.warning(f"Drifted features: {drifted_features}")

        # Save JSON summary
        json_summary = {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "drift_threshold": self.drift_threshold,
            "drifted_features": drifted_features,
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
            "timestamp": timestamp,
            "html_report": str(html_path),
        }

        json_path = self.report_output_dir / "drift_summary.json"
        with open(json_path, "w") as f:
            json.dump(json_summary, f, indent=2)

        logger.info(f"Drift summary JSON saved: {json_path}")

        return DriftReport(
            drift_detected=drift_detected,
            drift_score=drift_score,
            drifted_features=drifted_features,
            timestamp=timestamp,
            report_path=str(html_path),
        )

    def _parse_results(self, result) -> tuple[float, list]:
        """Extract drift score and drifted features from Evidently 0.7.x result."""

        drifted_features = []
        drift_score = 0.0

        try:
            result_dict = result.dict()
            metrics = result_dict.get("metrics", [])

            for metric in metrics:
                metric_name = metric.get("metric_name", "")
                value = metric.get("value", {})

                # Dataset-level drift share
                if "DriftedColumnsCount" in metric_name:
                    if isinstance(value, dict):
                        drift_score = float(value.get("share", 0.0))

                # Per-column drift
                if "ValueDrift" in metric_name:
                    config = metric.get("config", {})
                    column = config.get("column", "")
                    threshold = config.get("threshold", 0.1)

                    if isinstance(value, float) and value > threshold:
                        drifted_features.append(column)

        except Exception as e:
            logger.error(f"Error parsing Evidently results: {e}")

        return drift_score, drifted_features
