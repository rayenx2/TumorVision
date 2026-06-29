import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.entity.config_entity import UncertaintyConfig
from src.utils.exception import PredictionError
from src.utils.logger import logger


@dataclass
class UncertaintyResult:
    """Prediction result enriched with MC Dropout uncertainty metrics."""

    predicted_class: str
    predicted_index: int
    confidence: float
    uncertainty_score: float
    predictive_entropy: float
    expected_entropy: float
    mutual_information: float
    probability_margin: float
    is_uncertain: bool
    flags: list[str]
    probabilities: dict[str, float]
    probability_std: dict[str, float]
    iterations: int


class UncertaintyEstimator:
    """
    Estimate predictive uncertainty for kidney CT classification with MC Dropout.

    The trained production model may be wrapped with augmentation layers:
    augmented_model -> kidney_tumor_efficientnetb4 -> EfficientNetB4/dropout/head.
    For uncertainty estimation we prefer calling the configured nested wrapper
    directly so random augmentation is not active during repeated passes.
    """

    def __init__(self, config: UncertaintyConfig, model: Any | None = None):
        self.config = config
        self.model = model
        self.root_dir = Path(config.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._inference_model = None

    def load_model(self):
        """Load a Keras model from local disk, or download it from Hugging Face."""
        try:
            if self.model is not None:
                return self.model

            import tensorflow as tf

            local_path = Path(self.config.model_path)
            if local_path.exists():
                logger.info("Loading model from local path: %s", local_path)
                model_path = str(local_path)
            else:
                from huggingface_hub import hf_hub_download

                logger.info(
                    "Local model not found at %s. Downloading from Hugging Face: %s/%s",
                    local_path,
                    self.config.hf_repo_id,
                    self.config.hf_model_filename,
                )
                model_path = hf_hub_download(
                    repo_id=self.config.hf_repo_id,
                    filename=self.config.hf_model_filename,
                )
                logger.info("Model downloaded to: %s", model_path)

            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully for uncertainty estimation")
            return self.model
        except Exception as e:
            raise PredictionError(e, sys)

    def predict(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UncertaintyResult:
        """Run MC Dropout prediction for one image and optionally save JSON output."""
        try:
            batch = self.preprocess_image(image_path)
            samples = self.mc_dropout_predict(batch)
            result = self.summarize_predictions(samples)

            if output_path is not None:
                self.save_result(result, output_path)

            return result
        except Exception as e:
            raise PredictionError(e, sys)

    def estimate(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UncertaintyResult:
        """Alias for predict, kept for readable pipeline code."""
        return self.predict(image_path=image_path, output_path=output_path)

    def estimate_uncertainty(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UncertaintyResult:
        """Compatibility alias used by notebooks and scripts."""
        return self.predict(image_path=image_path, output_path=output_path)

    def run(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> UncertaintyResult:
        """Pipeline-friendly entry point."""
        return self.predict(image_path=image_path, output_path=output_path)

    def preprocess_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image, resize it, and apply EfficientNet preprocessing."""
        try:
            import tensorflow as tf

            image = tf.keras.utils.load_img(
                Path(image_path),
                target_size=tuple(self.config.image_size),
            )
            image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
            batch = np.expand_dims(image_array, axis=0)
            return tf.keras.applications.efficientnet.preprocess_input(batch)
        except Exception as e:
            raise PredictionError(e, sys)

    def mc_dropout_predict(self, image_batch: np.ndarray) -> np.ndarray:
        """
        Collect repeated stochastic forward passes with only the classifier dropout enabled.

        Strategy:
        image -> backbone(training=False) -> GAP -> BatchNorm(training=False)
              -> Dropout(training=True) -> classification head -> softmax
        """
        try:
            import tensorflow as tf

            # Set fixed seeds so MC Dropout is deterministic across API and Celery processes
            tf.random.set_seed(42)
            np.random.seed(42)

            model = self._get_inference_model()
            predictions = []

            for _ in range(int(self.config.mc_iterations)):
                prediction = self._mc_dropout_forward_pass(model, image_batch)
                prediction = self._to_numpy(prediction)[0]
                predictions.append(self._normalize_probabilities(prediction))

            return np.asarray(predictions, dtype=np.float64)
        except Exception as e:
            raise PredictionError(e, sys)

    def summarize_predictions(self, prediction_samples: np.ndarray) -> UncertaintyResult:
        """Convert MC probability samples into a compact uncertainty report."""
        try:
            samples = np.asarray(prediction_samples, dtype=np.float64)
            if samples.ndim != 2:
                raise ValueError("prediction_samples must have shape (iterations, classes).")
            if samples.shape[0] == 0:
                raise ValueError("At least one MC Dropout prediction is required.")
            if samples.shape[1] != len(self.config.class_names):
                raise ValueError(
                    "Prediction class count does not match configured class names: "
                    f"{samples.shape[1]} != {len(self.config.class_names)}"
                )

            mean_probabilities = samples.mean(axis=0)
            std_probabilities = samples.std(axis=0)
            predicted_index = int(np.argmax(mean_probabilities))
            confidence = float(mean_probabilities[predicted_index])
            uncertainty_score = float(std_probabilities[predicted_index])

            sorted_probs = np.sort(mean_probabilities)
            probability_margin = (
                float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else confidence
            )

            predictive_entropy = self._entropy(mean_probabilities)
            expected_entropy = float(np.mean([self._entropy(sample) for sample in samples]))
            mutual_information = float(max(predictive_entropy - expected_entropy, 0.0))

            flags = self._build_flags(confidence, uncertainty_score)

            probabilities = {
                class_name: float(mean_probabilities[index])
                for index, class_name in enumerate(self.config.class_names)
            }
            probability_std = {
                class_name: float(std_probabilities[index])
                for index, class_name in enumerate(self.config.class_names)
            }

            return UncertaintyResult(
                predicted_class=self.config.class_names[predicted_index],
                predicted_index=predicted_index,
                confidence=confidence,
                uncertainty_score=uncertainty_score,
                predictive_entropy=predictive_entropy,
                expected_entropy=expected_entropy,
                mutual_information=mutual_information,
                probability_margin=probability_margin,
                is_uncertain=bool(flags),
                flags=flags,
                probabilities=probabilities,
                probability_std=probability_std,
                iterations=int(samples.shape[0]),
            )
        except Exception as e:
            raise PredictionError(e, sys)

    def save_result(self, result: UncertaintyResult, output_path: str | Path) -> Path:
        """Save an uncertainty result as JSON."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(asdict(result), indent=4),
                encoding="utf-8",
            )
            logger.info("Saved uncertainty result to %s", output_path)
            return output_path
        except Exception as e:
            raise PredictionError(e, sys)

    def _get_inference_model(self):
        if self._inference_model is not None:
            return self._inference_model

        model = self.load_model()
        try:
            self._inference_model = model.get_layer(self.config.nested_wrapper_name)
            logger.info(
                "Using nested wrapper %s for MC Dropout inference.",
                self.config.nested_wrapper_name,
            )
        except Exception:
            logger.warning(
                "Nested wrapper %s not found. Using the loaded model directly.",
                self.config.nested_wrapper_name,
            )
            self._inference_model = model

        return self._inference_model

    def _build_flags(self, confidence: float, uncertainty_score: float) -> list[str]:
        flags: list[str] = []

        if confidence < float(self.config.confidence_threshold):
            flags.append("low_confidence")

        if uncertainty_score > float(self.config.uncertainty_threshold):
            flags.append("high_uncertainty")

        return flags

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _call_model(model, image_batch: np.ndarray, training: bool):
        if callable(model):
            try:
                return model(image_batch, training=training)
            except TypeError:
                return model(image_batch)

        if hasattr(model, "predict"):
            return model.predict(image_batch, verbose=0)

        raise TypeError("Model must be callable or expose a predict method.")

    def _mc_dropout_forward_pass(self, model, image_batch: np.ndarray):
        if not hasattr(model, "layers"):
            return self._call_model(model, image_batch, training=True)

        x = image_batch
        dropout_layers_seen = 0

        for layer in model.layers:
            if self._is_input_layer(layer):
                continue

            use_training_mode = self._is_dropout_layer(layer)
            if use_training_mode:
                dropout_layers_seen += 1

            x = self._call_layer(layer, x, training=use_training_mode)

        if dropout_layers_seen == 0:
            logger.warning(
                "No Dropout layer found in %s. MC predictions may be deterministic.",
                getattr(model, "name", model.__class__.__name__),
            )

        return x

    @staticmethod
    def _call_layer(layer, inputs, training: bool = False):
        try:
            return layer(inputs, training=training)
        except TypeError:
            return layer(inputs)

    @staticmethod
    def _is_input_layer(layer) -> bool:
        return layer.__class__.__name__ == "InputLayer"

    @staticmethod
    def _is_dropout_layer(layer) -> bool:
        return layer.__class__.__name__ == "Dropout"

    @staticmethod
    def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 0.0, None)
        total = values.sum()
        if total <= 0:
            raise ValueError("Model returned non-positive probabilities.")
        return values / total

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        return float(-np.sum(probabilities * np.log(probabilities)))


# Backward-friendly names for simple imports in notebooks or scripts.
MCUncertaintyEstimator = UncertaintyEstimator
MCUncertainty = UncertaintyEstimator
MCDropoutUncertainty = UncertaintyEstimator
Uncertainty = UncertaintyEstimator
UncertaintyQuantifier = UncertaintyEstimator
