import tensorflow as tf
from huggingface_hub import hf_hub_download

from api.config import Settings
from src.utils.logger import api_logger as logger


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: tf.keras.Model | None = None

    def load(self) -> None:
        try:
            logger.info(
                "Loading model from Hugging Face repo %s/%s",
                self.settings.hf_model_repo,
                self.settings.hf_model_filename,
            )
            model_path = hf_hub_download(
                repo_id=self.settings.hf_model_repo,
                filename=self.settings.hf_model_filename,
            )
            self._model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully from %s", model_path)
        except Exception:
            logger.exception("Failed to load model")
            raise

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        return self._model
