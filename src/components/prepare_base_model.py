import sys
from pathlib import Path

from src.entity.config_entity import PrepareBaseModelConfig
from src.utils.common import create_directories
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger


class PrepareBaseModel:
    """
    Build and save an EfficientNetB4 base model and a project-specific
    classification model for kidney CT image classification.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        """
        Create EfficientNetB4 using the configured input shape, weights, and
        include_top setting, then save it to base_model_path.
        """
        try:
            tf = self._get_tensorflow()
            create_directories([self.config.root_dir])

            self.model = tf.keras.applications.EfficientNetB4(
                input_shape=tuple(self.config.image_size),
                weights=self.config.weights,
                include_top=self.config.include_top,
            )

            self.save_model(path=self.config.base_model_path, model=self.model)
            logger.info("EfficientNetB4 base model saved at %s", self.config.base_model_path)
            return self.model

        except Exception as e:
            raise KidneyTumorException(e, sys)

    def update_base_model(self):
        """
        Attach a trainable classification head to the EfficientNetB4 base model
        and save the complete model to updated_base_model_path.
        """
        try:
            if self.model is None:
                self.get_base_model()

            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.classes,
            )
            self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
            logger.info(
                "Updated EfficientNetB4 classification model saved at %s",
                self.config.updated_base_model_path,
            )
            return self.full_model

        except Exception as e:
            raise KidneyTumorException(e, sys)

    def run(self):
        """
        Execute the full prepare-base-model stage.
        """
        self.get_base_model()
        return self.update_base_model()

    def _prepare_full_model(self, model, classes: int):
        tf = self._get_tensorflow()

        model.trainable = False

        inputs = tf.keras.Input(shape=tuple(self.config.image_size), name="input_image")
        x = model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
        outputs = tf.keras.layers.Dense(
            classes,
            activation="softmax",
            name="classification_head",
        )(x)

        full_model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="kidney_tumor_efficientnetb4",
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        full_model.summary(print_fn=logger.info)
        return full_model

    @staticmethod
    def save_model(path: Path, model) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to prepare EfficientNetB4. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        return tf
