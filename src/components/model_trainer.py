import json
import sys
from pathlib import Path
from typing import Any

from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from src.utils.common import create_directories
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger


class MLflowEpochLogger:
    """
    Lightweight Keras callback wrapper that logs per-epoch metrics to MLflow.
    """

    def __init__(self, phase_name: str):
        self.phase_name = phase_name

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        import mlflow

        logs = logs or {}
        step = epoch + 1
        metric_names = {
            "accuracy": f"{self.phase_name}_train_accuracy",
            "val_accuracy": f"{self.phase_name}_val_accuracy",
            "loss": f"{self.phase_name}_train_loss",
            "val_loss": f"{self.phase_name}_val_loss",
        }

        for keras_name, mlflow_name in metric_names.items():
            if keras_name in logs:
                mlflow.log_metric(mlflow_name, float(logs[keras_name]), step=step)


class ModelTrainer:
    """
    Train the prepared EfficientNetB4 kidney CT classifier in two phases.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_config: DataTransformationConfig,
    ):
        self.config = config
        self.data_transformation_config = data_transformation_config
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def run(self) -> dict[str, Any]:
        try:
            import mlflow

            mlflow.set_experiment("kidney_tumor_model_training")
            self.mlflow = mlflow

            create_directories([self.config.root_dir, Path(self.config.checkpoint_path).parent])

            self.train_ds, self.val_ds, self.test_ds = self._load_datasets()
            class_weights = self._calculate_class_weights(self.train_ds)
            self._save_class_weights(class_weights)

            loaded_model = self._load_prepared_model()
            self.model = self._build_augmented_model(loaded_model)

            with mlflow.start_run(run_name="model_training"):
                phase1_history = self._train_phase(
                    phase_name="phase1",
                    epochs=self.config.phase1_epochs,
                    learning_rate=self.config.phase1_learning_rate,
                    class_weights=class_weights,
                )

                self._unfreeze_fine_tuning_layers(loaded_model)
                phase2_history = self._train_phase(
                    phase_name="phase2",
                    epochs=self.config.phase2_epochs,
                    learning_rate=self.config.phase2_learning_rate,
                    class_weights=class_weights,
                )

                combined_history = self._combine_histories([phase1_history, phase2_history])
                self._plot_training_history(combined_history)
                self._save_model()

            return {
                "trained_model_path": str(self.config.trained_model_path),
                "history_plot_path": str(self.config.history_plot_path),
                "class_weights": class_weights,
            }
        except Exception as e:
            raise KidneyTumorException(e, sys)

    def _load_datasets(self):
        logger.info("Loading train, validation, and test datasets from data transformation.")
        data_transformation = DataTransformation(config=self.data_transformation_config)
        return data_transformation.get_train_val_test_datasets()

    def _load_prepared_model(self):
        import tensorflow as tf

        if not self.config.base_model_path.exists():
            raise FileNotFoundError(f"Prepared model not found: {self.config.base_model_path}")

        logger.info("Loading prepared model from %s", self.config.base_model_path)
        return tf.keras.models.load_model(self.config.base_model_path)

    def _build_augmented_model(self, loaded_model):
        import tensorflow as tf

        input_shape = tuple(self.data_transformation_config.image_size[:2]) + (3,)

        base_model = self._find_base_model(loaded_model)
        if base_model is not None:
            base_model.trainable = False

        inputs = tf.keras.Input(shape=input_shape, name="augmented_input")
        x = tf.keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
        x = tf.keras.layers.RandomRotation(15 / 360, name="random_rotation")(x)
        x = tf.keras.layers.RandomZoom(0.10, name="random_zoom")(x)
        x = tf.keras.layers.RandomBrightness(0.15, name="random_brightness")(x)
        outputs = loaded_model(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="augmented_kidney_model")
        logger.info("Augmentation layers attached before the prepared EfficientNetB4 model.")
        return model

    def _train_phase(
        self,
        phase_name: str,
        epochs: int,
        learning_rate: float,
        class_weights: dict[int, float],
    ):
        import tensorflow as tf

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        with self.mlflow.start_run(run_name=f"model_training_{phase_name}", nested=True):
            self._log_phase_params(phase_name, epochs, learning_rate)

            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=self._get_callbacks(phase_name),
            )

            self.mlflow.log_artifact(str(self.config.csv_log_path))
            if self.config.history_plot_path.exists():
                self.mlflow.log_artifact(str(self.config.history_plot_path))

            return history

    def _get_callbacks(self, phase_name: str):
        import tensorflow as tf

        checkpoint_path = self._phase_checkpoint_path(phase_name)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        csv_append = phase_name != "phase1"

        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.reduce_lr_min_lr,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(self.config.csv_log_path),
                append=csv_append,
            ),
            self._keras_mlflow_callback(phase_name),
        ]

    def _keras_mlflow_callback(self, phase_name: str):
        import tensorflow as tf

        class KerasMLflowCallback(tf.keras.callbacks.Callback):
            def __init__(callback_self, callback_phase_name: str):
                super().__init__()
                callback_self.logger = MLflowEpochLogger(callback_phase_name)

            def on_epoch_end(callback_self, epoch, logs=None):
                callback_self.logger.on_epoch_end(epoch, logs)

        return KerasMLflowCallback(phase_name)

    def _calculate_class_weights(self, train_ds) -> dict[int, float]:
        import tensorflow as tf

        class_names = self._get_class_names()
        class_count = len(class_names)
        class_counts = {index: 0 for index in range(class_count)}

        for _, labels in train_ds:
            batch_counts = tf.reduce_sum(labels, axis=0).numpy()
            for index, count in enumerate(batch_counts):
                class_counts[index] = class_counts.get(index, 0) + int(count)

        total_samples = sum(class_counts.values())
        if total_samples == 0 or class_count == 0:
            raise ValueError("Cannot calculate class weights from an empty training dataset.")

        class_weights = {
            index: total_samples / (class_count * count)
            for index, count in class_counts.items()
            if count > 0
        }

        logger.info("Class names: %s", class_names)
        logger.info("Class counts: %s", class_counts)
        logger.info("Calculated class weights: %s", class_weights)
        return class_weights

    def _get_class_names(self) -> list[str]:
        train_dir = Path(self.data_transformation_config.root_dir) / "train"
        return [
            path.name
            for path in sorted(train_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_dir()
        ]

    def _save_class_weights(self, class_weights: dict[int, float]) -> None:
        payload = {str(class_index): float(weight) for class_index, weight in class_weights.items()}
        self.config.class_weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.class_weights_path.write_text(
            json.dumps(payload, indent=4),
            encoding="utf-8",
        )

    def _unfreeze_fine_tuning_layers(self, loaded_model) -> None:
        import tensorflow as tf

        base_model = self._find_base_model(loaded_model)
        if base_model is None:
            logger.warning("No nested base model found. Fine tuning will train the loaded model.")
            loaded_model.trainable = True
            return

        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = False

        fine_tune_from = self.config.fine_tune_from_layer
        for layer in base_model.layers[fine_tune_from:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        logger.info("Unfroze EfficientNetB4 layers from index %s for phase 2.", fine_tune_from)

    def _find_base_model(self, model):
        import tensorflow as tf

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
                return layer
        return None

    def _phase_checkpoint_path(self, phase_name: str) -> Path:
        checkpoint_path = Path(self.config.checkpoint_path)
        return checkpoint_path.with_name(f"{phase_name}_{checkpoint_path.name}")

    def _log_phase_params(self, phase_name: str, epochs: int, learning_rate: float) -> None:
        self.mlflow.log_params(
            {
                "phase": phase_name,
                "phase1_epochs": self.config.phase1_epochs,
                "phase2_epochs": self.config.phase2_epochs,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "phase1_learning_rate": self.config.phase1_learning_rate,
                "phase2_learning_rate": self.config.phase2_learning_rate,
                "batch_size": self.config.batch_size,
                "dropout_rate": self.config.dropout_rate,
                "dense_units": self.config.dense_units,
                "fine_tune_from_layer": self.config.fine_tune_from_layer,
            }
        )
        self.mlflow.log_artifact(str(self.config.class_weights_path))

    def _combine_histories(self, histories) -> dict[str, list[float]]:
        combined: dict[str, list[float]] = {}
        for history in histories:
            for key, values in history.history.items():
                combined.setdefault(key, []).extend(float(value) for value in values)
        return combined

    def _plot_training_history(self, history: dict[str, list[float]]) -> None:
        try:
            import matplotlib.pyplot as plt

            self.config.history_plot_path.parent.mkdir(parents=True, exist_ok=True)

            epochs = range(1, len(history.get("loss", [])) + 1)
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, history.get("accuracy", []), label="train_accuracy")
            plt.plot(epochs, history.get("val_accuracy", []), label="val_accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, history.get("loss", []), label="train_loss")
            plt.plot(epochs, history.get("val_loss", []), label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.config.history_plot_path)
            plt.close()
            logger.info("Training history plot saved at %s", self.config.history_plot_path)
        except ImportError:
            logger.warning("matplotlib is not installed. Skipping training history plot.")

    def _save_model(self) -> None:
        self.config.trained_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.config.trained_model_path)
        self.mlflow.log_artifact(str(self.config.trained_model_path))
        if self.config.history_plot_path.exists():
            self.mlflow.log_artifact(str(self.config.history_plot_path))
        logger.info("Final trained model saved at %s", self.config.trained_model_path)

    @staticmethod
    def _get_tensorflow():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for model training. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        return tf

    @staticmethod
    def _get_mlflow():
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is required for training logging. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        mlflow.set_experiment("kidney_tumor_model_training")
        return mlflow
