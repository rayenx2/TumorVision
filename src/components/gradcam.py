import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.entity.config_entity import GradCAMConfig
from src.utils.exception import PredictionError
from src.utils.logger import logger


@dataclass
class GradCAMResult:
    """Container for Grad-CAM prediction and visualization outputs."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    heatmap: np.ndarray
    overlay: np.ndarray
    overlay_path: Path | None = None


class GradCAM:
    """
    Generate Grad-CAM heatmaps for the kidney CT classifier.

    The trained model stores EfficientNetB4 inside a custom nested wrapper:
    model -> kidney_tumor_efficientnetb4 -> efficientnetb4 -> top_conv
    """

    def __init__(self, config: GradCAMConfig, model: Any | None = None):
        self.config = config
        self.model = model
        self.root_dir = Path(config.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._grad_model = None

    def load_model(self):
        """Load the configured Keras model if one was not supplied.

        Loading strategy:
        1. Use pre-loaded model if provided in __init__.
        2. Try local model file (fast path).
        3. Fall back to Hugging Face Hub download.
        """
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
            logger.info("Model loaded successfully")

            return self.model
        except Exception as e:
            raise PredictionError(e, sys)

    def explain(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        class_index: int | None = None,
    ) -> GradCAMResult:
        """
        Run prediction, generate a heatmap, and create an overlay image.

        Args:
            image_path: CT image path.
            output_path: Optional path where the overlay image will be saved.
            class_index: Optional target class index. Defaults to predicted class.
        """
        try:
            model = self.load_model()
            image_path = Path(image_path)

            original_image, batch = self.preprocess_image(image_path)
            predictions = model(batch, training=False).numpy()[0]

            predicted_index = int(np.argmax(predictions))
            target_index = predicted_index if class_index is None else int(class_index)

            heatmap = self.generate_heatmap(batch, target_index)
            overlay = self.overlay_heatmap(original_image, heatmap)

            overlay_path = None
            if output_path is not None:
                overlay_path = self.save_overlay(overlay, output_path)

            probabilities = {
                class_name: float(predictions[index])
                for index, class_name in enumerate(self.config.class_names)
            }

            return GradCAMResult(
                predicted_class=self.config.class_names[predicted_index],
                predicted_index=predicted_index,
                confidence=float(predictions[predicted_index]),
                probabilities=probabilities,
                heatmap=heatmap,
                overlay=overlay,
                overlay_path=overlay_path,
            )
        except Exception as e:
            raise PredictionError(e, sys)

    def preprocess_image(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load an image and return both display-ready RGB image and model batch."""
        try:
            import tensorflow as tf

            image_path = Path(image_path)
            image = tf.keras.utils.load_img(image_path, target_size=tuple(self.config.image_size))
            image_array = tf.keras.utils.img_to_array(image).astype(np.float32)

            batch = np.expand_dims(image_array.copy(), axis=0)
            batch = tf.keras.applications.efficientnet.preprocess_input(batch)

            return image_array.astype(np.uint8), batch
        except Exception as e:
            raise PredictionError(e, sys)

    def generate_heatmap(
        self, image_batch: np.ndarray, class_index: int | None = None
    ) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap for a preprocessed image batch."""
        try:
            import tensorflow as tf

            model = self.load_model()

            if self._grad_model is None:
                self._grad_model = self._build_connected_grad_model(model, tf)
            grad_model = self._grad_model

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_batch, training=False)
                if class_index is None:
                    class_index = int(tf.argmax(predictions[0]))
                class_score = predictions[:, class_index]

            grads = tape.gradient(class_score, conv_outputs)
            if grads is None:
                raise ValueError(
                    "Could not compute Grad-CAM gradients. "
                    "The target convolution output is not connected to the prediction output."
                )

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)

            max_value = tf.reduce_max(heatmap)
            if float(max_value) > 0:
                heatmap = heatmap / max_value

            return heatmap.numpy()
        except Exception as e:
            raise PredictionError(e, sys)

    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Resize and blend a Grad-CAM heatmap over an RGB image."""
        try:
            import cv2
            import matplotlib.cm as cm

            image_uint8 = self._to_uint8(image)
            heatmap_uint8 = np.uint8(255 * heatmap)

            heatmap_resized = cv2.resize(
                heatmap_uint8,
                (image_uint8.shape[1], image_uint8.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            colormap = cm.get_cmap(self.config.colormap)
            colored_heatmap = colormap(heatmap_resized / 255.0)[:, :, :3]
            colored_heatmap = np.uint8(255 * colored_heatmap)

            overlay = (1.0 - float(self.config.alpha)) * image_uint8 + float(
                self.config.alpha
            ) * colored_heatmap

            return np.clip(overlay, 0, 255).astype(np.uint8)
        except Exception as e:
            raise PredictionError(e, sys)

    def save_overlay(self, overlay: np.ndarray, output_path: str | Path) -> Path:
        """Save an RGB overlay image to disk."""
        try:
            import cv2

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr_overlay)

            logger.info("Saved Grad-CAM overlay to %s", output_path)
            return output_path
        except Exception as e:
            raise PredictionError(e, sys)

    def _get_target_layer(self, model):
        """Find the configured target convolution layer, preferring known nested access."""
        try:
            wrapper = model.get_layer(self.config.nested_wrapper_name)
            backbone = wrapper.get_layer(self.config.nested_backbone_name)
            return backbone.get_layer(self.config.last_conv_layer_name)
        except Exception:
            logger.warning(
                "Nested Grad-CAM layer lookup failed; falling back to recursive search for %s",
                self.config.last_conv_layer_name,
            )

        target_layer = self._find_layer_recursive(model, self.config.last_conv_layer_name)
        if target_layer is None:
            raise ValueError(f"Could not find layer: {self.config.last_conv_layer_name}")

        return target_layer

    def _build_connected_grad_model(self, model, tf):
        """
        Build a Grad-CAM model whose activation and prediction outputs share one graph.

        Keras 3 cannot always connect `target_layer.output` from a nested model directly
        to the outer model input. To keep the graph connected, we call the nested
        EfficientNet backbone as a multi-output submodel inside the wrapper path.
        """
        wrapper = model.get_layer(self.config.nested_wrapper_name)
        backbone = wrapper.get_layer(self.config.nested_backbone_name)
        target_layer = backbone.get_layer(self.config.last_conv_layer_name)

        backbone_grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[self._first_output(target_layer), self._first_output(backbone)],
            name=f"{backbone.name}_gradcam",
        )

        inputs = model.inputs
        x = inputs[0] if len(inputs) == 1 else inputs
        conv_outputs = None

        for layer in model.layers:
            if self._is_input_layer(layer):
                continue

            if layer.name == self.config.nested_wrapper_name:
                x, conv_outputs = self._call_wrapper_with_gradcam_backbone(
                    wrapper=layer,
                    backbone_grad_model=backbone_grad_model,
                    input_tensor=x,
                )
            else:
                x = self._call_layer(layer, x, training=False)

        if conv_outputs is None:
            raise ValueError(
                f"Layer {self.config.last_conv_layer_name} was not reached while "
                "building the connected Grad-CAM graph."
            )

        return tf.keras.Model(inputs=inputs, outputs=[conv_outputs, x], name="gradcam_model")

    def _call_wrapper_with_gradcam_backbone(self, wrapper, backbone_grad_model, input_tensor):
        x = input_tensor
        conv_outputs = None

        for layer in wrapper.layers:
            if self._is_input_layer(layer):
                continue

            if layer.name == self.config.nested_backbone_name:
                backbone_outputs = backbone_grad_model(x, training=False)
                conv_outputs = backbone_outputs[0]
                x = backbone_outputs[1]
            else:
                x = self._call_layer(layer, x, training=False)

        return x, conv_outputs

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
    def _first_output(layer):
        outputs = getattr(layer, "outputs", None)
        if outputs:
            return outputs[0]
        output = layer.output
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def _find_layer_recursive(self, layer, layer_name: str):
        if getattr(layer, "name", None) == layer_name:
            return layer

        for child in getattr(layer, "layers", []):
            found = self._find_layer_recursive(child, layer_name)
            if found is not None:
                return found

        return None

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if image.dtype == np.uint8:
            return image

        image = image.copy()
        if image.max() <= 1.0:
            image = image * 255.0

        return np.clip(image, 0, 255).astype(np.uint8)
