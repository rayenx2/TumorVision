"""
Grad-CAM explainability for TumorVision kidney tumor detection model.

Generates activation maps showing which CT scan regions drove the EfficientNetB4
prediction. Works as a standalone CLI tool — no running server required.

Usage:
    python scripts/gradcam_explainer.py scan.jpg --model weights/model.keras
    python scripts/gradcam_explainer.py scan.jpg --demo

Author: Rayen Lassoued
        github.com/rayenx2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE = (380, 380)


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

def compute_gradcam_tf(
    model,
    img_array: np.ndarray,
    target_layer_name: str = "top_conv",
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap using TensorFlow GradientTape.

    Args:
        model:             Loaded tf.keras Model.
        img_array:         Preprocessed image array, shape (1, H, W, C).
        target_layer_name: Name of the convolutional layer to visualize.
                           EfficientNetB4 default is 'top_conv'.
        target_class:      Class index to explain. Uses argmax if None.

    Returns:
        Heatmap as float32 numpy array (H, W), values in [0, 1].
    """
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_outputs)  # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]  # (h, w, C)
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    cam = tf.squeeze(cam).numpy()

    # ReLU + normalize
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def monte_carlo_uncertainty(
    model,
    img_array: np.ndarray,
    n_passes: int = 10,
) -> dict:
    """
    Run N stochastic forward passes with dropout enabled.

    Returns:
        dict with keys: mean_probs, std_probs, predicted_class, uncertainty_score
    """
    import tensorflow as tf

    preds = []
    for _ in range(n_passes):
        # training=True keeps dropout active at inference time
        p = model(img_array, training=True).numpy()[0]
        preds.append(p)

    preds = np.stack(preds)
    mean_probs = preds.mean(axis=0)
    std_probs = preds.std(axis=0)
    pred_idx = int(mean_probs.argmax())

    return {
        "mean_probs": mean_probs.tolist(),
        "std_probs": std_probs.tolist(),
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_class_idx": pred_idx,
        "uncertainty_score": float(std_probs[pred_idx]),
        "is_uncertain": float(std_probs[pred_idx]) > 0.05,
    }


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load a kidney CT scan image and apply EfficientNetB4 preprocessing.

    Supports JPEG, PNG. Returns array of shape (1, 380, 380, 3).
    """
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def apply_clahe(image_path: str, output_path: str) -> str:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a CT scan.
    Improves soft-tissue contrast before model inference.
    """
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    cv2.imwrite(output_path, enhanced)
    return output_path


def overlay_heatmap(
    original_image_path: str,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
) -> str:
    """
    Overlay Grad-CAM heatmap on original CT scan and save result.

    Args:
        original_image_path: Path to original scan image.
        heatmap:             Grad-CAM heatmap, shape (H, W), values in [0, 1].
        output_path:         Where to save the overlay image.
        alpha:               Heatmap opacity (0 = invisible, 1 = full).

    Returns:
        Path to saved overlay image.
    """
    import cv2

    original = cv2.imread(original_image_path)
    if original is None:
        raise ValueError(f"Could not load image: {original_image_path}")

    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
    cv2.imwrite(output_path, overlay)
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def explain_prediction(
    image_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "gradcam_output",
    target_layer: str = "top_conv",
    run_mc_dropout: bool = True,
    mc_passes: int = 10,
) -> dict:
    """
    Full pipeline: load image -> preprocess -> classify -> Grad-CAM -> overlay -> save.

    Args:
        image_path:      Path to CT scan image.
        model_path:      Path to .keras model weights. Uses dummy model if None.
        output_dir:      Directory to save Grad-CAM overlay images.
        target_layer:    EfficientNetB4 layer to visualize (default: 'top_conv').
        run_mc_dropout:  Whether to run Monte Carlo Dropout uncertainty estimation.
        mc_passes:       Number of stochastic passes for MC Dropout.

    Returns:
        dict with prediction details, uncertainty scores, and output file paths.
    """
    import tensorflow as tf

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_array = load_and_preprocess(image_path)

    if model_path and Path(model_path).exists():
        model = tf.keras.models.load_model(model_path)
    else:
        # Placeholder for demo — returns random logits
        inp = tf.keras.Input(shape=(380, 380, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inp)
        out = tf.keras.layers.Dense(4, activation="softmax", name="predictions")(x)
        model = tf.keras.Model(inp, out)

    # Standard prediction
    logits = model(img_array, training=False).numpy()[0]
    pred_idx = int(logits.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(logits[pred_idx])

    result = {
        "image": image_path,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(logits)
        },
    }

    # Monte Carlo Dropout uncertainty
    if run_mc_dropout:
        mc = monte_carlo_uncertainty(model, img_array, n_passes=mc_passes)
        result["uncertainty"] = {
            "score": mc["uncertainty_score"],
            "is_uncertain": mc["is_uncertain"],
            "mc_passes": mc_passes,
        }

    # Grad-CAM (only if model has the target layer)
    layer_names = [l.name for l in model.layers]
    if target_layer in layer_names:
        cam = compute_gradcam_tf(model, img_array, target_layer, pred_idx)
        overlay_path = str(out_dir / f"{Path(image_path).stem}_gradcam.png")
        overlay_heatmap(image_path, cam, overlay_path)
        result["gradcam_output"] = overlay_path
    else:
        result["gradcam_output"] = None
        result["gradcam_note"] = (
            f"Layer '{target_layer}' not found. "
            f"Available conv layers: {[l.name for l in model.layers if 'conv' in l.name][:5]}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TumorVision — Grad-CAM explainability for kidney CT scans"
    )
    parser.add_argument("image", nargs="?", help="Path to CT scan image (JPEG/PNG)")
    parser.add_argument("--model", help="Path to trained .keras model weights")
    parser.add_argument("--output-dir", default="gradcam_output", help="Output directory for heatmaps")
    parser.add_argument("--layer", default="top_conv", help="EfficientNetB4 layer to visualize")
    parser.add_argument("--mc-passes", type=int, default=10, help="Monte Carlo Dropout passes")
    parser.add_argument("--no-mc", action="store_true", help="Skip Monte Carlo Dropout")
    parser.add_argument("--demo", action="store_true", help="Show example output and exit")
    args = parser.parse_args()

    if args.demo or not args.image:
        print("TumorVision Grad-CAM Explainer")
        print("Author: Rayen Lassoued | github.com/rayenx2")
        print()
        print("Usage:")
        print("  python scripts/gradcam_explainer.py scan.jpg --model weights/model.keras")
        print("  python scripts/gradcam_explainer.py scan.jpg --layer top_conv --mc-passes 20")
        print()
        print("Example output:")
        example = {
            "image": "scan_001.jpg",
            "predicted_class": "Tumor",
            "confidence": 0.9764,
            "probabilities": {"Cyst": 0.0093, "Normal": 0.0051, "Stone": 0.0092, "Tumor": 0.9764},
            "uncertainty": {"score": 0.021, "is_uncertain": False, "mc_passes": 10},
            "gradcam_output": "gradcam_output/scan_001_gradcam.png",
        }
        print(json.dumps(example, indent=2))
        sys.exit(0)

    result = explain_prediction(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        target_layer=args.layer,
        run_mc_dropout=not args.no_mc,
        mc_passes=args.mc_passes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
