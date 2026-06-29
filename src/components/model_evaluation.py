import json
import sys
from pathlib import Path

import numpy as np

from src.entity.config_entity import DataTransformationConfig, ModelEvaluationConfig
from src.utils.common import create_directories
from src.utils.exception import ModelEvaluationError
from src.utils.logger import logger


class ModelEvaluation:
    """
    Evaluate the trained kidney CT classifier on the held-out test split.
    """

    CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

    def __init__(
        self,
        config: ModelEvaluationConfig,
        data_transformation_config: DataTransformationConfig,
    ):
        self.config = config
        self.data_transformation_config = data_transformation_config

    def run(self) -> dict:
        try:
            import mlflow
            import tensorflow as tf
            from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

            mlflow.set_experiment("kidney_tumor_model_training")

            create_directories(
                [
                    self.config.root_dir,
                    Path(self.config.metrics_path).parent,
                    Path(self.config.confusion_matrix_path).parent,
                    Path(self.config.roc_curve_path).parent,
                ]
            )

            with mlflow.start_run(run_name="model_evaluation"):
                # ── Load model ──────────────────────────────────────────────
                model = self._load_model(tf)
                logger.info("Model loaded successfully.")

                # ── Load test dataset ────────────────────────────────────────
                test_ds = self._load_test_dataset(tf)
                logger.info("Test dataset loaded.")

                # ── Collect predictions ──────────────────────────────────────
                y_true, y_pred_proba = self._collect_predictions(model, test_ds)
                y_pred = np.argmax(y_pred_proba, axis=1)

                # ── Metrics ──────────────────────────────────────────────────
                metrics = self._calculate_metrics(
                    y_true, y_pred, y_pred_proba, classification_report
                )
                self._save_metrics(metrics)

                mlflow.log_metrics(
                    {
                        "test_accuracy": metrics.get("accuracy", 0.0),
                        "test_auc_roc": metrics.get("auc_roc", 0.0),
                        "test_sensitivity": metrics.get("sensitivity", 0.0),
                        "test_specificity": metrics.get("specificity", 0.0),
                        "test_f1_score": metrics.get("f1_score", 0.0),
                        "test_precision": metrics.get("precision", 0.0),
                    }
                )

                # ── Plots ────────────────────────────────────────────────────
                self._save_confusion_matrix(y_true, y_pred, confusion_matrix)
                self._save_roc_curve(y_true, y_pred_proba, roc_curve, auc)

                if Path(self.config.metrics_path).exists():
                    mlflow.log_artifact(str(self.config.metrics_path))
                if Path(self.config.confusion_matrix_path).exists():
                    mlflow.log_artifact(str(self.config.confusion_matrix_path))
                if Path(self.config.roc_curve_path).exists():
                    mlflow.log_artifact(str(self.config.roc_curve_path))

                # ── Evaluation gate ──────────────────────────────────────────
                self._check_evaluation_gate(metrics)

            return metrics

        except Exception as e:
            raise ModelEvaluationError(e, sys)

    def _load_model(self, tf):
        model_path = Path(self.config.metrics_path).parent.parent / "model_trainer" / "model.keras"

        if not model_path.exists():
            model_path = Path("artifacts/model_trainer/model.keras")

        if not model_path.exists():
            logger.info("Local model not found. Downloading from Hugging Face Hub.")
            import os

            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=os.environ.get("HF_REPO_ID", "Himel000/kidney-tumor-efficientnetb4"),
                filename="model.keras",
            )

        return tf.keras.models.load_model(model_path)

    def _load_test_dataset(self, tf):
        from src.components.data_transformation import DataTransformation

        data_transformation = DataTransformation(config=self.data_transformation_config)
        _, _, test_ds = data_transformation.get_train_val_test_datasets()
        return test_ds

    def _collect_predictions(self, model, test_ds):
        y_true_list = []
        y_pred_list = []

        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_pred_list.append(preds)
            y_true_list.append(np.argmax(labels.numpy(), axis=1))

        y_true = np.concatenate(y_true_list)
        y_pred_proba = np.concatenate(y_pred_list)
        return y_true, y_pred_proba

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, classification_report):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        accuracy = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="weighted"))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        sensitivity = float(recall_score(y_true, y_pred, average="weighted"))

        # AUC-ROC — one-vs-rest for multiclass
        try:
            auc_roc = float(
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            auc_roc = 0.0
            logger.warning("AUC-ROC could not be calculated.")

        # Specificity — mean across classes
        from sklearn.metrics import confusion_matrix as cm_fn

        cm = cm_fn(y_true, y_pred)
        specificities = []
        for i in range(len(self.CLASS_NAMES)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.CLASS_NAMES,
            output_dict=True,
        )

        metrics = {
            "accuracy": accuracy,
            "auc_roc": auc_roc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "precision": precision,
            "classification_report": report,
            "num_test_samples": int(len(y_true)),
            "class_names": self.CLASS_NAMES,
        }

        logger.info("Accuracy:    %.4f", accuracy)
        logger.info("AUC-ROC:     %.4f", auc_roc)
        logger.info("Sensitivity: %.4f", sensitivity)
        logger.info("Specificity: %.4f", specificity)
        logger.info("F1 Score:    %.4f", f1)

        return metrics

    def _save_metrics(self, metrics: dict) -> None:
        serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
        serializable["classification_report"] = metrics.get("classification_report", {})

        metrics_path = Path(self.config.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(serializable, indent=4), encoding="utf-8")
        logger.info("Metrics saved at %s", metrics_path)

    def _save_confusion_matrix(self, y_true, y_pred, confusion_matrix) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.CLASS_NAMES,
                yticklabels=self.CLASS_NAMES,
            )
            plt.title("Confusion Matrix — Kidney CT Classification")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()

            cm_path = Path(self.config.confusion_matrix_path)
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(cm_path, dpi=150)
            plt.close()
            logger.info("Confusion matrix saved at %s", cm_path)

        except ImportError:
            logger.warning("matplotlib or seaborn not installed. Skipping confusion matrix plot.")

    def _save_roc_curve(self, y_true, y_pred_proba, roc_curve, auc) -> None:
        try:
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import label_binarize

            y_true_bin = label_binarize(y_true, classes=list(range(len(self.CLASS_NAMES))))
            plt.figure(figsize=(8, 6))

            for i, class_name in enumerate(self.CLASS_NAMES):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve — Kidney CT Classification")
            plt.legend(loc="lower right")
            plt.tight_layout()

            roc_path = Path(self.config.roc_curve_path)
            roc_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(roc_path, dpi=150)
            plt.close()
            logger.info("ROC curve saved at %s", roc_path)

        except ImportError:
            logger.warning("matplotlib not installed. Skipping ROC curve plot.")

    def _check_evaluation_gate(self, metrics: dict) -> None:
        auc_roc = metrics.get("auc_roc", 0.0)
        sensitivity = metrics.get("sensitivity", 0.0)

        if auc_roc < self.config.min_auc_threshold:
            raise ValueError(
                f"Evaluation gate FAILED: AUC-ROC {auc_roc:.4f} "
                f"< threshold {self.config.min_auc_threshold}. "
                "Model will NOT be deployed."
            )

        if sensitivity < self.config.min_sensitivity_threshold:
            raise ValueError(
                f"Evaluation gate FAILED: Sensitivity {sensitivity:.4f} "
                f"< threshold {self.config.min_sensitivity_threshold}. "
                "Model will NOT be deployed."
            )

        logger.info(
            "Evaluation gate PASSED — AUC: %.4f, Sensitivity: %.4f",
            auc_roc,
            sensitivity,
        )
