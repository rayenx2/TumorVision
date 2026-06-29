import sys
from pathlib import Path

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    DriftDetectionConfig,
    GradCAMConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
    PrepareBaseModelConfig,
    ReportConfig,
    UncertaintyConfig,
)
from src.utils.common import create_directories, read_yaml
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger


class ConfigurationManager:
    """
    Central configuration manager.
    Reads config.yaml and params.yaml once and provides
    typed config objects for each pipeline stage.

    Usage:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            create_directories([self.config.artifacts_root])
            logger.info("ConfigurationManager initialised successfully.")
        except Exception as e:
            raise KidneyTumorException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            kaggle_dataset=config.kaggle_dataset,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            valid_classes=config.valid_classes,
            min_images_per_class=config.min_images_per_class,
            allowed_extensions=config.allowed_extensions,
            status_file=Path(config.status_file),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            image_size=config.image_size,
            train_split=config.train_split,
            val_split=config.val_split,
            test_split=config.test_split,
            batch_size=config.batch_size,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            image_size=config.image_size,
            include_top=config.include_top,
            weights=config.weights,
            classes=config.classes,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            trained_model_path=Path(config.trained_model_path),
            checkpoint_path=config.checkpoint_path,
            csv_log_path=Path(config.csv_log_path),
            history_plot_path=Path(config.history_plot_path),
            class_weights_path=Path(config.class_weights_path),
            phase1_epochs=params.TRAINING.phase1_epochs,
            phase1_learning_rate=params.TRAINING.phase1_learning_rate,
            phase2_epochs=params.TRAINING.phase2_epochs,
            phase2_learning_rate=params.TRAINING.phase2_learning_rate,
            fine_tune_from_layer=params.TRAINING.fine_tune_from_layer,
            batch_size=params.TRAINING.batch_size,
            dropout_rate=params.TRAINING.dropout_rate,
            dense_units=params.TRAINING.dense_units,
            early_stopping_patience=params.CALLBACKS.early_stopping_patience,
            reduce_lr_factor=params.CALLBACKS.reduce_lr_factor,
            reduce_lr_patience=params.CALLBACKS.reduce_lr_patience,
            reduce_lr_min_lr=params.CALLBACKS.reduce_lr_min_lr,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            metrics_path=Path(config.metrics_path),
            confusion_matrix_path=Path(config.confusion_matrix_path),
            roc_curve_path=Path(config.roc_curve_path),
            min_auc_threshold=params.EVALUATION.min_auc_threshold,
            min_sensitivity_threshold=params.EVALUATION.min_sensitivity_threshold,
        )

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        gradcam_config = self.config.gradcam

        return PredictionConfig(
            model_path=Path(config.model_path),
            mc_dropout_iterations=config.mc_dropout_iterations,
            uncertainty_threshold=config.uncertainty_threshold,
            confidence_threshold=config.confidence_threshold,
            gradcam_last_conv_layer=gradcam_config.last_conv_layer_name,
            gradcam_alpha=gradcam_config.alpha,
        )

    def get_drift_detection_config(self) -> DriftDetectionConfig:
        config = self.config.drift_detection
        create_directories([config.root_dir])

        return DriftDetectionConfig(
            root_dir=Path(config.root_dir),
            reference_data_path=Path(config.reference_data_path),
            current_data_path=Path(config.current_data_path),
            drift_threshold=config.drift_threshold,
        )

    def get_gradcam_config(self) -> GradCAMConfig:
        """Build configuration for Grad-CAM component.
        Combines:
        config.gradcam section (visualization settings)
        - config.prediction section (model path)
        - config.model_hub section (Hugging Face info)
        - params.IMAGE section (image dimensions)
        - params.CLASSES (class labels)
        """
        config = self.config.gradcam
        prediction_config = self.config.prediction
        model_hub_config = self.config.model_hub
        params = self.params

        # Ensure output directory exists
        create_directories([config.root_dir])
        gradcam_config = GradCAMConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(prediction_config.model_path),
            hf_repo_id=model_hub_config.repo_id,
            hf_model_filename=model_hub_config.model_filename,
            last_conv_layer_name=config.last_conv_layer_name,
            colormap=config.colormap,
            alpha=config.alpha,
            image_size=(params.IMAGE.size, params.IMAGE.size),
            class_names=params.CLASSES,
            nested_wrapper_name="kidney_tumor_efficientnetb4",
            nested_backbone_name="efficientnetb4",
        )

        return gradcam_config

    def get_uncertainty_config(self) -> UncertaintyConfig:
        """Build configuration for MC Dropout uncertainty component."""
        config = self.config.prediction
        model_hub_config = self.config.model_hub
        model_hub_config = self.config.model_hub
        params = self.params

        # Output directory for uncertainty artifacts
        output_dir = Path("artifacts/uncertainty")
        create_directories([str(output_dir)])

        uncertainty_config = UncertaintyConfig(
            root_dir=output_dir,
            model_path=Path(config.model_path),
            hf_repo_id=model_hub_config.repo_id,
            hf_model_filename=model_hub_config.model_filename,
            mc_iterations=config.mc_dropout_iterations,
            uncertainty_threshold=config.uncertainty_threshold,
            confidence_threshold=config.confidence_threshold,
            image_size=(params.IMAGE.size, params.IMAGE.size),
            class_names=params.CLASSES,
            nested_wrapper_name="kidney_tumor_efficientnetb4",
        )

        return uncertainty_config

    def get_report_config(self) -> ReportConfig:
        """Build configuration for report generation component."""
        config = self.config.report
        prediction_config = self.config.prediction
        params = self.params

        create_directories([config.output_dir])

        return ReportConfig(
            output_dir=Path(config.output_dir),
            disclaimer=config.disclaimer,
            image_size=(params.IMAGE.size, params.IMAGE.size),
            class_names=params.CLASSES,
            organization_name="Kidney CT Analysis System",
            confidence_threshold=prediction_config.confidence_threshold,
            uncertainty_threshold=prediction_config.uncertainty_threshold,
        )
