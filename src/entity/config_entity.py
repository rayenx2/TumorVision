from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    valid_classes: list
    min_images_per_class: int
    allowed_extensions: list
    status_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_dir: Path
    image_size: list
    train_split: float
    val_split: float
    test_split: float
    batch_size: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    image_size: list
    include_top: bool
    weights: str
    classes: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    base_model_path: Path
    trained_model_path: Path
    checkpoint_path: str
    csv_log_path: Path
    history_plot_path: Path
    class_weights_path: Path
    # Phase 1
    phase1_epochs: int
    phase1_learning_rate: float
    # Phase 2
    phase2_epochs: int
    phase2_learning_rate: float
    fine_tune_from_layer: int
    # Common
    batch_size: int
    dropout_rate: float
    dense_units: int
    # Callbacks
    early_stopping_patience: int
    reduce_lr_factor: float
    reduce_lr_patience: int
    reduce_lr_min_lr: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_path: Path
    confusion_matrix_path: Path
    roc_curve_path: Path
    min_auc_threshold: float
    min_sensitivity_threshold: float


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    mc_dropout_iterations: int
    uncertainty_threshold: float
    confidence_threshold: float
    gradcam_last_conv_layer: str
    gradcam_alpha: float


@dataclass(frozen=True)
class DriftDetectionConfig:
    root_dir: Path
    reference_data_path: Path
    current_data_path: Path
    drift_threshold: float


@dataclass(frozen=True)
class GradCAMConfig:
    """Configuration for Grad-CAM heatmap generation component."""

    # Output paths
    root_dir: Path

    # Model loading
    model_path: Path
    hf_repo_id: str
    hf_model_filename: str

    # Grad-CAM target layer (Phase 3 inspection confirmed)
    last_conv_layer_name: str

    # Visualization settings
    colormap: str
    alpha: float

    # Image preprocessing
    image_size: tuple

    # Class info
    class_names: list

    # Nested architecture access path (from inspection findings)
    # Required because EfficientNetB4 is double-nested in our model
    nested_wrapper_name: str  # "kidney_tumor_efficientnetb4"
    nested_backbone_name: str  # "efficientnetb4"


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for MC Dropout uncertainty estimation component."""

    # Output paths
    root_dir: Path

    # Model loading
    model_path: Path
    hf_repo_id: str
    hf_model_filename: str

    # MC Dropout settings
    mc_iterations: int  # Number of forward passes (20)
    uncertainty_threshold: float  # Above this, flag as uncertain (0.15)
    confidence_threshold: float  # Below this, flag as low confidence (0.70)

    # Image preprocessing
    image_size: tuple

    # Class info
    class_names: list

    # Model architecture (to access wrapper directly, skipping augmentation)
    nested_wrapper_name: str  # "kidney_tumor_efficientnetb4"


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for PDF report generation."""

    output_dir: Path
    disclaimer: str
    image_size: tuple
    class_names: list
    organization_name: str
    confidence_threshold: float
    uncertainty_threshold: float
