import os
import sys
from typing import Callable

from src.pipeline.stage_01_data_ingestion import STAGE_NAME as DATA_INGESTION_STAGE_NAME
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import STAGE_NAME as DATA_VALIDATION_STAGE_NAME
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import STAGE_NAME as DATA_TRANSFORMATION_STAGE_NAME
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_prepare_base_model import STAGE_NAME as PREPARE_BASE_MODEL_STAGE_NAME
from src.pipeline.stage_04_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.pipeline.stage_05_model_training import STAGE_NAME as MODEL_TRAINING_STAGE_NAME
from src.pipeline.stage_05_model_training import ModelTrainingPipeline
from src.pipeline.stage_06_model_evaluation import STAGE_NAME as MODEL_EVALUATION_STAGE_NAME
from src.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger

PipelineStage = tuple[str, Callable[[], None]]


def run_stage(stage_name: str, stage_callable: Callable[[], None]) -> None:
    """
    Run one pipeline stage with consistent logging and error handling.
    """
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", stage_name)
        stage_callable()
        logger.info(">>>>>>> stage %s completed <<<<<<<", stage_name)
    except Exception as e:
        logger.exception(">>>>>>> stage %s failed <<<<<<<", stage_name)
        raise KidneyTumorException(e, sys)


def get_pipeline_stages() -> list[PipelineStage]:
    """
    Return the training pipeline stages that are currently implemented.
    """
    stages = [
        (
            DATA_INGESTION_STAGE_NAME,
            DataIngestionTrainingPipeline().main,
        ),
        (
            DATA_VALIDATION_STAGE_NAME,
            DataValidationTrainingPipeline().main,
        ),
        (
            DATA_TRANSFORMATION_STAGE_NAME,
            DataTransformationTrainingPipeline().main,
        ),
        (
            PREPARE_BASE_MODEL_STAGE_NAME,
            PrepareBaseModelTrainingPipeline().main,
        ),
    ]

    if os.getenv("RUN_TRAINING", "false").lower() == "true":
        stages.extend(
            [
                (
                    MODEL_TRAINING_STAGE_NAME,
                    ModelTrainingPipeline().main,
                ),
                (
                    MODEL_EVALUATION_STAGE_NAME,
                    ModelEvaluationPipeline().main,
                ),
            ]
        )

    return stages


def main() -> None:
    """
    Project entry point for running the end-to-end training pipeline.
    """
    for stage_name, stage_callable in get_pipeline_stages():
        run_stage(stage_name, stage_callable)


if __name__ == "__main__":
    main()
