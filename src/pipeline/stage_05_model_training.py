import sys

from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    """
    Pipeline stage responsible for two-phase model training.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            data_transformation_config = config.get_data_transformation_config()

            model_trainer = ModelTrainer(
                config=model_trainer_config,
                data_transformation_config=data_transformation_config,
            )
            model_trainer.run()
        except Exception as e:
            raise KidneyTumorException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
