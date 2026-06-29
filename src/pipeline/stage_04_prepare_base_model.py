import sys

from src.components.prepare_base_model import PrepareBaseModel
from src.config.configuration import ConfigurationManager
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelTrainingPipeline:
    """
    Pipeline stage responsible for preparing the EfficientNetB4 base model.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.run()
        except Exception as e:
            raise KidneyTumorException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
