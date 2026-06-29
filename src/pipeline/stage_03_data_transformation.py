import sys

from src.components.data_transformation import DataTransformation
from src.config.configuration import ConfigurationManager
from src.utils.exception import KidneyTumorException
from src.utils.logger import logger

STAGE_NAME = "Data Transformation"


class DataTransformationTrainingPipeline:
    """
    Pipeline stage responsible for preparing train, validation, and test splits.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.run()
        except Exception as e:
            raise KidneyTumorException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
