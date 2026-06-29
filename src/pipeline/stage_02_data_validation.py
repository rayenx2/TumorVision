import sys

from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager
from src.utils.exception import DataValidationError
from src.utils.logger import logger

STAGE_NAME = "Data Validation"


class DataValidationTrainingPipeline:
    """
    Pipeline stage responsible for validating the extracted dataset.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.run()
        except Exception as e:
            raise DataValidationError(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
