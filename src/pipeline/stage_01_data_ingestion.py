import sys

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.utils.exception import DataIngestionError
from src.utils.logger import logger

STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    """
    Pipeline stage responsible for downloading and extracting the dataset.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.run()
        except Exception as e:
            raise DataIngestionError(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
