import sys

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.utils.exception import ModelEvaluationError
from src.utils.logger import logger

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    """
    Pipeline stage responsible for evaluating the trained model on the test split.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            data_transformation_config = config.get_data_transformation_config()

            model_evaluation = ModelEvaluation(
                config=model_evaluation_config,
                data_transformation_config=data_transformation_config,
            )
            model_evaluation.run()
        except Exception as e:
            raise ModelEvaluationError(e, sys)


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception("Stage %s failed.", STAGE_NAME)
        raise e
