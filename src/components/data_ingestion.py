
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataPreprocessing
from src.components.data_transformation import DataPreprocessingConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTraining

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\EURUSD_D1 (1).csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info('Saved the raw data')

            return df
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    data = obj.initiate_data_ingestion()

    data_preprocessing = DataPreprocessing(
        file_path='artifacts/data.csv')
    preprocessing_result = data_preprocessing.preprocess()

    logging.info('Data preprocessing completed')
    modeltrainer = ModelTraining(
        df=preprocessing_result['df'],
        predictors=preprocessing_result['predictors'],
        target=preprocessing_result['target']
    )
    modeltrainer.train_model()
    logging.info('Model training completed')
