import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_file_path: str = os.path.join(
        os.getcwd(), "artifacts", "raw_data.csv")
    train_data_file_path: str = os.path.join(
        os.getcwd(), "artifacts", "train_data.csv")
    test_data_file_path: str = os.path.join(
        os.getcwd(), "artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Data Ingestion Started")
            df = pd.read_csv(
                r"C:\Users\patil\Downloads\dataAnalytics\MLOPS\Project01_data.csv")
            os.makedirs(os.path.dirname(
                self.data_ingestion_config.raw_data_file_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_file_path,
                      index=False, header=True)
            logging.info("Raw data saved successfully")
            logging.info("test train split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=217)
            os.makedirs(os.path.dirname(
                self.data_ingestion_config.train_data_file_path), exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.train_data_file_path, index=False, header=True)
            os.makedirs(os.path.dirname(
                self.data_ingestion_config.test_data_file_path), exist_ok=True)
            test_set.to_csv(
                self.data_ingestion_config.test_data_file_path, index=False, header=True)
            logging.info("Data Ingestion completed")
            return (
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
