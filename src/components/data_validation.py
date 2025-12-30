from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    train_data_path: str
    test_data_path: str

class DataValidation:
    def __init__(self, train_data: str, test_data: str):
        self.config = DataValidationConfig(train_data, test_data)

    def initiate_data_validation(self):
        try:
            # Read train and test data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info("Read train and test data for validation")

            # Check missing values and log detailed info
            missing_train = train_df.isnull().sum()
            missing_test = test_df.isnull().sum()

            if missing_train.any():
                logging.warning(f"Missing values found in training data:\n{missing_train[missing_train > 0]}")
            else:
                logging.info("No missing values in training data")

            if missing_test.any():
                logging.warning(f"Missing values found in testing data:\n{missing_test[missing_test > 0]}")
            else:
                logging.info("No missing values in testing data")

            # Check column consistency
            if set(train_df.columns) != set(test_df.columns):
                raise CustomException("Training and testing data have different columns", sys)
            logging.info("Training and testing data have consistent columns")

            # Check datatypes per column
            for col in train_df.columns:
                if train_df[col].dtype != test_df[col].dtype:
                    logging.warning(
                        f"Column '{col}' has different dtype: train={train_df[col].dtype}, test={test_df[col].dtype}"
                    )
            logging.info("Column data types checked")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)
