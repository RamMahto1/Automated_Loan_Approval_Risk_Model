from src.logger import logging
import os
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer

## Step 1: Data Ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

logging.info(f"Train data path: {train_data_path}")
logging.info(f"Test data path: {test_data_path}")


## step 2: Data Transformation
data_transformation = DataTransformation()
train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data_path, test_data_path)

logging.info("Data Transformation Completed")
logging.info(f"train array shape:{train_arr.shape}")
logging.info(f"test array shape:{test_arr.shape}")

## Step 3: Data validation
data_validation = DataValidation(train_data_path, test_data_path)
train_df, test_df = data_validation.initiate_data_validation()

logging.info("Data Validation Completed")
logging.info(f"train dataframe shape:{train_df.shape}")
logging.info(f"test dataframe shape:{test_df.shape}")


## Step 4: Model Trainer
model_trainer = ModelTrainer()
best_model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
logging.info(f"Best model saved at: {best_model_path}")






# try:
#     result = 0/1
#     logging.info(f"zero divided by one is {result}")
# except Exception as e:
#     raise CustomException(e, sys)


# logging.info("Starting main execution.")
