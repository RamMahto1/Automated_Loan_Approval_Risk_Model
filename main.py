from src.logger import logging
import os
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion


## Step 1: Data Ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

logging.info(f"Train data path: {train_data_path}")
logging.info(f"Test data path: {test_data_path}")



# try:
#     result = 0/1
#     logging.info(f"zero divided by one is {result}")
# except Exception as e:
#     raise CustomException(e, sys)


# logging.info("Starting main execution.")
