from src.logger import logging
from src.exception import CustomException
import os
import sys

from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass

class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion started")
            df = pd.read_csv("notebook/credit_data.csv")
            logging.info("dataset read as pandas dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("raw data saved")
            
            logging.info("splitting dataset into train and test sets")
            
            train_set, test_set = train_test_split(df,test_size=0.20, random_state=42,stratify=df["default"])
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info("ingestion of data is comppleted")
            
            return(
                self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e,sys)