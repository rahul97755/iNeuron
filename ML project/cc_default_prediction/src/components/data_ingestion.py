import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import *
from src.utils1 import *

#from src.components.data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

    train_df_flited_woe_path:str=os.path.join('artifacts','train_df_flited_woe.csv')
    test_df_flited_woe_path:str=os.path.join('artifacts','test_df_flited_woe.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','UCI_Credit_Card.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            df_flited=var_filter(df,'target')

            logging.info('Train test split')
            train_set,test_set=train_test_split(df_flited,test_size=0.30,random_state=42)
            
            
            bins =woebin(df_flited, y="target")

            breaks_adj = {
                            'AGE': [25, 30, 35,40,45 ],
                            'BILL_AMT1':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'BILL_AMT2':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'BILL_AMT3':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'BILL_AMT4':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'BILL_AMT5':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'BILL_AMT6':[100,500,1000,3000,5000,10000,20000,30000,40000,50000,70000,100000,200000],
                            'PAY_0':[-2,-1,0,1,2],
                            'PAY_2':[-2,-1,0,1,2],
                            'PAY_3':[-2,-1,0,1,2],
                            
                            'PAY_4':[-2,-1,0,1,2],
                            'PAY_5':[-2,-1,0,1,2],
                            'PAY_6':[-2,-1,0,1,2,3,4,5,6,7,8]
                        }
            bins_adj = woebin(df_flited, y="target", breaks_list=breaks_adj)

            train_df_flited_woe = woebin_ply(train_set, bins_adj)
            test_df_flited_woe = woebin_ply(test_set, bins_adj)


            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            
            train_df_flited_woe.to_csv(self.ingestion_config.train_df_flited_woe_path,index=False,header=True)
            test_df_flited_woe.to_csv(self.ingestion_config.test_df_flited_woe_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_df_flited_woe_path,
                self.ingestion_config.test_df_flited_woe_path,
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)


## run data ingestion

#if __name__=='__main__':
 #  obj=DataIngestion()
  # train_data, test_data,train_df_flited_woe,test_df_flited_woe=obj.initiate_data_ingestion()
