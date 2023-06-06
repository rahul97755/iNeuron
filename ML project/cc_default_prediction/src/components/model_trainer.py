import pandas as pd

from datetime import datetime
import time
import sys
import src.utils
from src.utils import *
from src.utils1 import *
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, 
                             classification_report, f1_score, average_precision_score, precision_recall_fscore_support,
                            ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')
from src.components.data_ingestion import DataIngestion
from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_path,test_path):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train=train_df.iloc[:,1:]
            
            X_test=test_df.iloc[:,1:]

            Y_train=train_df.iloc[:,:1]

            Y_test=test_df.iloc[:,:1]
            
            lst_in=['BILL_AMT3_woe','AGE_woe','PAY_2_woe','PAY_AMT4_woe','PAY_3_woe','PAY_0_woe','BILL_AMT1_woe','PAY_AMT3_woe',
                     'PAY_6_woe','PAY_4_woe','PAY_5_woe','PAY_AMT1_woe','LIMIT_BAL_woe', 'PAY_AMT6_woe','PAY_AMT2_woe']
                

            lr = LogisticRegression()
            lr.fit(X_train[lst_in], Y_train)


            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=lr
            )


        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e,sys)    

