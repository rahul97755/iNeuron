
import os
import sys
from src.logger import logging
from src.exception import CustomException
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
#from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data, test_data,train_df_flited_woe,test_df_flited_woe=obj.initiate_data_ingestion()
    #data_transformation = DataTransformation()
    #train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_df_flited_woe,test_df_flited_woe)




