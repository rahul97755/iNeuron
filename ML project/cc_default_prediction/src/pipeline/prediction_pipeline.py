import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from src.utils import *
from src.utils1 import *
from src.components.data_transformation import DataModifier



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,data):
        try:
            #preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            #preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            #data_scaled=preprocessor.transform(features)

            pred=model.predict(data)
            prob=model.predict_proba(data)[: , 1]

            return prob

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 BILL_AMT3:float,
                 AGE:float,
                 PAY_2:float,
                 PAY_AMT4:float,
                 PAY_3:float,
                 PAY_0:float,
                 BILL_AMT1:float,
                 PAY_AMT3:float,
                 PAY_6:float,
                 PAY_4:float,
                 PAY_5:float,
                 PAY_AMT1:float,
                 LIMIT_BAL:float,
                 PAY_AMT6:float,
                 PAY_AMT2:float
                ):
        
        self.BILL_AMT3=BILL_AMT3
        self.AGE=AGE
        self.PAY_2=PAY_2
        self.PAY_AMT4=PAY_AMT4
        self.PAY_3=PAY_3
        self.PAY_0=PAY_0
        self.BILL_AMT1=BILL_AMT1
        self.PAY_AMT3=PAY_AMT3
        self.PAY_6=PAY_6
        self.PAY_4=PAY_4
        self.PAY_5=PAY_5
        self.PAY_AMT1=PAY_AMT1
        self.LIMIT_BAL=LIMIT_BAL
        self.PAY_AMT6=PAY_AMT6
        self.PAY_AMT2=PAY_AMT2

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'BILL_AMT3_woe':[self.BILL_AMT3],
                'AGE_woe':[self.AGE],
                'PAY_2_woe':[self.PAY_2],
                'PAY_AMT4_woe':[self.PAY_AMT4],
                'PAY_3_woe':[self.PAY_3],
                'PAY_0_woe':[self.PAY_0],
                'BILL_AMT1_woe':[self.BILL_AMT1],
                'PAY_AMT3_woe':[self.PAY_AMT3],
                'PAY_6_woe':[self.PAY_6],
                'PAY_4_woe':[self.PAY_4],
                'PAY_5_woe':[self.PAY_5],
                'PAY_AMT1_woe':[self.PAY_AMT1],
                'LIMIT_BAL_woe':[self.LIMIT_BAL],
                'PAY_AMT6_woe':[self.PAY_AMT6],
                'PAY_AMT2_woe':[self.PAY_AMT2]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)