import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import *
from src.utils1 import *


class DataModifier:
    def __init__(self,dataframe):
        self.df=dataframe
        
        
    def change_column1(self,col_name1):
        self.df[col_name1] = self.df.apply(lambda row: self._apply_conditions_column1(row[col_name1]), axis=1)

        
    def _apply_conditions_column1(self,val):
        if val<=10000 :
            return -0.023295
        elif val>10001 and val<= 30000:
            return 0.123502
        elif val>30001 and val<= 50000:
            return 0.064145
        elif val>50001 and val <= 90000:
            return -0.032482
        else:
            return -0.140391
        
    def change_column2(self,col_name2):
        self.df[col_name2] = self.df.apply(lambda row: self._apply_conditions_column2(row[col_name2]), axis=1)

        
    def _apply_conditions_column2(self,val):
        if val<=26 :
            return 0.246734
        elif val>26 and val<= 29:
            return -0.092369
        elif val>29 and val<= 36:
            return -0.160764
        elif val>36 and val <= 46:
            return -0.016469
        else:
            return 0.172317
        
    def change_column3(self,col_name3):
        self.df[col_name3] = self.df.apply(lambda row: self._apply_conditions_column3(row[col_name3]), axis=1)

        
    def _apply_conditions_column3(self,val):
        if val<=1 :
            return -0.405003
        elif val>1 and val<= 2:
            return -0.239627
        else:
            return 1.501653
        
    def change_column4(self,col_name4):
        self.df[col_name4] = self.df.apply(lambda row: self._apply_conditions_column4(row[col_name4]), axis=1)

        
    def _apply_conditions_column4(self,val):
        if val<=500 :
            return 0.332804
        elif val>500 and val<= 2000:
            return -0.094988
        elif val>2000 and val<= 4500:
            return -0.163460
        elif val>4500 and val<= 15500:
            return -0.411755
        else:
            return -0.734958 
        
    def change_column5(self,col_name5):
        self.df[col_name5] = self.df.apply(lambda row: self._apply_conditions_column5(row[col_name5]), axis=1)

        
    def _apply_conditions_column5(self,val):
        if val<=-2 :
            return -0.222077
        elif val== -1:
            return -0.430029
        elif val==0:
            return -0.295297
        elif val==1:
            return -0.160075
        elif val==2:
            return -1.321027
        else:
            return -1.642829
        
    def change_column6(self,col_name6):
        self.df[col_name6] = self.df.apply(lambda row: self._apply_conditions_column6(row[col_name6]), axis=1)

        
    def _apply_conditions_column6(self,val):
        if val<=-2 :
            return -0.622137
        elif val== -1:
            return -0.342753
        elif val==0:
            return -0.659061
        elif val==1:
            return 0.593072
        elif val==2:
            return 2.065423
        else:
            return 2.199295 
        
    def change_column7(self,col_name7):
        self.df[col_name7] = self.df.apply(lambda row: self._apply_conditions_column7(row[col_name7]), axis=1)

        
    def _apply_conditions_column7(self,val):
        if val<=100 :
            return 0.147990
        elif val>100 and val<=500:
            return 0.108500
        elif val>500 and val<= 1000:
            return 0.001812
        elif val>1000 and val<=3000:
            return 0.003922
        elif val>3000 and val<=5000:
            return -0.278409
        elif val>5000 and val<=10000:
            return -0.016078
        elif val>10000 and val<=20000:
            return 0.145377
        elif val>20000 and val<=30000:
            return 0.081216
        elif val>30000 and val<=40000:
            return 0.045896
        elif val>40000 and val<=50000:
            return 0.016974
        elif val>50000 and val<=70000:
            return -0.077195
        elif val>70000 and val<=100000:
            return -0.049646
        elif val>100000 and val<=200000:
            return -0.180133
        else: 
            return -0.109648
        
        
    def change_column8(self,col_name8):
        self.df[col_name8] = self.df.apply(lambda row: self._apply_conditions_column8(row[col_name8]), axis=1)

        
    def _apply_conditions_column8(self,val):
        if val<=500 :
            return 0.411208
        elif val>500 and val<= 3000:
            return 0.027638
        elif val>3000 and val<= 5000:
            return -0.188312
        elif val>5000 and val<= 12500:
            return -0.420977
        else:
            return -0.801929
        
    def change_column9(self,col_name9):
        self.df[col_name9] = self.df.apply(lambda row: self._apply_conditions_column9(row[col_name9]), axis=1)

        
    def _apply_conditions_column9(self,val):
        if val<=-2 :
            return -0.125017
        elif val== -1:
            return -0.327890
        elif val==0:
            return -0.201427
        elif val==1:
            return -0.201427
        elif val==2:
            return 1.284757
        elif val==3:
            return 1.839755
        elif val==4:
            return 1.802341
        elif val==5:
            return 1.412876
        elif val==6:
            return 2.288345
        elif val==7:
            return 2.816870
        else:
            return 2.057233
        
    def change_column10(self,col_name10):
        self.df[col_name10] = self.df.apply(lambda row: self._apply_conditions_column10(row[col_name10]), axis=1)

        
    def _apply_conditions_column10(self,val):
        if val<=-2 :
            return -0.175145
        elif val== -1:
            return -0.407307
        elif val==0:
            return -0.235542
        elif val==1:
            return 1.258687
        elif val==2:
            return 1.351822
        else:
            return 1.854506
        
    def change_column11(self,col_name11):
        self.df[col_name11] = self.df.apply(lambda row: self._apply_conditions_column11(row[col_name11]), axis=1)

        
    def _apply_conditions_column11(self,val):
        if val<=-2 :
            return -0.147246
        elif val== -1:
            return -0.385158
        elif val==0:
            return -0.200910
        elif val==1:
            return -0.200910
        elif val==2:
            return 1.426636
        else:
            return 1.925632
        
    def change_column12(self,col_name12):
        self.df[col_name12] = self.df.apply(lambda row: self._apply_conditions_column12(row[col_name12]), axis=1)

        
    def _apply_conditions_column12(self,val):
        if val<=500 :
            return 0.552881
        elif val>500 and val<= 5000:
            return -0.017578
        elif val>5000 and val<= 17500:
            return -0.485817
        else:
            return -0.978401
        
    def change_column13(self,col_name13):
        self.df[col_name13] = self.df.apply(lambda row: self._apply_conditions_column13(row[col_name13]), axis=1)

        
    def _apply_conditions_column13(self,val):
        if val<=40000 :
            return 0.676765
        elif val>40000 and val<=140000:
            return 0.197458
        elif val>140000 and val<= 380000:
            return -0.338483
        else:
            return -0.744021
            
    def change_column14(self,col_name14):
        self.df[col_name14] = self.df.apply(lambda row: self._apply_conditions_column14(row[col_name14]), axis=1)

        
    def _apply_conditions_column14(self,val):
        if val<=1000:
            return 0.273671
        elif val>1000 and val<=3000:
            return 0.019319
        elif val>3000 and val<= 4000:
            return -0.165069
        elif val>4000 and val<= 10000:
            return -0.353411
        else:
            return -0.693093
        
    def change_column15(self,col_name15):
        self.df[col_name15] = self.df.apply(lambda row: self._apply_conditions_column15(row[col_name15]), axis=1)

        
    def _apply_conditions_column15(self,val):
        if val<=500:
            return 0.484679
        elif val>500 and val<=2000:
            return 0.072079
        elif val>2000 and val<= 5000:
            return -0.053113
        elif val>5000 and val<= 17500:
            return -0.445220
        else:
            return -1.177714
             