import sys
import os
import warnings
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.naive_bayes import GaussianNB
#import xgboost
from sklearn import metrics
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,precision_score,roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime
import time
import pickle
from src.exception import CustomException
from src.logger import logging


def EDA(df,label):
        tart_time = time.time()
        logging.info("try block started")
        try:
            df1=df.select_dtypes(include='number')
            logging.info("All the numeric varibale selected")
        
            plt.figure(figsize=(8,6))
            plt.rcParams["figure.autolayout"] = True
            warnings.filterwarnings("ignore")
            corr_val=df1.corr()
            sns.heatmap(data=corr_val,annot=True)
            logging.info("Coreletion ploted for all numeric data")
            
            for i in df1.columns:
                plt.figure(figsize=(6,4))
                plt.rcParams["figure.autolayout"] = True
                warnings.filterwarnings("ignore")
                sns.distplot(df1[i],
                            bins=10,
                            hist=True,
                            norm_hist=True)
                plt.xlabel('bins')
                print(f'The distribution of {i}')
                plt.show()
                logging.info(f"Histogram plotted for {i}")
                seconds = time.time() - start_time
                print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))
        except Exception as e :
            print(e)
            logging.critical(e,exc_info=True)
        

def split_data(df,label,split_percentage):
    '''df: Dataframe
       label: Target/Dependent predictor
       split_percentage: Size/percentage ratio of test data with respect to overall Population
       This function will split the population into trainning and testing data set with
       the predefined ratio
       return will be on order of x_train, x_test, y_train, y_test 

       '''
    try:
        logging.info(f" Data split operation started")
        y = df[[label]]
        x = df.loc[:, df.columns != label]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=split_percentage,random_state=1)

        print(f"shape of training set X:",x_train.shape)
        print(f"shape of training set y:",y_train.shape)
        print(f"ratio of label class of trainning:",y_train.value_counts(normalize=True))

        print(f"shape of testing set  x:",x_test.shape)
        print(f"shape of testing set  y:",y_test.shape)
        print(f"ratio of label class of test:",y_test.value_counts(normalize=True))
        logging.info(f" Data split with trinning and testing ")

        return x_train,x_test,y_train,y_test
    except Exception as e :
            print(e)
            logging.critical(e,exc_info=True)
def iv(x, y):
    try:
        def goodbad(df):
            #logging.info(f" Inside the goodbad module")
            names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
            logging.info(f"Good Bad distribution series has completed")
            return pd.Series(names)
            
    # iv calculation
        iv_total = pd.DataFrame({'x':x.astype('str'),'y':y}) \
          .fillna('missing') \
          .groupby('x') \
          .apply(goodbad) \
          .replace(0, 0.9) \
          .assign(
            DistrBad = lambda x: x.bad/sum(x.bad),
            DistrGood = lambda x: x.good/sum(x.good)
          ) \
          .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
          .iv.sum()
        logging.info(f"Information value returned ")
        return iv_total
        
    except Exception as e :
        logging.critical(e,exc_info=True)
        

def overview_data_classification(df,label):
    start_time = time.time()
    '''df: dataframe
       label: Dependent variable/class variable
       '''
    try:
        logging.info(f" Inside the try block of module overview_data")
        dat=df.drop(label,axis=1)
        logging.info(f" Inside the try block of module overview_data target variable dropped")
        overview = pd.DataFrame(dat.dtypes,columns=['dtypes'])
        logging.info(f" Inside the try block of module overview_data datatype calculated")
        overview = overview.reset_index()
        overview['Name'] = overview['index']
        logging.info(f" Inside the try block of module overview_data varibale name set as index")
        overview = overview[['Name','dtypes']]
        overview['Missing'] = dat.isnull().sum().values 
        logging.info(f" Inside the try block of module overview_data missing data calculated")
        overview['%Missing'] = dat.isnull().sum().values/dat.shape[0]
        overview['%Missing'] = overview['%Missing'].apply(lambda x: format(x, '.2%'))
        logging.info(f" Inside the try block of module overview_data missing % calculated")
        overview['Uniques'] = dat.nunique().values
        overview['%Unique'] = dat.nunique().values/dat.shape[0]
        overview['%Unique'] = overview['%Unique'].apply(lambda x: format(x, '.2%'))
        logging.info(f" Inside the try block of module overview_data unique value calculated")
#         overview['First Value'] = dat.loc[0].values
#         overview['Second Value'] = dat.loc[1].values
#         overview['Third Value'] = dat.loc[2].values
        for name in overview['Name'].value_counts().index:
            overview.loc[overview['Name'] == name, 'Entropy'] = round(stats.entropy(dat[name].value_counts(normalize=True), base=2),2)
        logging.info(f" Inside the try block of module overview_data Entropy calculated")
        for name in overview['Name'].value_counts().index:
            overview.loc[overview['Name'] == name, 'IV'] = round(iv(dat[name],dat.iloc[:,-1]),2)
        
        conditions = [
        (overview['IV'] <= 2.0),
        (overview['IV'] > 2.0) & (overview['IV'] <= 10.0),
        (overview['IV'] > 10.0) & (overview['IV'] <= 30.0),
        (overview['IV'] > 30.0) & (overview['IV'] <= 50.0),    
        (overview['IV'] > 50.0)
        ]
        values = ['not_useful', 'weak', 'Medium', 'Strong','Suspicious']
        overview['tier'] = np.select(conditions, values)
        logging.info(f" Inside the try block of module overview_data Iv calculated")
        seconds = time.time() - start_time
        print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))
        return overview
        
    except Exception as e :
        logging.critical(e,exc_info=True)
        
        
        


def compare_classification_model(train_X,train_Y,test_X,test_Y,classifiers_list,model_list):
    try:

        '''train_X: All independent variable of trainning set
        train_Y: Dependent/label variable of trainning set
        test_X: All independent variable of test set
        test_Y:  Dependent/label variable of test set
        classifier_list: The list of classifier
        model_list: list of models'''
        
        import pandas as pd
        from sklearn import svm
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier 
        from sklearn.ensemble import AdaBoostClassifier 
        from sklearn.naive_bayes import GaussianNB
        #import xgboost
        #from xgboost import XGBClassifier
        from sklearn import metrics
        from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,precision_score,roc_auc_score
        import warnings
        warnings.filterwarnings('ignore')
        
        
        abc=[]
        abc1=[]
        pre=[]
        pre1=[]
        rec=[]
        rec1=[]
        f=[]
        f1=[]
        auc=[]
        auc1=[]
        logging.info("All empty list declared")
        classifiers=classifiers_list
        models=model_list
        for i in models:
            model = i
            model.fit(train_X,train_Y)
            logging.info("models fit completed")
            prediction1=model.predict(train_X)
            logging.info("models predict completed")
            abc1.append(metrics.accuracy_score(prediction1,train_Y))
            logging.info("Accuracy score for train calculated")
            pre1.append(metrics.precision_score(train_Y,prediction1))
            logging.info("Precision score for train calculated")
            rec1.append(metrics.recall_score(train_Y,prediction1))
            f1.append(metrics.f1_score(train_Y,prediction1, average='binary'))
            logging.info("F1 score for train calculated")
            auc1.append(roc_auc_score(train_Y,prediction1))
            logging.info("ROC AUC score  for tarin calculated")
            
            prediction=model.predict(test_X)
            #predict_proba = model.predict_proba(test_X)[:,1]
            logging.info("models predict completed for test")
            abc.append(metrics.accuracy_score(prediction,test_Y))
            logging.info("Accuracy score for test calculated")
            pre.append(metrics.precision_score(test_Y,prediction))
            logging.info("Precision score for test calculated")
            rec.append(metrics.recall_score(test_Y,prediction))
            f.append(metrics.f1_score(test_Y,prediction, average='binary'))
            logging.info("F1 score for test calculated")
            auc.append(roc_auc_score(test_Y,prediction))
            logging.info("ROC AUC score test calculated")
            
        
        models_dataframe=pd.DataFrame(list(zip(abc1,abc)),
                                    columns=['Train_Accuracy','Test_Accuracy'],
                                    index=classifiers,)   
        models_dataframe=models_dataframe.reset_index()

        models_dataframe1=pd.DataFrame(list(zip(pre1,pre)),
                                    columns=['Train_Precision','Test_Precision'],
                                    index=classifiers)
        models_dataframe1=models_dataframe1.reset_index()

        models_dataframe2=pd.DataFrame(list(zip(rec1,rec)),
                                    columns=['Train_Recall','Test_Recall']
                                    ,index=classifiers)
        models_dataframe2=models_dataframe2.reset_index()

        models_dataframe3=pd.DataFrame(list(zip(f1,f))
                                    ,columns=['Train_Fscore','Test_Fscore'],
                                    index=classifiers)
        models_dataframe3=models_dataframe3.reset_index()
                    
        models_dataframe4=pd.DataFrame(list(zip(auc1,auc))
                                    ,columns=['Train_AUC','Test_AUC']
                                    ,index=classifiers)
        models_dataframe4=models_dataframe4.reset_index()
        logging.info("all data frame created")
        p1=pd.merge(models_dataframe,models_dataframe1,on='index',how='left')
        p2=pd.merge(p1,models_dataframe2,on='index',how='left')
        p3=pd.merge(p2,models_dataframe3,on='index',how='left')
                    
        p4=pd.merge(p3,models_dataframe4,on='index',how='left')
        logging.info("all data frame merged")
        return p4
    except Exception as e :
        logging.info(e,exc_info=True)



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def predict_result(array):
    if array>0.26:
        return 'Customer is risky'
    else:
        return 'Customer is not risky'    
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)



