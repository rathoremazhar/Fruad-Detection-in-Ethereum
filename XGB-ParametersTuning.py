###################################################################
# This module identifies the best hyperparameter values for XGBoost
# Best Hyperparamters values are found by applying different configurations on full training dataset  
# after performing the preprocessing functions such as,remove duplicate rows, Normalization using power transform, 
# , resampling for ballancing the data using SMOTE, on whole training dataset.
#Threading is applied for efficienct processing
# best values of hyperparamters (of XGBoost) are found individually, based on 1) accuracy, 3) Precision, , and 3) recall. 
# accuracy results are disolayed on whole training datasets on best params. 
###################################################################



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer_names, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import threading

def main():
    df = pd.read_csv('AllData.csv', index_col=0)
    #print("Class Distribution :",df['flag'].value_counts())
    df=df.drop_duplicates()
    #print("Class Distribution :",df['flag'].value_counts())

    
    #Spliting class-attribute and other atributes
    #df.info()
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    
    norm = PowerTransformer()
    norm_train_f = norm.fit_transform(X)

    #############re-sampling
    oversample = SMOTE()
    norm_train_f, y = oversample.fit_resample(norm_train_f, y)

    



 # ###########Hyperparameters tuning for XGB Classifier
    print("XGboos results after tunning\n")
    xgb_c = xgb.XGBClassifier(random_state=42)
    params_grid = {'learning_rate':[0.01, 0.1, 0.5],
              'n_estimators':[100, 150, 200],
              'subsample':[0.1, 0.3, 0.5, 0.9],
               'max_depth':[2,3,4],
               'colsample_bytree':[0.3,0.5,0.7]}
    
    #####Apply threading to find best hyper paarmeters by applying all hyper-params options. 
    x1=threading.Thread(target=thread_ParamTunningAccuracy, args=(xgb_c, params_grid, norm_train_f, y))
    x1.start()
    
    
    x2=threading.Thread(target=thread_ParamTunningRecall, args=(xgb_c, params_grid, norm_train_f, y))
    x2.start()

    
    x3=threading.Thread(target=thread_ParamTunningPrecision, args=(xgb_c, params_grid, norm_train_f, y))
    x3.start()
    x1.join()
    x2.join()
    x3.join()

##################Threads########
def thread_ParamTunningAccuracy(xgb_c, _params_grid, norm_train_f, y):
    print("Accuracy thread started")
    grid = GridSearchCV(estimator=xgb_c, param_grid=_params_grid, scoring='accuracy', cv = 10, verbose = 0)
    grid.fit(norm_train_f, y)
    print(f'Best params-accuracy found for XGBoost are: {grid.best_params_}')
    print(f'Best accuracy obtained by the best params: {grid.best_score_}')

    preds_best_xgb = grid.best_estimator_.predict(norm_train_f)
    print("Based on Accuracy Results\n\n")
    print(classification_report(y, preds_best_xgb))
    print(confusion_matrix(y, preds_best_xgb))


def thread_ParamTunningRecall(xgb_c, _params_grid, norm_train_f, y):
    print("Recall thread started")
    grid = GridSearchCV(estimator=xgb_c, param_grid=_params_grid, scoring='recall', cv = 10, verbose = 0)
    grid.fit(norm_train_f, y)
    print(f'Best params-Recall found for XGBoost are: {grid.best_params_}')
    print(f'Best recall obtained by the best params: {grid.best_score_}')

    preds_best_xgb = grid.best_estimator_.predict(norm_train_f)
    print("Based on Recall Results\n\n")
    print(classification_report(y, preds_best_xgb))
    print(confusion_matrix(y, preds_best_xgb))

def thread_ParamTunningPrecision(xgb_c, _params_grid, norm_train_f, y):
    print("Precision thread started")
    grid = GridSearchCV(estimator=xgb_c, param_grid=_params_grid, scoring='precision', cv = 10, verbose = 0)
    grid.fit(norm_train_f, y)
    print(f'Best params-Precision found for XGBoost are: {grid.best_params_}')
    print(f'Best precision obtained by the best params: {grid.best_score_}')

    preds_best_xgb = grid.best_estimator_.predict(norm_train_f)
    print("Based on Precision Results\n\n")
    print(classification_report(y, preds_best_xgb))
    print(confusion_matrix(y, preds_best_xgb))

if __name__ == "__main__":
    main()


