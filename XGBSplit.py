###################################################################
# This module compute the performance results by spliting the data into training and test sets.
# best hyperparameter values for XGBoost are applied individually, based on 1) accuracy, 3) Precision, , and 3) recall. 
# accuracy results are displayed on whole test datasets on best params. 
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
import time  

df = pd.read_csv('AllData.csv', index_col=0) 
df=df.drop_duplicates()
y = df.iloc[:, 0]
X = df.iloc[:, 1:]
    
norm = PowerTransformer()
norm_train_f = norm.fit_transform(X)

   
#############re-sampling
oversample = SMOTE()
norm_train_f, y = oversample.fit_resample(norm_train_f, y)

X_train, X_test, y_train, y_test = train_test_split(norm_train_f, y, test_size = 0.2, random_state = 123)

print("\n\nXGboost Results on best parameters based on accuracy")
xgb_c = xgb.XGBClassifier(colsample_bytree=0.7, learning_rate=0.5, max_depth=4, n_estimators= 200, subsample=0.9, random_state=42)
st = time.time()
xgb_c.fit(X_train, y_train)
et = time.time()
elapsed_time = et - st
print('Model Built time:', elapsed_time, 'seconds')
st = time.time()
preds_xgb = xgb_c.predict(X_test)
et = time.time()
elapsed_time = et - st
print('Model tesy time:', elapsed_time, 'seconds')
print(classification_report(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))

print("\n\nXGboost Results on best parameters based on Recall")
xgb_c = xgb.XGBClassifier(colsample_bytree=0.7, learning_rate=0.5, max_depth=4, n_estimators= 200, subsample=0.9, random_state=42)
st = time.time()
xgb_c.fit(X_train, y_train)
et = time.time()
elapsed_time = et - st
print('Model Built time:', elapsed_time, 'seconds')
st = time.time()
preds_xgb = xgb_c.predict(X_test)
et = time.time()
elapsed_time = et - st
print('Model tesy time:', elapsed_time, 'seconds')
print(classification_report(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))

print("\n\nXGboost Results on best parameters based on Precision")
xgb_c = xgb.XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators= 200, subsample=0.5, random_state=42)
st = time.time()
xgb_c.fit(X_train, y_train)
et = time.time()
elapsed_time = et - st
print('Model Built time:', elapsed_time, 'seconds')
st = time.time()
preds_xgb = xgb_c.predict(X_test)
et = time.time()
elapsed_time = et - st
print('Model tesy time:', elapsed_time, 'seconds')
print(classification_report(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))