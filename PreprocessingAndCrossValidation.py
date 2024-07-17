###################################################################
#This module compute the accuracy using cross validation after 
# performing the preprocessing functions such as,remove duplicate rows, Normalization using power transform, 
# , resampling for ballancing the data using SMOTE.
# Classifiers applied are CART, Random forest with CART, LGBM, XGBoost. 
# #################################################################### 


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

def main():
    #df = pd.read_csv('abc.csv', index_col=0)
    df = pd.read_csv('AllData.csv', index_col=0) ####inlcude your dataset file in CSv format
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


    print("CART Results")
    dtm = tree.DecisionTreeClassifier(random_state=42)
    results=cross_validation(dtm, norm_train_f, y, 10)
    print("Accuracy : ", results['test_accuracy'].mean(), 
            "\nPrecision: ", results['test_precision'].mean(),
            "\nRecall:", results['test_recall'].mean(),
             "\nF1 Score:", results['test_f1'].mean()
            )


    print("\n\nRandom Forest Results")
    RF = RandomForestClassifier(random_state=42)
    results=cross_validation(RF, norm_train_f, y, 10)
    print("Accuracy : ", results['test_accuracy'].mean(), 
            "\nPrecision: ", results['test_precision'].mean(),
            "\nRecall:", results['test_recall'].mean(),
             "\nF1 Score:", results['test_f1'].mean()
            )

    print("\n\nLGBM Results")
    lgbm=lgb.LGBMClassifier(random_state=42)
    results=cross_validation(lgbm, norm_train_f, y, 10)
    print("Accuracy : ", results['test_accuracy'].mean(), 
            "\nPrecision: ", results['test_precision'].mean(),
            "\nRecall:", results['test_recall'].mean(),
             "\nF1 Score:", results['test_f1'].mean()
            )

    print("\n\nXGboost Results")
    xgb_c = xgb.XGBClassifier(random_state=42)
    results=cross_validation(xgb_c, norm_train_f, y, 10)
    print("Accuracy : ", results['test_accuracy'].mean(), 
            "\nPrecision: ", results['test_precision'].mean(),
            "\nRecall:", results['test_recall'].mean(),
             "\nF1 Score:", results['test_f1'].mean()
            )
    



def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    return results;  



if __name__ == "__main__":
    main()


