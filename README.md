# Fruad-Detection-in-Ethereum
Code to detect frauds in Ethereum cryptocurency. Work is publsihed at IEEE Globecome 2023. Please cite as follows. 

--Rathore, M. Mazhar, et al. "Detection of Fraudulent Entities in Ethereum Cryptocurrency: A Boosting-based Machine Learning Approach." GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023.



###All codes are wirtten in python 3.8.18
###Datasets are not given. you can use your own datasets  
#################### Pre-requisite installtions#################
#numpy
#pandas
#matplotlib
#seaborn
#sklearn
#imblearn
#xgboost
#lightgbm
#pickle
#############################################################



##################### Running the Python code files : python 3.8.18######################
#python3 PreprocessingAndCrossValidation.py
#python3 XGB-ParametersTuning.py
#python3 XGBSplit.py
#python3 XGB-TainTestSeparateData.py
################################################################


###########################PreprocessingAndCrossValidation.py########################################
#This module compute the accuracy using cross validation after 
#performing the preprocessing functions such as,remove duplicate rows, Normalization using power transform, 
#, resampling for ballancing the data using SMOTE.
#Classifiers applied are CART, Random forest with CART, LGBM, XGBoost. 
# #################################################################### 

##########################XGB-ParametersTuning.py#########################################
#This module identifies the best hyperparameter values for XGBoost
#Best Hyperparamters values are found by applying different configurations on full training dataset  
#after performing the preprocessing functions such as,remove duplicate rows, Normalization using power transform, 
#, resampling for ballancing the data using SMOTE, on whole training dataset.
#Threading is applied for efficienct processing
#best values of hyperparamters (of XGBoost) are found individually, based on 1) accuracy, 3) Precision, , and 3) recall. 
#accuracy results are disolayed on whole training datasets on best params.  
###################################################################


################################XGBSplit.py###################################
#This module computes the performance results by spliting the data into training and test sets.
#best hyperparameter values for XGBoost are applied individually, based on 1) accuracy, 3) Precision, , and 3) recall. 
#accuracy results are displayed on whole test datasets using best params.  
##################################################################



################################XGB-TainTestSeparateData.py###################################
#This module compute the performance results by spliting the data into training and test sets.
#best hyperparameter values for XGBoost are applied individually, based on 1) accuracy, 3) Precision, , and 3) recall. 
#accuracy results are displayed on whole test datasets on best params. 
###################################################################
