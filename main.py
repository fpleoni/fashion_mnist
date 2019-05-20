# Import libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Lead train and test datasets
fashion_mnist_train = pd.read_csv("fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("fashion-mnist_test.csv")

# Check training data shape
fashion_mnist_train.shape

# Check training data shape
fashion_mnist_test.shape

fashion_mnist_train.head()
# Separate target and features for training set
target_train = fashion_mnist_train["label"]
features_train = fashion_mnist_train.drop(["label"], axis = 1)

# Sanity Check
features_train.head()

# Print shapes of target and features
print(target_train.shape)
print(features_train.shape)

# Separate target and features for training set
target_test = fashion_mnist_test["label"]
features_test = fashion_mnist_test.drop(["label"], axis = 1)
# Print shapes of target and features
print(target_test.shape)
print(features_test.shape)

# Use sklearn StandardScaler to scale pixel values
from sklearn.preprocessing import StandardScaler
# Create scale object
scaler = StandardScaler()
# Fit scaler to training data only
scaler_fit = scaler.fit(features_train)
# Transform both train and test data with the trained scaler
X_train = scaler_fit.transform(features_train)
X_test = scaler_fit.transform(features_test)

import xgboost as xgb
# Create XGB Classifier object
xgb_clf = xgb.XGBClassifier(objective = "multi:softmax", tree_method = "gpu_exact",
                            predictor = "gpu_predictor", verbosity = True)
# Fit model
xgb_model = xgb_clf.fit(X_train, target_train)
# Predictions
y_train_preds = xgb_model.predict(X_train)
y_test_preds = xgb_model.predict(X_test)
# Print F1 scores and Accuracy
print("Training F1 Micro Average: ", f1_score(target_train, y_train_preds, average = "micro"))
print("Test F1 Micro Average: ", f1_score(target_test, y_test_preds, average = "micro"))
print("Test Accuracy: ", accuracy_score(target_test, y_test_preds))

import xgboost as xgb

# Create XGB Classifier object
xgb_clf = xgb.XGBClassifier(tree_method = "exact", predictor = "cpu_predictor", verbosity = True,
                            objective = "multi:softmax")

# Create parameter grid
parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 250, 500, 1000]}

from sklearn.model_selection import RandomizedSearchCV
# Create RandomizedSearchCV Object
xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions = parameters, scoring = "f1_micro",
                             cv = 10, verbose = 3, random_state = 40 )

# Fit the model
model_xgboost = xgb_rscv.fit(X_train, target_train)

# Model best estimators
print("Learning Rate: ", model_xgboost.best_estimator_.get_params()["learning_rate"])
print("Gamma: ", model_xgboost.best_estimator_.get_params()["gamma"])
print("Max Depth: ", model_xgboost.best_estimator_.get_params()["max_depth"])
print("Subsample: ", model_xgboost.best_estimator_.get_params()["subsample"])
print("Max Features at Split: ", model_xgboost.best_estimator_.get_params()["colsample_bytree"])
print("Alpha: ", model_xgboost.best_estimator_.get_params()["reg_alpha"])
print("Lamda: ", model_xgboost.best_estimator_.get_params()["reg_lambda"])
print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",
      model_xgboost.best_estimator_.get_params()["min_child_weight"])
print("Number of Trees: ", model_xgboost.best_estimator_.get_params()["n_estimators"])
# Predictions
y_train_pred = model_xgboost.predict(X_train)
y_test_pred = model_xgboost.predict(X_test)
# Print F1 scores
print("Training F1 Micro Average: ", f1_score(target_train, y_train_pred, average = "micro"))
print("Test F1 Micro Average: ", f1_score(target_test, y_test_pred, average = "micro"))
print("Test Accuracy: ", accuracy_score(target_test, y_test_pred))
