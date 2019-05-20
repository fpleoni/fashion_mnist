import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend
import xgboost as xgb
import pickle

fashion_mnist_train = pd.read_csv("fashion-mnist_train.csv")

target_train = fashion_mnist_train["label"]
features_train = fashion_mnist_train.drop(["label"], axis = 1)

scaler_fit = StandardScaler().fit(features_train)

X_train = scaler_fit.transform(features_train)

# Create XGB Classifier object
xgb_clf = xgb.XGBClassifier(tree_method = "gpu_exact", predictor = "gpu_predictor",
                              verbosity = True, objective = "multi:softmax")

# Parameter grid
parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 250, 500, 1000]}

xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions = parameters, scoring = "f1_micro",
                             cv = 10, verbose = 8, random_state = 40 )

# Fit the model - this will fail.
#with parallel_backend('loky', n_jobs=2):
model_xgboost = xgb_rscv.fit(X_train, target_train)

# If it worked, pickle the model

pkl_filename = "pickled_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model_xgboost, file)