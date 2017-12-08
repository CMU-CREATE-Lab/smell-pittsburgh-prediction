import numpy as np
import numpy.matlib
import copy
from util import *
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from CRnnLearner import CRnnLearner
from ANCnnLearner import ANCnnLearner
from DmlpLearner import DmlpLearner

# Train a regression or classification model
# To find a function F such that Y=F(X)
# OUTPUT: model
def trainModel(
    train, # the training set in pandas dataframe (contain train["X"] and train["Y"])
    test=None, # the testing set in pandas dataframe (contain test["X"] and test["Y"])
    out_p=None, # the path for saving the model
    method="SVM", # the regression or classification method
    is_regr=False, # is regression or not
    balance=False, # oversample or undersample data or not
    logger=None):

    log("Training model...", logger)

    # Build model
    if is_regr:
        if method == "RF":
            model = RandomForestRegressor(n_estimators=800, random_state=0, n_jobs=-1)
        elif method == "ET":
            model = ExtraTreesRegressor(n_estimators=800, random_state=0, n_jobs=-1)
        elif method == "SVM":
            model = SVR(max_iter=5000)
            model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "RLR":
            model = HuberRegressor(max_iter=1000)
            model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "LR":
            model = LinearRegression()
            model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "EN":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
            model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "GP":
            model = GaussianProcessRegressor(n_restarts_optimizer=10)
            model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "MLP":
            model = MLPRegressor(hidden_layer_sizes=512)
        elif method == "KN":
            model = KNeighborsRegressor(n_neighbors=10, weights="uniform")
        elif method == "DMLP":
            model = DmlpLearner(test=test, logger=logger, is_regr=is_regr)
        elif method == "CRNN":
            model = CRnnLearner(test=test, logger=logger, is_regr=is_regr)
        elif method == "ANCNN":
            model = ANCnnLearner(test=test, logger=logger, is_regr=is_regr)
        else:
            log("ERROR: method " + method + " is not supported", logger)
            return None
    else:
        if method == "RF":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=20)
        elif method == "ET":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=20) 
        elif method == "SVM":
            model = SVC(max_iter=5000, kernel="rbf", random_state=0, probability=True)
        elif method == "GP":
            model = GaussianProcessClassifier(max_iter_predict=1000)
        elif method == "MLP":
            model = MLPClassifier(hidden_layer_sizes=512)
        elif method == "KN":
            model = KNeighborsClassifier(n_neighbors=10, weights="uniform")
        elif method == "LG":
            model = LogisticRegression()
        elif method == "DMLP":
            model = DmlpLearner(test=test, logger=logger, is_regr=is_regr)
        elif method == "CRNN":
            model = CRnnLearner(test=test, logger=logger, is_regr=is_regr)
        elif method == "ANCNN":
            model = ANCnnLearner(test=test, logger=logger, is_regr=is_regr)
        else:
            log("ERROR: method " + method + " is not supported", logger)
            return None

    # Use balanced dataset or not
    if balance:
        log("Compute balanced dataset...", logger)
        X, Y = balanceDataset(train["X"], train["Y"])
    else:
        X, Y = copy.deepcopy(train["X"]), copy.deepcopy(train["Y"])

    # For one-class classification task, we only want to use the minority class (because we are sure that they are labeled)
    if not is_regr and method == "IF":
        y_minor = findLeastCommon(Y)
        select_y = (Y == y_minor)
        X, Y = X[select_y], Y[select_y]
    
    # Fit data to the model
    if method == "ANCNN":
        model.fit(X, Y, copy.deepcopy(train["X_pretrain"]))
    else:
        model.fit(X, Y)

    # Save and return model
    if out_p is not None:
        if method == "CRNN" or method == "CRMANN":
            model.save(out_p)
        else:
            joblib.dump(model, out_p)
        log("Model saved at " + out_p, logger)
    return model
