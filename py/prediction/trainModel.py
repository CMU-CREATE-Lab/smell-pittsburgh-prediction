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
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
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
    method="ET", # the regression or classification method
    is_regr=False, # is regression or not
    balance=False, # oversample or undersample data or not
    logger=None):

    log("Training model with " + str(train["X"].shape[1]) + " features...", logger)

    # Build model
    multi_output = True if len(train["Y"]) > 1 and train["Y"].shape[1] > 1 else False
    if is_regr:
        if method == "RF":
            model = RandomForestRegressor(n_estimators=800, random_state=0, n_jobs=-1)
        elif method == "ET":
            model = ExtraTreesRegressor(n_estimators=800, random_state=0, n_jobs=-1)
        elif method == "SVM":
            model = SVR(max_iter=5000)
            if multi_output: model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "RLR":
            model = HuberRegressor(max_iter=1000)
            if multi_output: model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "LR":
            model = LinearRegression()
            if multi_output: model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "EN":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
            if multi_output: model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "LA":
            model = Lasso(alpha=0.1, max_iter=1000)
            if multi_output: model = MultiOutputRegressor(model, n_jobs=-1)
        elif method == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(128, 64))
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
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=None)
        elif method == "ET":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=None)
        elif method == "SVM":
            model = SVC(max_iter=5000, kernel="rbf", random_state=0, probability=True)
        elif method == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(128, 64))
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
        elif method == "RF-feat-10":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=10)
        elif method == "ET-feat-10":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=10)
        elif method == "RF-feat-20":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=20)
        elif method == "ET-feat-20":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=20)
        elif method == "RF-feat-30":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=30)
        elif method == "ET-feat-30":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=30)
        elif method == "RF-feat-40":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=40)
        elif method == "ET-feat-40":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=40)
        elif method == "RF-feat-50":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=50)
        elif method == "ET-feat-50":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=50)
        elif method == "RF-feat-60":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=60)
        elif method == "ET-feat-60":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=60)
        elif method == "RF-feat-70":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=70)
        elif method == "ET-feat-70":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=70)
        elif method == "RF-feat-80":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=80)
        elif method == "ET-feat-80":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=80)
        elif method == "RF-feat-90":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=90)
        elif method == "ET-feat-90":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=90)
        elif method == "RF-feat-100":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=100)
        elif method == "ET-feat-100":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=100)
        elif method == "RF-feat-110":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=110)
        elif method == "ET-feat-110":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=110)
        elif method == "RF-feat-120":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=120)
        elif method == "ET-feat-120":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=120)
        elif method == "RF-feat-130":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=130)
        elif method == "ET-feat-130":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=130)
        elif method == "RF-feat-140":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=140)
        elif method == "ET-feat-140":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=140)
        elif method == "RF-feat-150":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=150)
        elif method == "ET-feat-150":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=150)
        elif method == "RF-feat-160":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=160)
        elif method == "ET-feat-160":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=160)
        elif method == "RF-feat-170":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=170)
        elif method == "ET-feat-170":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=170)
        elif method == "RF-feat-all":
            model = RandomForestClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=None)
        elif method == "ET-feat-all":
            model = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1, max_features=None)
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
        model.fit(X, np.squeeze(Y), copy.deepcopy(train["X_pretrain"]))
    else:
        model.fit(X, np.squeeze(Y))

    # Save and return model
    if out_p is not None:
        if method == "CRNN" or method == "CRMANN":
            model.save(out_p)
        else:
            joblib.dump(model, out_p)
        log("Model saved at " + out_p, logger)
    return model
