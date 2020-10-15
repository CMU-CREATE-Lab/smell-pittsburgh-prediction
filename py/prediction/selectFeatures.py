import pandas as pd
import numpy as np
from util import *
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression


# Perform feature selection (or variable selection)
# Notice that we are not going to use dimension reduction like PCA
# The reason is because dimension reduction makes variables hard to interpret
# OUTPUT: selected features
def selectFeatures(
    df_X, # dataset containing features (in pandas dataframe format)
    df_Y, # dataset containing label (in pandas dataframe format)
    is_regr=False, # regression or classification
    method="RFE", # method for selecting features
    out_p=None, # the path for saving features,
    num_feat_rfe=30, # number of features to select for RFE
    step_rfe=10, # step for RFE
    logger=None):

    log("Select features using method: " + method, logger)

    # Select model
    if is_regr:
        if method == "percent":
            model = SelectPercentile(score_func=f_regression, percentile=10)
        elif method == "fpr":
            model = SelectFpr(score_func=f_regression, alpha=0.01)
        elif method == "fdr":
            model = SelectFdr(score_func=f_regression, alpha=0.01)
        elif method == "model":
            base = Lasso(alpha=0.1, max_iter=1000)
            model = SelectFromModel(base)
        elif method == "RFE":
            base = Lasso(alpha=0.1, max_iter=1000)
            model = RFE(base, step=step_rfe, verbose=1, n_features_to_select=num_feat_rfe)
    else:
        if method == "percent":
            model = SelectPercentile(score_func=f_classif, percentile=10)
        elif method == "fpr":
            model = SelectFpr(score_func=f_classif, alpha=0.01)
        elif method == "fdr":
            model = SelectFdr(score_func=f_classif, alpha=0.01)
        elif method == "model":
            base = LogisticRegression(penalty="l1", C=0.1)
            model = SelectFromModel(base)
        elif method == "RFE":
            base = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            model = RFE(base, step=step_rfe, verbose=1, n_features_to_select=num_feat_rfe)

    # If method is None or not supported, just return the original features
    if model is None:
        log("Method is None or not supported. Return original features.", logger)
        return df_X, df_Y

    # Get X and Y, we want to fit function F such that Y=F(X)
    # Keep day and hour information (we always want to use time for features)
    log("Separate DayOfWeek and HourOfDay from features...", logger)
    label_t = ["DayOfWeek", "HourOfDay"]
    df_t = df_X[label_t].copy(deep=True)
    df_X = df_X.drop(label_t, axis=1)
    df_X_cp, df_Y_cp = df_X.copy(deep=True), df_Y.copy(deep=True)

    # Select features
    log("Perform feature selection...", logger)
    model.fit(df_X_cp, df_Y_cp.squeeze())
    selected_cols = df_X.columns[model.get_support()]
    df_X = df_X[selected_cols]
    log("Select " + str(len(selected_cols)) + " features...", logger)

    # Print feature importance
    log("Compute feature importance...", logger)
    if is_regr:
        m = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
    else:
        m = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    m.fit(df_X,df_Y.squeeze())
    feat_names = df_X.columns.copy()
    feat_ims = np.array(m.feature_importances_)
    sorted_ims_idx = np.argsort(feat_ims)[::-1]
    feat_names = feat_names[sorted_ims_idx]
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    for k in zip(feat_ims, feat_names):
        log("{0:.5f}".format(k[0]) + " -- " + str(k[1]), logger)

    # Merge
    log("Merge DayOfWeek and HourOfDay back to selected features...", logger)
    df_X = pd.concat([df_X, df_t], join="outer", axis=1)

    # Save feature names
    if out_p:
        df_X.to_csv(out_p, index=False)
        log("Selected features created at " + out_p, logger)
    return df_X, df_Y
