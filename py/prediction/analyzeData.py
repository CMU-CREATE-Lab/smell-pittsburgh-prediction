from util import *
import numpy as np
import pandas as pd
from computeFeatures import *
from trainModel import *
import copy

def analyzeData(
    in_p=None, # input path for raw esdr and smell data
    logger=None):

    log("Analyze data...", logger)

    # Evaluate model performance
    #evalModel(in_p, logger=logger)
    
    # Correlational study
    corrStudy(in_p, logger=logger)

# Correlational study
def corrStudy(in_p, logger=None):
    log("Compute correlation of lagged time-series data...", logger)
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=0, thr=40, is_regr=True,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    max_t_lag = 8 # the maximum time lag
    df = pd.DataFrame()
    for col in df_X.columns:
        if col in ["Day", "DayOfWeek", "HourOfDay"]: continue
        s = []
        Y = df_Y.squeeze()
        X = df_X[col]
        for i in range(0, max_t_lag+1):
            s.append(np.round(Y.corr(X.shift(i)), 3))
        df[col] = pd.Series(data=s)
    for col in df.columns:
        print df[col]

# Evaluate the model and compute feature performance
def evalModel(in_p, logger=None):
    # Compute feature importance
    log("Compute feature importance using ExtraTrees...", logger)
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=3, thr=40, is_regr=False,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    model = trainModel({"X": df_X, "Y": df_Y}, method="ET", logger=logger)
    feat_ims = np.array(model.feature_importances_)
    sorted_ims_idx = np.argsort(feat_ims)
    feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
    feat_names = df_X.columns.copy()
    feat_names = feat_names[sorted_ims_idx]
    for k in zip(feat_ims, feat_names):
        log("{0:.5f}".format(k[0]) + "--" + str(k[1]), logger)

    # Evaluate performance
    log("Compute evaluation metrics...", logger)
    metric = computeMetric(df_Y, model.predict(df_X), False)
    for m in metric:
        log(metric[m], logger)
