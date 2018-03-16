from util import *
import numpy as np
import pandas as pd
from computeFeatures import *
from ForestInterpreter import *
import copy

def analyzeData(
    in_p=None, # input path for raw esdr and smell data
    logger=None):

    log("Analyze data...", logger)

    # Evaluate model performance
    evalModel(in_p, logger=logger)
    
    # Correlational study
    #corrStudy(in_p, logger=logger)

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
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=3, thr=40, is_regr=False,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    model = ForestInterpreter(df_X=df_X, df_Y=df_Y, logger=logger)
    #model.reportFeatureImportance()
    #model.reportPerformance()
