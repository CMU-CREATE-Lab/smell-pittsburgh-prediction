import numpy as np
import pandas as pd
from util import *
from sklearn import preprocessing

# Merge all data together and compute features
# OUTPUT: pandas dataframe containing features
def computeFeatures(
    df_esdr=None, # the pandas dataframe that contains the predictors (esdr data)
    df_smell=None, # the pandas dataframe that contains the responses (smell data)
    in_p=None, # input path for raw esdr and smell data
    out_p=None, # output path for features and labels
    out_p_mean=None, # output path for the mean of features
    out_p_std=None, # output path for the standard deviation of features
    is_regr=False, # regression or classification
    f_hr=8, # the number of hours to look further and compute responses (Y)
    b_hr=12, # the number of hours to look back and compute features (X)
    thr=80, # for binning the smell value into two classes (this is for classification)
    add_roll=True, # add rolling features
    add_diff=True, # add differential features
    add_inter=True, # add variable interaction (X1*X2) terms in the features or not
    in_p_mean=None, # the path of mean values in pandas dataframe for scaling features
    in_p_std=None, # the path of std in pandas dataframe for scaling features
    logger=None):
    
    log("Compute features...", logger)

    # Read ESDR and smell report data
    if df_esdr is None or df_smell is None:
        if in_p is not None:
            df_esdr = pd.read_csv(in_p[0], parse_dates=True, index_col="DateTime")
            df_smell = pd.read_csv(in_p[1], parse_dates=True, index_col="DateTime")
        else:
            if df_esdr is None:
                log("ERROR: no data, return None.", logger)
                return None
            else:
                df_esdr = df_esdr.set_index("DateTime")
    else:
        df_esdr = df_esdr.set_index("DateTime")
        df_smell = df_smell.set_index("DateTime")

    # Replace -1 values in esdr data to NaN
    df_esdr[df_esdr==-1] = np.nan

    # Convert degrees in wind direction to (cos(direction), sin(direction))
    df_esdr = convertWindDirection(df_esdr)

    # Extract features (X) from ESDR data
    # For models that do not have time-series structure, we want to add time-series features
    # For models that do have time-series structure (e.g. RNN), we want to use original features
    df_X = extractFeatures(df_esdr, b_hr, add_inter, add_roll, add_diff)
    df_X[df_X < 1e-6] = 0 # prevent extreme small values 
    
    # Transform features
    if in_p_mean is not None and in_p_std is not None:
        df_X_mean = pd.read_csv(in_p_mean, header=None, index_col=0, squeeze=True)
        df_X_std = pd.read_csv(in_p_std, header=None, index_col=0, squeeze=True)
    else:
        df_X_mean = df_X.mean()
        df_X_std = df_X.std()
    df_X = (df_X - df_X_mean) / df_X_std
    df_X = df_X.round(8)
    df_X = df_X.fillna(0)
    
    # Add day of month, days of week, and hours of day
    df_X["Day"] = df_X.index.day
    df_X["DayOfWeek"] = df_X.index.dayofweek
    df_X["HourOfDay"] = df_X.index.hour
    df_X = df_X.reset_index()

    # Extract responses from smell data
    if df_smell is not None:
        bins = None if is_regr else [-np.inf, thr, np.inf] # bin smell reports into labels or not
        labels = None if is_regr else [0, 1]
        #aggr_axis = False if is_regr else True
        aggr_axis = True
        df_Y = extractSmellResponse(df_smell, f_hr, bins, labels, aggr_axis=aggr_axis)
        df_Y = df_Y.reset_index()
        # Sync DateTime column in esdr and smell data
        df_Y = pd.merge_ordered(df_X["DateTime"].to_frame(), df_Y, on="DateTime", how="inner", fill_method=None)
        df_X = pd.merge_ordered(df_Y["DateTime"].to_frame(), df_X, on="DateTime", how="inner", fill_method=None)

    # drop datetime
    df_X = df_X.drop("DateTime", axis=1)
    if df_smell is not None:
        df_Y = df_Y.drop("DateTime", axis=1)
    else:
        df_Y = None

    # Write dataframe into a csv file
    if out_p:
        for p in out_p: checkAndCreateDir(p)
        df_X.to_csv(out_p[0], index=False)
        df_Y.to_csv(out_p[1], index=False)
        log("Features created at " + out_p[0], logger)
        log("Labels created at " + out_p[1], logger)
    if out_p_mean:
        checkAndCreateDir(out_p_mean)
        df_X_mean.to_csv(out_p_mean, index=True)
        log("Original mean created at " + out_p_mean, logger)
    if out_p_std:
        checkAndCreateDir(out_p_std)
        df_X_std.to_csv(out_p_std, index=True)
        log("Original std created at " + out_p_std, logger)
    return df_X, df_Y

def extractFeatures(df, b_hr, add_inter, add_roll, add_diff):
    df = df.copy(deep=True)
    df_all = []

    # Add interaction of variables
    #if add_inter:
    #    df_inte = pd.DataFrame()
    #    L = len(df.columns)
    #    for i in range(0, L):
    #        for j in range(0, L):
    #            if j > i:
    #                c1 = df.columns[i]
    #                c2 = df.columns[j]
    #                c = "interaction..." + c1 + "...and..." + c2
    #                df_inte[c] = df[c1] * df[c2]
    #    df_all.append(df_inte)

    # Extract time series features
    df_diff = df.diff()
    for b_hr in range(1, b_hr + 1):
        # Add the previous readings
        df_previous = df.shift(b_hr)
        df_previous.columns += ".previous." + str(b_hr) + ".hour"
        df_all.append(df_previous)
        if add_diff:
            # Add differential feature
            df_previous_diff = df_diff.shift(b_hr - 1).copy(deep=True)
            df_previous_diff.columns += ".diff.of.previous." + str(b_hr-1) + ".and." + str(b_hr) + ".hour"
            df_all.append(df_previous_diff)
        if add_roll:
            # Perform rolling mean and max (data is already resampled by hour)
            if b_hr <= 1: continue
            df_roll = df.rolling(b_hr, min_periods=1)
            df_roll_max = df_roll.max()
            df_roll_mean = df_roll.mean()
            df_roll_max.columns += ".max.of.last." + str(b_hr) + ".hours"
            df_roll_mean.columns += ".mean.of.last." + str(b_hr) + ".hours"
            df_all.append(df_roll_max)
            df_all.append(df_roll_mean)
    
    # Combine dataframes
    df.columns += ".current"
    df_feat = df
    for d in df_all:
        df_feat = df_feat.join(d)
    
    # Delete the first b_hr rows
    df_feat = df_feat.iloc[b_hr:]

    # Add interaction of variables
    if add_inter:
        df_inte = pd.DataFrame()
        L = len(df_feat.columns)
        for i in range(0, L):
            for j in range(0, L):
                if j > i:
                    c1 = df_feat.columns[i]
                    c2 = df_feat.columns[j]
                    c = "interaction..." + c1 + "...and..." + c2
                    df_inte[c] = df_feat[c1] * df_feat[c2]
        df_feat = df_feat.join(df_inte)
    
    return df_feat

def extractSmellResponse(df, f_hr, bins, labels, aggr_axis=False):
    df = df.copy(deep=True)
    
    # Compute the total smell_values in future f_hr hours
    if f_hr is not None:
        df_resp = df.rolling(f_hr, min_periods=1).sum().shift(-1*f_hr)
        # Remove the last f_hr rows
        df_resp = df_resp.iloc[:-1*f_hr]
    
    # Bin smell values for classification
    if aggr_axis: df_resp = df_resp.sum(axis=1)
    if bins is not None and labels is not None:
        df_resp = pd.cut(df_resp, bins, labels=labels, right=False)
    if aggr_axis: df_resp.name = "smell"

    return df_resp

def convertWindDirection(df):
    df_cp = df.copy(deep=True)
    for c in df.columns:
        if "SONICWD_DEG" in c:
            df_c = df[c]
            df_c_cos = np.cos(np.deg2rad(df_c))
            df_c_sin = np.sin(np.deg2rad(df_c))
            df_c_cos.name += ".cosine"
            df_c_sin.name += ".sine"
            df_cp.drop([c], axis=1, inplace=True) 
            df_cp[df_c_cos.name] = df_c_cos
            df_cp[df_c_sin.name] = df_c_sin
    return df_cp
