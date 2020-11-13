"""
Compute features for classification or regression
"""


import numpy as np
import pandas as pd
from util import log, checkAndCreateDir, isDatetimeObjTzAware
import pytz


def computeFeatures(df_esdr=None, df_smell=None, in_p=None, out_p=None, out_p_mean=None,
    out_p_std=None, is_regr=False, f_hr=8, b_hr=3, thr=40, add_roll=False, add_diff=False,
    add_inter=False, add_sqa=False, in_p_mean=None, in_p_std=None, aggr_axis=True, logger=None):
    """
    Merge all data together and compute features
    
    Input:
        df_esdr (pandas Dataframe): the preprocessed sensor data obtained from ESDR
        df_smell (pandas Dataframe): the preprocessed smell data obtained from SmellPGH
        in_p (str): input path for reading raw sensor and smell data
        out_p (str): output path for writing features and labels
        out_p_mean (str): output path for the mean of features (X)
        out_p_std (str): output path for the standard deviation of features (X)
        is_regr (bool): True means regression, and False means classification
        f_hr (int): the number of hours to look further and compute responses (Y),
            ...which is the sum of smell ratings (that are larger than 3) over the future f_hr hours
        b_hr (int): the number of hours to look back and compute features (X),
            ...which are the sensor readings (on ESDR) over the past b_hr hours
        thr: the threshold for binning the smell value into two classes (for classification)
        add_roll (bool): add rolling features
        add_diff (bool): add differential features
        add_inter (bool): add variable interaction (X1*X2) terms in the features or not
        add_sqa (bool): include the squared terms (X1^2) in the features
        in_p_mean (str): the path to read the mean values for scaling features (X)
        in_p_std (str): the path to read the standard deviation values for scaling features (X)
        aggr_axis (bool): whether we want to sum all smell reports together for all zipcodes
        logger: the python logger created by the generateLogger() function
    Output:
        df_X (pandas Dataframe): the features (X)
        df_Y (pandas Dataframe): the responses (Y)
        df_C (pandas Dataframe): the crowdsourced information (C)
    """
    log("Compute features...", logger)

    # Read preprocessed ESDR and smell report data
    if df_esdr is None or df_smell is None:
        if in_p is not None:
            df_esdr = pd.read_csv(in_p[0], parse_dates=True, index_col="DateTime")
            df_smell = pd.read_csv(in_p[1], parse_dates=True, index_col="DateTime")
        else:
            if df_esdr is None:
                log("ERROR: no data, return None.", logger)
                return None
            df_esdr = df_esdr.set_index("DateTime")
    else:
        df_esdr = df_esdr.set_index("DateTime")
        df_smell = df_smell.set_index("DateTime")

    # Convert datetime to local time zone
    if isDatetimeObjTzAware(df_esdr.index):
        df_esdr.index = df_esdr.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df_esdr.index = df_esdr.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    if df_smell is not None:
        if isDatetimeObjTzAware(df_smell.index):
            df_smell.index = df_smell.index.tz_convert(pytz.timezone("US/Eastern"))
        else:
            df_smell.index = df_smell.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))

    # Replace -1 values in esdr data to NaN
    df_esdr[df_esdr==-1] = np.nan

    # Convert degrees in wind direction to (cos(direction), sin(direction))
    df_esdr = convertWindDirection(df_esdr)

    # Extract features (X) from ESDR data
    # For models that do not have time-series structure, we want to add time-series features
    df_X = extractFeatures(df_esdr, b_hr, add_inter, add_roll, add_diff, add_sqa)
    df_X[df_X < 1e-6] = 0 # prevent extreme small values

    # Transform features
    if in_p_mean is not None and in_p_std is not None:
        df_X_mean = pd.read_csv(in_p_mean, index_col=0, squeeze=True)
        df_X_std = pd.read_csv(in_p_std, index_col=0, squeeze=True)
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
        df_Y = extractSmellResponse(df_smell, f_hr, bins, labels, aggr_axis=aggr_axis)
        if df_Y is not None:
            df_Y = df_Y.reset_index()
            # Sync DateTime column in esdr and smell data
            df_Y = pd.merge_ordered(df_X["DateTime"].to_frame(), df_Y, on="DateTime", how="inner", fill_method=None)
            df_X = pd.merge_ordered(df_Y["DateTime"].to_frame(), df_X, on="DateTime", how="inner", fill_method=None)
        # Extract crowd features (total smell values for the previous hour)
        df_C = extractSmellResponse(df_smell, None, None, None, aggr_axis=aggr_axis)
        df_C = df_C.reset_index()
        df_C = pd.merge_ordered(df_X["DateTime"].to_frame(), df_C, on="DateTime", how="inner", fill_method=None)

    # drop datetime
    df_X = df_X.drop("DateTime", axis=1)
    if df_smell is not None:
        if df_Y is not None:
            df_Y = df_Y.drop("DateTime", axis=1)
        df_C = df_C.drop("DateTime", axis=1)
    else:
        df_Y = None
        df_C = None

    # Write dataframe into a csv file
    if out_p:
        for p in out_p: checkAndCreateDir(p)
        df_X.to_csv(out_p[0], index=False)
        df_Y.to_csv(out_p[1], index=False)
        df_C.to_csv(out_p[2], index=False)
        log("Features created at " + out_p[0], logger)
        log("Labels created at " + out_p[1], logger)
        log("Crowd feature created at " + out_p[2], logger)
    if out_p_mean:
        checkAndCreateDir(out_p_mean)
        df_X_mean.to_csv(out_p_mean, index=True)
        log("Original mean created at " + out_p_mean, logger)
    if out_p_std:
        checkAndCreateDir(out_p_std)
        df_X_std.to_csv(out_p_std, index=True)
        log("Original std created at " + out_p_std, logger)
    return df_X, df_Y, df_C


def extractFeatures(df, b_hr, add_inter, add_roll, add_diff, add_sqa):
    df = df.copy(deep=True)
    df_all = []

    # Extract time series features
    df_diff = df.diff()
    for bh in range(1, b_hr + 1):
        # Add the previous readings
        df_previous = df.shift(bh)
        df_previous.columns += "_" + str(bh) + "h"
        df_all.append(df_previous)
        if add_diff:
            # Add differential feature
            df_previous_diff = df_diff.shift(bh - 1).copy(deep=True)
            df_previous_diff.columns += "_Diff" + str(bh-1) + "&" + str(bh)
            df_all.append(df_previous_diff)
        if add_roll:
            # Perform rolling mean and max (data is already resampled by hour)
            if bh <= 1: continue
            df_roll = df.rolling(bh, min_periods=1)
            df_roll_max = df_roll.max()
            df_roll_mean = df_roll.mean()
            df_roll_max.columns += "_Max" + str(bh)
            df_roll_mean.columns += "_Mean" + str(bh)
            df_all.append(df_roll_max)
            df_all.append(df_roll_mean)

    # Combine dataframes
    #df.columns += ".Now"
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
                    c = c1 + " * " + c2
                    df_inte[c] = df_feat[c1] * df_feat[c2]

    # Add squared terms
    if add_sqa:
        df_sqa = pd.DataFrame()
        L = len(df_feat.columns)
        for i in range(0, L):
            c1 = df_feat.columns[i]
            c = c1 + "_sqare"
            df_sqa[c] = df_feat[c1]**2

    # Merge dataframes
    if add_inter:
        df_feat = df_feat.join(df_inte)
    if add_sqa:
        df_feat = df_feat.join(df_sqa)

    return df_feat


def extractSmellResponse(df, f_hr, bins, labels, aggr_axis=False):
    df_resp = df.copy(deep=True)

    # Compute the total smell_values in future f_hr hours
    if f_hr is not None:
        df_resp = df_resp.rolling(f_hr, min_periods=1).sum().shift(-1*f_hr)
        # Remove the last f_hr rows
        df_resp = df_resp.iloc[:-1*f_hr]

    # Bin smell values for classification
    if aggr_axis: df_resp = df_resp.sum(axis=1)
    if bins is not None and labels is not None:
        df_resp = pd.cut(df_resp, bins, labels=labels, right=False)
    if aggr_axis: df_resp.name = "smell"

    # Sanity check
    if len(df_resp) == 0: df_resp = None

    return df_resp


def convertWindDirection(df):
    df_cp = df.copy(deep=True)
    for c in df.columns:
        if "SONICWD_DEG" in c or "@" in c:
            df_c = df[c]
            df_c.name = df_c.name.replace("@", "")
            df_c_cos = np.cos(np.deg2rad(df_c))
            df_c_sin = np.sin(np.deg2rad(df_c))
            df_c_cos.name += "cosine"
            df_c_sin.name += "sine"
            df_cp.drop([c], axis=1, inplace=True)
            df_cp[df_c_cos.name] = df_c_cos
            df_cp[df_c_sin.name] = df_c_sin
    return df_cp
