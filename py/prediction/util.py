"""
Utility functions for smell-pittsburgh-prediction
"""

import logging
from os import listdir
from os.path import isfile, join
import os
from datetime import datetime
from copy import deepcopy
import uuid
import pytz
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generateLogger(file_name, log_level=logging.INFO, name=str(uuid.uuid4()), format="%(asctime)s %(levelname)s %(message)s"):
    """Generate a logger for loggin files"""
    if log_level == "debug": log_level = logging.DEBUG
    checkAndCreateDir(file_name)
    formatter = logging.Formatter(format)
    #handler = logging.FileHandler(file_name, mode="w") # mode "w" is for overriding file
    handler = logging.FileHandler(file_name, mode="a")
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    for hdlr in logger.handlers[:]: logger.removeHandler(hdlr) # remove old handlers
    logger.addHandler(handler)
    return logger


def log(msg, logger=None, level="info"):
    """Log and print"""
    if logger is not None:
        if level == "info":
            logger.info(msg)
        elif level == "error":
            logger.error(msg)
    print(msg)


def findLeastCommon(arr):
    """Find the least common elements in a given array"""
    m = Counter(arr)
    return m.most_common()[-1][0]


def isFileHere(path):
    """Check if a file exists"""
    return os.path.isfile(path)


def getAllFileNamesInFolder(path):
    """Return a list of all files in a folder"""
    return  [f for f in listdir(path) if isfile(join(path, f))]


def esdrRootUrl():
    """Return the root url for ESDR"""
    return "https://esdr.cmucreatelab.org/"


def smellPghRootUrl():
    """Return the root url for SmellPGH production"""
    return "http://api.smellpittsburgh.org/"


def datetimeToEpochtime(dt):
    """Convert a datetime object to epoch time"""
    if dt.tzinfo is None:
        dt_utc = dt
    else:
        dt_utc = dt.astimezone(pytz.utc).replace(tzinfo=None)
    epoch_utc = datetime.utcfromtimestamp(0)
    return int((dt_utc - epoch_utc).total_seconds() * 1000)


def checkAndCreateDir(path):
    """Check if a directory exists, if not, create it"""
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name)


def epochtimeIdxToDatetime(df):
    """Convert the epochtime index in a pandas dataframe to datetime index"""
    df = df.copy(deep=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    df.index.name = "DateTime"
    return df


def removeNonAsciiChars(str_in):
    """Remove all non-ascii characters in the string"""
    if str_in is None:
        return ""
    return str_in.encode("ascii", "ignore").decode()


def evalEventDetection(Y_true, Y_pred, thr=40, h=1, round_to_decimal=3):
    """
    Compute a custom metric for evaluating the regression function
    Notice that for daytime cases, the Y arrays may contain NaN
    For each smoke event, the prediction only need to hit the event at some time point
    (for an event from 9am to 11am, good enough if there are at least one predicted event within it)
    Denote T the 1D signal of the true data
    Denote P the 1D signal of the predicted data
    1. Detect the time intervals in T and P that has values larger than a threshold "thr"
    2. Merge intervals that are less or equal than "h" hours away from each other
        (e.g., for h=1, intervals [1,3] and [4,5] need to be merged into [1,5])
    3. Compute the precision, recall, and f-score for each interval ...
        true positive: for each t in T, if it overlaps with a least one p in P
        false positive: for each p in P, if there is no t in T that overlaps with it
        false negative: for each t in T, if there is no p in P that overlaps with it
    """
    # Convert Y_true and Y_pred into binary signals and to intervals
    Y_true_iv, Y_pred_iv = binary2Interval(Y_true>=thr), binary2Interval(Y_pred>=thr)

    # Merge intervals
    Y_true_iv, Y_pred_iv = mergeInterval(Y_true_iv, h=h), mergeInterval(Y_pred_iv, h=h)

    # Compute true positive and false negative
    TP = 0
    FN = 0
    for t in Y_true_iv:
        has_overlap = False
        for p in Y_pred_iv:
            # find overlaps (four possible cases)
            c1 = t[0]<=p[0]<=t[1] and t[0]<=p[1]<=t[1]
            c2 = p[0]<t[0] and t[0]<=p[1]<=t[1]
            c3 = t[0]<=p[0]<=t[1] and p[1]>t[1]
            c4 = p[0]<t[0] and p[1]>t[1]
            if c1 or c2 or c3 or c4:
                has_overlap = True
                break
        if has_overlap: TP += 1
        else: FN += 1

    # Compute false positive
    FP = 0
    for p in Y_pred_iv:
        has_overlap = False
        for t in Y_true_iv:
            # find overlaps (four possible cases)
            c1 = p[0]<=t[0]<=p[1] and p[0]<=t[1]<=p[1]
            c2 = t[0]<p[0] and p[0]<=t[1]<=p[1]
            c3 = p[0]<=t[0]<=p[1] and t[1]>p[1]
            c4 = t[0]<p[0] and t[1]>p[1]
            if c1 or c2 or c3 or c4:
                has_overlap = True
                break
        if not has_overlap: FP += 1

    # Compute precision, recall, f-score
    TP, FN, FP = float(TP), float(FN), float(FP)
    if TP + FP == 0: precision = 0
    else: precision = TP / (TP + FP)
    if TP + FN == 0: recall = 0
    else: recall = TP / (TP + FN)
    if precision + recall == 0: f_score = 0
    else: f_score = 2 * (precision * recall) / (precision + recall)

    # Round to
    precision = round(precision, round_to_decimal)
    recall = round(recall, round_to_decimal)
    f_score = round(f_score, round_to_decimal)

    return {"TP":TP, "FP":FP, "FN":FN, "precision":precision, "recall":recall, "f_score":f_score}


def mergeInterval(intervals, h=1):
    """
    Merge intervals that are less or equal than "h" hours away from each other
    (e.g., for h=1, intervals [1,3] and [4,5] need to be merged into [1,5])
    """
    intervals_merged = []
    current_iv = None
    for iv in intervals:
        if current_iv is None:
            current_iv = iv
        else:
            if iv[0] - current_iv[1] <= h:
                current_iv[1] = iv[1]
            else:
                intervals_merged.append(current_iv)
                current_iv = iv
    if current_iv is not None:
        intervals_merged.append(current_iv)
    return intervals_merged


def binary2Interval(Y):
    """
    Convert a binary array with False and True to intervals
    input = [False, True, True, False, True, False]
    output = [[1,2], [4,4]]
    """
    Y_cp = np.append(Y, False) # this is important for case like [False, True, True]
    intervals = []
    current_iv = None
    for i in range(0, len(Y_cp)):
        if Y_cp[i] and current_iv is None:
            current_iv = [i, i]
        if not Y_cp[i] and current_iv is not None:
            current_iv[1] = i - 1
            intervals.append(current_iv)
            current_iv = None
    return intervals


def computeMetric(Y_true, Y_pred, is_regr, flatten=False, simple=False,
        round_to_decimal=3, aggr_axis=False, only_binary=True, event_thr=40):
    """
    Compute the evaluation result of regression or classification Y=F(X)
    INPUTS:
    - Y_true: the true values of Y
    - Y_pred: the predicted values of Y
    - is_regr: is regression or classification
    - event_thr: the threshold for defining an event, used when applying regression to detect events
    OUTPUT:
    - r2: r-squared (for regression)
    - mse: mean squared error (for regression)
    - prf: precision, recall, and f-score (for classification) in pandas dataframe format
    - cm: confusion matrix (for classification) in pandas dataframe format
    """
    Y_true, Y_pred = deepcopy(Y_true), deepcopy(Y_pred)
    if len(Y_true.shape) > 2: Y_true = np.reshape(Y_true, (Y_true.shape[0], -1))
    if len(Y_pred.shape) > 2: Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], -1))
    if aggr_axis and is_regr:
        if len(Y_true.shape) > 1:
            Y_true = np.sum(Y_true, axis=1)
        if len(Y_pred.shape) > 1:
            Y_pred = np.sum(Y_pred, axis=1)
    if only_binary and not is_regr:
        Y_pred[Y_pred>1] = 1
    Y_true_origin, Y_pred_origin = deepcopy(Y_true), deepcopy(Y_pred)
    Y_true, Y_pred = Y_true[~np.isnan(Y_true)], Y_pred[~np.isnan(Y_pred)]
    metric = {}
    # Compute the precision, recall, and f-score for smoke events
    if not simple:
        thr = event_thr if is_regr else 1
        event_prf = evalEventDetection(Y_true_origin, Y_pred_origin, thr=thr)
        metric["event_prf"] = event_prf
    if is_regr:
        # Compute r-squared value and mean square error
        r2 = r2_score(Y_true, Y_pred, multioutput="variance_weighted")
        mse = mean_squared_error(Y_true, Y_pred, multioutput="uniform_average")
        metric["r2"] = round(r2, round_to_decimal)
        metric["mse"] = round(mse, round_to_decimal)
    else:
        # Compute precision, recall, fscore, and confusion matrix
        cm = confusion_matrix(Y_true, Y_pred).round(round_to_decimal)
        prf_class = precision_recall_fscore_support(Y_true, Y_pred, average=None)
        prf_avg = precision_recall_fscore_support(Y_true, Y_pred, average="macro")
        prf = []
        idx = []
        col = ["p", "r", "f", "s"] if simple else ["precision", "recall", "fscore", "support"]
        for i in range(0, len(prf_class)):
            prf.append(np.append(prf_class[i], prf_avg[i]))
        for i in range(0, len(prf_class[0])):
            if simple:
                idx.append(str(i))
            else:
                idx.append("class_" + str(i))
        prf[-1][-1] = np.sum(prf_class[3])
        prf = np.array(prf).astype(float).round(round_to_decimal).T
        idx_avg = "avg" if simple else "average"
        df_prf = pd.DataFrame(data=prf, index=np.append(idx, idx_avg), columns=col)
        df_cm = pd.DataFrame(data=cm, index=idx, columns=idx)
        df_cm.index = ("t" + df_cm.index) if simple else ("true_" + df_cm.index)
        df_cm.columns = ("p" + df_cm.columns) if simple else ("predicted_" + df_cm.columns)
        metric["prf"] = df_prf
        metric["cm"] = df_cm
        if flatten:
            metric["prf"] = flattenDataframe(metric["prf"])
            metric["cm"] = flattenDataframe(metric["cm"])
    return metric


def evaluateData(Y_true, Y_pred, X, col_names=None):
    """
    Get wrongly and correctly classified data points
    Only works for classification with label 0 and 1
    INPUT:
    - Y_true: the true values of responses (in numpy format, 1D array)
    - Y_pred: the predicted values of responses (in numpy format, 1D array)
    - X: the predictors (in numpy format, 2D array)
    - col_names: the column names for creating the pandas dataframe
    OUTPUT:
    - true positives (tp), false positives (fp), true negatives (tn), false negatives(fn)
    """
    if col_names is None: col_names = map(str, range(0,X.shape[1]))
    Y_true = np.squeeze(Y_true)
    Y_pred = np.squeeze(Y_pred)
    # Get index
    idx_tp = (Y_true==1)&(Y_pred==1)
    idx_fp = (Y_true==0)&(Y_pred==1)
    idx_tn = (Y_true==0)&(Y_pred==0)
    idx_fn = (Y_true==1)&(Y_pred==0)
    # Get X
    X_tp = X[idx_tp]
    X_fp = X[idx_fp]
    X_tn = X[idx_tn]
    X_fn = X[idx_fn]
    # Get Y
    Y_true_tp = Y_true[idx_tp]
    Y_true_fp = Y_true[idx_fp]
    Y_true_tn = Y_true[idx_tn]
    Y_true_fn = Y_true[idx_fn]
    Y_pred_tp = Y_pred[idx_tp]
    Y_pred_fp = Y_pred[idx_fp]
    Y_pred_tn = Y_pred[idx_tn]
    Y_pred_fn = Y_pred[idx_fn]
    # Create dataframe
    data_tp = np.insert(X_tp, 0 , [Y_true_tp, Y_pred_tp], axis=1)
    data_fp = np.insert(X_fp, 0 , [Y_true_fp, Y_pred_fp], axis=1)
    data_tn = np.insert(X_tn, 0 , [Y_true_tn, Y_pred_tn], axis=1)
    data_fn = np.insert(X_fn, 0 , [Y_true_fn, Y_pred_fn], axis=1)
    columns = ["Y_true", "Y_pred"] + list(col_names)
    df_tp = pd.DataFrame(data=data_tp, columns=columns)
    df_fp = pd.DataFrame(data=data_fp, columns=columns)
    df_tn = pd.DataFrame(data=data_tn, columns=columns)
    df_fn = pd.DataFrame(data=data_fn, columns=columns)
    return {"tp": df_tp, "fp": df_fp, "tn": df_tn, "fn": df_fn}


def flattenDataframe(df):
    """Flatten a pandas dataframe"""
    df = df.stack()
    idx = df.index.values.tolist()
    for i in range(0, len(idx)):
        idx[i] = ".".join(idx[i])
    val = df.values.tolist()
    return [idx, val]


def getEsdrData(source, **options):
    """
    Get data from ESDR
    source = [
        [{"feed": 27, "channel": "NO_PPB"}],
        [{"feed": 1, "channel": "PM25B_UG_M3"}, {"feed": 1, "channel": "PM25T_UG_M3"}]
    ]
    if source = [[A,B],[C]], this means that A and B will be merged
    start_time: starting epochtime in seconds
    end_time: ending epochtime in seconds
    """
    print("Get ESDR data...")

    # Url parts
    api_url = esdrRootUrl() + "api/v1/"
    export_para = "/export?format=csv"
    if "start_time" in options:
        export_para += "&from=" + str(options["start_time"])
    if "end_time" in options:
        export_para += "&to=" + str(options["end_time"])

    # Loop each source
    data = []
    for s_all in source:
        df = None
        for s in s_all:
            # Read data
            feed_para = "feeds/" + s["feed"]
            channel_para = "/channels/" + s["channel"]
            df_s = pd.read_csv(api_url + feed_para + channel_para + export_para)
            df_s.set_index("EpochTime", inplace=True)
            if "factor" in s:
                df_s = df_s * s["factor"]
            if df is None:
                df = df_s
            else:
                # Merge column names
                c = []
                for k in zip(df.columns, df_s.columns):
                    if k[0] != k[1]:
                        c.append(k[0] + ".." + k[1])
                    else:
                        c.append(k[0])
                df.columns = c
                df_s.columns = c
                df = pd.concat([df[~df.index.isin(df_s.index)], df_s])
        df = df.apply(pd.to_numeric, errors="coerce") # To numeric values
        data.append(df)

    # Return
    return data


def getSmellReports(**options):
    if "api_version" in options and options["api_version"] == 1:
        return getSmellReportsV1(**options)
    else:
        return getSmellReportsV2(**options)


def getSmellReportsV2(**options):
    """Get smell reports data from SmellPGH"""
    print("Get smell reports from V2 API...")

    # Url
    api_url = smellPghRootUrl() + "api/v2/"
    api_para = "smell_reports?"
    if "allegheny_county" in options and options["allegheny_county"] == True:
        # This is for general dataset usage in the Allegheny County in Pittsburgh
        api_para += "zipcodes=15006,15007,15014,15015,15017,15018,15020,15024,15025,15028,15030,15031,15032,15034,15035,15037,15044,15045,15046,15047,15049,15051,15056,15064,15065,15071,15075,15076,15082,15084,15086,15088,15090,15091,15095,15096,15101,15102,15104,15106,15108,15110,15112,15116,15120,15122,15123,15126,15127,15129,15131,15132,15133,15134,15135,15136,15137,15139,15140,15142,15143,15144,15145,15146,15147,15148,15201,15202,15203,15204,15205,15206,15207,15208,15209,15210,15211,15212,15213,15214,15215,15216,15217,15218,15219,15220,15221,15222,15223,15224,15225,15226,15227,15228,15229,15230,15231,15232,15233,15234,15235,15236,15237,15238,15239,15240,15241,15242,15243,15244,15250,15251,15252,15253,15254,15255,15257,15258,15259,15260,15261,15262,15264,15265,15267,15268,15270,15272,15274,15275,15276,15277,15278,15279,15281,15282,15283,15286,15289,15290,15295"
    else:
        # This is for our smell pgh paper
        api_para += "zipcodes=15221,15218,15222,15219,15201,15224,15213,15232,15206,15208,15217,15207,15260,15104"
    if "start_time" in options:
        api_para += "&start_time=" + str(options["start_time"])
    if "end_time" in options:
        api_para += "&end_time=" + str(options["end_time"])

    # Load smell reports
    df = pd.read_json(api_url + api_para, convert_dates=False)

    # If empty, return None
    if df.empty:
        return None

    # Wrangle text
    df["smell_description"] = df["smell_description"].replace(np.nan, "").map(removeNonAsciiChars)
    df["feelings_symptoms"] = df["feelings_symptoms"].replace(np.nan, "").map(removeNonAsciiChars)
    df["additional_comments"] = df["additional_comments"].replace(np.nan, "").map(removeNonAsciiChars)

    # Set index and drop columns
    df.set_index("observed_at", inplace=True)
    df.index.names = ["EpochTime"]
    df.rename(columns={"latitude": "skewed_latitude", "longitude": "skewed_longitude"}, inplace=True)
    df.drop(["zip_code_id"], axis=1, inplace=True)

    # Return
    return df


def getSmellReportsV1(**options):
    """Get smell reports data from SmellPGH"""
    print("Get smell reports from V1 API...")

    # Url
    api_url = smellPghRootUrl() + "api/v1/"
    api_para = "smell_reports?"
    if "allegheny_county" in options and options["allegheny_county"] == True:
        # This is for general dataset usage in the Allegheny County in Pittsburgh
        api_para += "zipcodes=15006,15007,15014,15015,15017,15018,15020,15024,15025,15028,15030,15031,15032,15034,15035,15037,15044,15045,15046,15047,15049,15051,15056,15064,15065,15071,15075,15076,15082,15084,15086,15088,15090,15091,15095,15096,15101,15102,15104,15106,15108,15110,15112,15116,15120,15122,15123,15126,15127,15129,15131,15132,15133,15134,15135,15136,15137,15139,15140,15142,15143,15144,15145,15146,15147,15148,15201,15202,15203,15204,15205,15206,15207,15208,15209,15210,15211,15212,15213,15214,15215,15216,15217,15218,15219,15220,15221,15222,15223,15224,15225,15226,15227,15228,15229,15230,15231,15232,15233,15234,15235,15236,15237,15238,15239,15240,15241,15242,15243,15244,15250,15251,15252,15253,15254,15255,15257,15258,15259,15260,15261,15262,15264,15265,15267,15268,15270,15272,15274,15275,15276,15277,15278,15279,15281,15282,15283,15286,15289,15290,15295"
    else:
        # This is for our smell pgh paper
        api_para += "zipcodes=15221,15218,15222,15219,15201,15224,15213,15232,15206,15208,15217,15207,15260,15104"
    if "start_time" in options:
        api_para += "&start_time=" + str(options["start_time"])
    if "end_time" in options:
        api_para += "&end_time=" + str(options["end_time"])
    if "min_smell_value" in options:
        api_para += "&min_smell_value=" + str(options["min_smell_value"])
    if "max_smell_value" in options:
        api_para += "&max_smell_value=" + str(options["max_smell_value"])

    # Load smell reports
    df = pd.read_json(api_url + api_para, convert_dates=False)

    # If empty, return None
    if df.empty:
        return None

    # Wrangle text
    df["smell_description"] = df["smell_description"].replace(np.nan, "").map(removeNonAsciiChars)
    df["feelings_symptoms"] = df["feelings_symptoms"].replace(np.nan, "").map(removeNonAsciiChars)

    # Set index and drop columns
    df.set_index("created_at", inplace=True)
    df.index.names = ["EpochTime"]
    df.drop(["latitude", "longitude"], axis=1, inplace=True)

    # Return
    return df


def plotClusterPairGrid(X, Y, out_p, w, h, title, is_Y_continuous,
    c_ls=((0.5, 0.5, 0.5), (0.2275, 0.298, 0.7529), (0.702, 0.0118, 0.149), (0, 1, 0)), # color
    c_alpha=(0.1, 0.1, 0.2, 0.1), # color opacity
    c_bin=(0, 1), # color is mapped to index [Y<c_bin[0], Y==c_bin[0], Y==c_bin[1], Y>c_bin[1]]
    logger=None):
    """
    Plot a grid of scatter plot pairs in X, with point colors representing binary labels
    """
    if not is_Y_continuous:
        c_idx = [Y<c_bin[0]]
        for k in range(0, len(c_bin)):
            c_idx.append(Y==c_bin[k])
        c_idx.append(Y>c_bin[-1])
        if not (len(c_idx)==len(c_ls)==len(c_alpha)):
            log("Parameter sizes does not match.", logger)
            return

    dot_size = 15
    title_font_size = 24
    label_font_size = 16
    tick_font_size = 16
    alpha = 0.3
    fig = plt.figure(figsize=(6*w, 5*h+1), dpi=150)
    num_cols = X.shape[1]
    cmap = "coolwarm"
    c = 1
    for i in range(0, num_cols-1):
        for j in range(i+1, num_cols):
            plt.subplot(h, w, c)
            if is_Y_continuous:
                plt.scatter(X[:,i], X[:,j], c=Y, s=dot_size, alpha=alpha, cmap=cmap)
            else:
                for k in range(0, len(c_idx)):
                    plt.scatter(X[c_idx[k],i], X[c_idx[k],j], c=c_ls[k], s=dot_size, alpha=c_alpha[k])
            plt.xlabel("Component " + str(i), fontsize=label_font_size)
            plt.ylabel("Component " + str(j), fontsize=label_font_size)
            plt.xticks(fontsize=tick_font_size)
            plt.yticks(fontsize=tick_font_size)
            c += 1
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.suptitle(title, fontsize=title_font_size)
    fig.savefig(out_p)
    fig.clf()
    plt.close()


def isDatetimeObjTzAware(dt):
    """Find if the datetime object is timezone aware"""
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
