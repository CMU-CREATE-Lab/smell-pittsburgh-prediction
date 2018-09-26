# Python helper functions
# Developed by Yen-Chia Hsu, hsu.yenchia@gmail.com
# v1.3

import logging
from os import listdir
from os.path import isfile, join
import os
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
import json
import requests
import uuid
import pytz
import pandas as pd
import numpy as np
from collections import Counter
import time
import re

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

import scipy.ndimage as ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import seaborn as sns

# For Google Analytics
# sudo pip install --upgrade google-api-python-client
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

# Generate a logger for loggin files
def generateLogger(file_name, log_level=logging.INFO, name=str(uuid.uuid4()),
        format="%(asctime)s %(levelname)s %(message)s"):
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

# log and print
def log(msg, logger, level="info"):
    if logger is not None:
        if level == "info":
            logger.info(msg)
        elif level == "error":
            logger.error(msg)
    print msg

# find the least common elements in a given array
def findLeastCommon(arr):
    m = Counter(arr)
    return m.most_common()[-1][0]

# Convert string to float in a safe way
def str2float(string, **options):
    try:
        return float(string)
    except ValueError:
        if "default_value" in options:
            return options["default_value"]
        else:
            return None

# Check if a file exists
def isFileHere(path):
    return os.path.isfile(path)

# Return a list of all files in a folder
def getAllFileNamesInFolder(path):
    return  [f for f in listdir(path) if isfile(join(path, f))]

# Return the root url for ESDR
def esdrRootUrl():
    return "https://esdr.cmucreatelab.org/"

# Return the root url for smell Pittsburgh
def smellPghRootUrl():
    return "http://api.smellpittsburgh.org/"

# Return the root url for smell Pittsburgh Staging
def smellPghStagingRootUrl():
    return "http://staging.api.smellpittsburgh.org/"

# Replace a non-breaking space to a normal space
def sanitizeUnicodeSpace(string):
    type_string = type(string)
    if string is not None and (type_string is str or type_string is unicode):
        return string.replace(u'\xa0', u' ')
    else:
        return None

# Convert a datetime object to epoch time
def datetimeToEpochtime(dt):
    if dt.tzinfo is None:
        dt_utc = dt
    else:
        dt_utc = dt.astimezone(pytz.utc).replace(tzinfo=None)
    epoch_utc = datetime.utcfromtimestamp(0)
    return int((dt_utc - epoch_utc).total_seconds() * 1000)

# Sum up two dictionaries
def dictSum(a, b):
    d = defaultdict(list, deepcopy(a))
    for key, val in b.items():
        d[key] += val
    return dict(d)

# Flip keys and values in a dictionary
def flipDict(a):
    d = defaultdict(list)
    for key, val in a.items():
        d[val] += [key]
    return dict(d)

# Check if a directory exists, if not, create it
def checkAndCreateDir(path):
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Convert the epochtime index in a pandas dataframe to datetime index
def epochtimeIdxToDatetime(df):
    df = df.copy(deep=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    df.index.name = "DateTime"
    return df

# Get the base name of a file path
def getBaseName(path, **options):
    with_extension = options["with_extension"] if "with_extension" in options else False
    do_strip = options["do_strip"] if "do_strip" in options else True
    base_name = os.path.basename(path)
    if with_extension:
        return base_name
    else:
        base_name_no_ext = os.path.splitext(base_name)[0]
        if do_strip:
            return base_name_no_ext.strip()
        else:
            return base_name_no_ext

# Remove all non-ascii characters in the string
def removeNonAsciiChars(str_in):
    if str_in is None:
        return ""
    else:
        return str(unicode(str_in.encode("utf-8"), "ascii", "ignore"))

# Augment time series data
# INPUT:
# - df_X: 2D numpy array with shape (sample_size, feature_size, sequence_length, 1)
# - df_Y: 1D numpy array with shape (sample_size)
def augmentTimeSeriesData(X, Y):
    X_new, Y_new = [], []
    for i in range(X.shape[0]):
        x = X[i,:,:,:]
        X_new.append(x)
        if Y is not None:
            y = Y[i]
            Y_new.append(y)
        # Resize a small part of image
        for kernel_size in [2]:
            for j in range(0, int(np.floor(x.shape[1]-kernel_size+1))):
                # Choose a block in the middle
                before = x[:, 0:j, :]
                block = x[:, j:j+kernel_size, :]
                after = x[:, j+kernel_size:, :]
                s = block.shape
                # Make the block smaller or larger and resize the array
                block_1 = ndimage.zoom(block, (1, 0.5, 1))
                block_2 = ndimage.zoom(block, (1, 2, 1))
                img_1 = np.concatenate((before, block_1, after), axis=1)
                img_2 = np.concatenate((before, block_2, after), axis=1)
                z_1 = float(x.shape[1]) / img_1.shape[1]
                z_2 = float(x.shape[1]) / img_2.shape[1]
                img_1 = ndimage.zoom(img_1, (1, z_1, 1))
                img_2 = ndimage.zoom(img_2, (1, z_2, 1))
                X_new.append(img_1)
                X_new.append(img_2)
                if Y is not None:
                    Y_new.append(y)
                    Y_new.append(y)
        if i % 1000 == 0:
            print "Augment data: i = " + str(i)
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new

# Compute time series batches from a 2D numpy array
# INPUT:
# - X: 2D numpy array with shape (sample_size, feature_size)
# - Y: 2D numpy array with shape (sample_size, label_size)
# - sequence_length: the number of data points to look back
# - index_filter: 1D numpy boolean array, (only process and return the indices that has filter value True)
# OUTPUT:
# - X: 3D numpy array with shape (num_of_mini_batches, sequence_length, feature_size)
# - Y: 1D numpy array with shape (num_of_mini_batches)
def computeTimeSeriesBatches(X, Y, sequence_length, index_filter=None):
    data_X = []
    data_Y = []
    index_all = np.array(range(sequence_length, X.shape[0]))
    if index_filter is not None:
        index_filter = np.array(index_filter[sequence_length:])
        index_all = index_all[index_filter]
    for i in index_all:
        data_X.append(X[i-sequence_length+1:i+1])
        if Y is not None: data_Y.append(Y[i])
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return data_X, data_Y

# Compute a custom metric for evaluating the regression function
# Notice that for daytime cases, the Y arrays may contain NaN
# For each smoke event, the prediction only need to hit the event at some time point
# (for an event from 9am to 11am, good enough if there are at least one predicted event within it)
# Denote T the 1D signal of the true data
# Denote P the 1D signal of the predicted data
# 1. Detect the time intervals in T and P that has values larger than a threshold "thr"
# 2. Merge intervals that are less or equal than "h" hours away from each other
#    (e.g., for h=1, intervals [1,3] and [4,5] need to be merged into [1,5])
# 3. Compute the precision, recall, and f-score for each interval ...
#    ... true positive: for each t in T, if it overlaps with a least one p in P
#    ... false positive: for each p in P, if there is no t in T that overlaps with it
#    ... false negative: for each t in T, if there is no p in P that overlaps with it
def evalEventDetection(Y_true, Y_pred, thr=40, h=1, round_to_decimal=3):
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

# Merge intervals that are less or equal than "h" hours away from each other
# (e.g., for h=1, intervals [1,3] and [4,5] need to be merged into [1,5])
def mergeInterval(intervals, h=1):
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

# Convert a binary array with False and True to intervals
# input = [False, True, True, False, True, False]
# output = [[1,2], [4,4]]
def binary2Interval(Y):
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

# Compute the evaluation result of regression or classification Y=F(X)
# INPUTS:
# - Y_true: the true values of Y
# - Y_pred: the predicted values of Y
# - is_regr: is regression or classification
# OUTPUT:
# - r2: r-squared (for regression)
# - mse: mean squared error (for regression)
# - prf: precision, recall, and f-score (for classification) in pandas dataframe format
# - cm: confusion matrix (for classification) in pandas dataframe format
def computeMetric(Y_true, Y_pred, is_regr, flatten=False, simple=False,
        round_to_decimal=3, labels=[0,1], aggr_axis=False, only_binary=True):
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
        thr = 40 if is_regr else 1
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

# Get wrongly and correctly classified data points
# Only works for classification with label 0 and 1
# INPUT:
# - Y_true: the true values of responses (in numpy format, 1D array)
# - Y_pred: the predicted values of responses (in numpy format, 1D array)
# - X: the predictors (in numpy format, 2D array)
# - col_names: the column names for creating the pandas dataframe
# OUTPUT:
# - true positives (tp), false positives (fp), true negatives (tn), false negatives(fn)
def evaluateData(Y_true, Y_pred, X, col_names=None):
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

# Flatten a pandas dataframe
def flattenDataframe(df):
    df = df.stack()
    idx = df.index.values.tolist()
    for i in range(0, len(idx)):
        idx[i] = ".".join(idx[i])
    val = df.values.tolist()
    return [idx, val]

# Oversample the minority class and undersample the majority class for classification
# INPUT:
# - X: predictors (or features), 2D array
# - Y: responses (or labels), 1D array
def balanceDataset(X, Y):
    # Check types
    X_columns, Y_name = None, None
    if isinstance(X, pd.DataFrame): X_columns = X.columns
    if isinstance(Y, pd.Series): Y_name = Y.name
    
    # Use SMOTE algorithm
    X = np.array(X)
    Y = np.array(Y)
    s = X.shape
    if len(s) > 4:
        print "Unable to resample a dataset with feature having more than 4 dimensions"
        return None, None
    if len(s) == 4: # this is the CNN case (convolutional neural network)
        X = X.reshape(s[0], s[1]*s[2]*s[3])
    model = SMOTE(random_state=0, n_jobs=-1, kind="svm")
    #model = RandomOverSampler(random_state=0)
    X_res, Y_res = model.fit_sample(X, Y)
    if len(s) == 4:
        X_res = X_res.reshape(X_res.shape[0], s[1], s[2], s[3])

    # Revert types
    if X_columns is not None: X_res = pd.DataFrame(data=X_res, columns=X_columns)
    if Y_name is not None: Y_res = pd.Series(data=Y_res, name=Y_name)

    return X_res, Y_res

# Load json file
def loadJson(fpath):
    with open(fpath, "r") as f:
        return json.load(f)

# Save json file
def saveJson(content, fpath):
    with open(fpath, "w") as f:
        json.dump(content, f)

# Save text file
def saveText(content, fpath):
    with open(fpath, "w") as f:
        f.write(content)

# Get the access token from ESDR, need the auth.json file
# See https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md
def getEsdrAccessToken(auth_json_path):
    logger = generateLogger("log.log")
    logger.info("Get access token from ESDR")
    auth_json = loadJson(auth_json_path)
    url = esdrRootUrl() + "oauth/token"
    headers = {"Authorization": "", "Content-Type": "application/json"}
    r = requests.post(url, data=json.dumps(auth_json), headers=headers)
    r_json = r.json()
    if r.status_code is not 200:
        logger.error("ESDR returns: " + json.dumps(r_json) + " when getting the access token")
        return None, None
    else:
        access_token = r_json["access_token"]
        user_id = r_json["userId"]
        logger.debug("ESDR returns: " + json.dumps(r_json) + " when getting the access token")
        logger.info("Receive access token " + access_token)
        logger.info("Receive user ID " + str(user_id))
        return access_token, user_id 

# Upload data to ESDR, use the getEsdrAccessToken() function to get the access_token
# data_json = {
#   "channel_names": ["particle_concentration", "particle_count", "raw_particles", "temperature"],
#   "data": [[1449776044, 0.3, 8.0, 6.0, 2.3], [1449776104, 0.1, 3.0, 0.0, 4.9]]
# }
def uploadDataToEsdr(device_name, data_json, product_id, access_token, **options):
    logger = generateLogger("log.log")

    # Set the header for http request
    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-Type": "application/json"
    }
   
    # Check if the device exists
    logger.info("Try getting the device ID of device name '" + device_name + "'")
    url = esdrRootUrl() + "api/v1/devices?where=name=" + device_name + ",productId=" + str(product_id)
    r = requests.get(url, headers=headers)
    r_json = r.json()
    device_id = None
    if r.status_code is not 200:
        logger.error("ESDR returns: " + json.dumps(r_json) + " when getting the device ID for '" + device_name + "'")
    else:
        logger.debug("ESDR returns: " + json.dumps(r_json) + " when getting the device ID for '" + device_name + "'")
        if r_json["data"]["totalCount"] < 1:
            logger.error("'" + device_name + "' did not exist")
        else:
            device_id = r_json["data"]["rows"][0]["id"]
            logger.info("Receive existing device ID " + str(device_id))

    # Create a device if it does not exist
    if device_id is None:
        logger.info("Create a device for '" + device_name + "'")
        url = esdrRootUrl() + "api/v1/products/" + str(product_id) + "/devices"
        device_json = {
            "name": device_name,
            "serialNumber": options["serialNumber"] if "serialNumber" in options else str(uuid.uuid4())
        }
        r = requests.post(url, data=json.dumps(device_json), headers=headers)
        r_json = r.json()
        if r.status_code is not 201:
            logger.error("ESDR returns: " + json.dumps(r_json) + " when creating a device for '" + device_name + "'")
            return None
        else:
            logger.debug("ESDR returns: " + json.dumps(r_json) + " when creating a device for '" + device_name + "'")
            device_id = r_json["data"]["id"]
            logger.info("Create new device ID " + str(device_id))

    # Check if a feed exists for the device
    logger.info("Get feed ID for '" + device_name + "'")
    url = esdrRootUrl() + "api/v1/feeds?where=deviceId=" + str(device_id)
    r = requests.get(url, headers=headers)
    r_json = r.json()
    feed_id = None
    api_key = None
    api_key_read_only = None
    if r.status_code is not 200:
        logger.debug("ESDR returns: " + json.dumps(r_json) + " when getting the feed ID")
    else:
        logger.debug("ESDR returns: " + json.dumps(r_json) + " when getting the feed ID")
        if r_json["data"]["totalCount"] < 1:
            logger.info("No feed ID exists for device " + str(device_id))
        else:
            row = r_json["data"]["rows"][0]
            feed_id = row["id"]
            api_key = row["apiKey"]
            api_key_read_only = row["apiKeyReadOnly"]
            logger.info("Receive existing feed ID " + str(feed_id))

    # Create a feed if no feed ID exists
    if feed_id is None:
        logger.info("Create a feed for '" + device_name + "'")
        url = esdrRootUrl() + "api/v1/devices/" + str(device_id) + "/feeds"
        feed_json = {
            "name": device_name,
            "exposure": options["exposure"] if "exposure" in options else "virtual",
            "isPublic": options["isPublic"] if "isPublic" in options else 0,
            "isMobile": options["isMobile"] if "isMobile" in options else 0,
            "latitude": options["latitude"] if "latitude" in options else None,
            "longitude": options["longitude"] if "longitude" in options else None
        }
        r = requests.post(url, data=json.dumps(feed_json), headers=headers)
        r_json = r.json()
        if r.status_code is not 201:
            logger.error("ESDR returns: " + json.dumps(r_json) + " when creating a feed")
            return None
        else:
            logger.info("ESDR returns: " + json.dumps(r_json) + " when creating a feed")
            feed_id = r_json["data"]["id"]
            api_key = r_json["data"]["apiKey"]
            api_key_read_only = r_json["data"]["apiKeyReadOnly"]
            logger.info("Create new feed ID " + str(feed_id))
    
    # Upload Speck data to ESDR
    logger.info("Upload sensor data for '" + device_name + "'")
    url = esdrRootUrl() + "api/v1/feeds/" + str(feed_id)
    r = requests.put(url, data=json.dumps(data_json), headers=headers)
    r_json = r.json()
    if r.status_code is not 200:
        logger.error("ESDR returns: " + json.dumps(r_json) + " when uploading data")
        return None
    else:
        logger.debug("ESDR returns: " + json.dumps(r_json) + " when uploading data")

    # Return a list of information for getting data from ESDR
    logger.info("Data uploaded")
    return [device_id, feed_id, api_key, api_key_read_only]

# Get data from ESDR
# source = [
#    [{"feed": 27, "channel": "NO_PPB"}],
#    [{"feed": 1, "channel": "PM25B_UG_M3"}, {"feed": 1, "channel": "PM25T_UG_M3"}]
# ]
# if source = [[A,B],[C]], this means that A and B will be merged 
# start_time: starting epochtime in seconds
# end_time: ending epochtime in seconds
def getEsdrData(source, **options):
    print "Get ESDR data..."
    
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

# Get smell reports data from smell PGH
def getSmellReports(**options):
    print "Get smell reports..."

    # Url
    api_url = smellPghRootUrl() + "api/v1/"
    api_para = "smell_reports?"
    if "allegheny_county_only" in options and options["allegheny_county_only"] == True:
        api_para += "allegheny_county_only=True"
    else:
        api_para += "zipcodes=15221,15218,15222,15219,15201,15224,15213,15232,15206,15208,15217,15207,15260,15104"
    api_para += "&prediction_query=true" # this returns the user hash for identifying unique users
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

# Get Google Analytics data, need to obtain the client secret from Google API console first
# see https://developers.google.com/analytics/devguides/config/mgmt/v3/authorization
def getGA(
    in_path="client_secrets.json", # client secret json file
    out_path="GA/", # the path to store CSV files
    date_info=[{"startDate":"2017-12-11", "endDate":"2017-12-12"},
        {"startDate":"2018-01-10", "endDate":"2018-01-11"}],
    view_id="ga:131141811", # obtain this ID from Google Analytics dashboard
    metrics=[{"expression": "ga:pageviews"}],
    metrics_col_names=["Pageviews"], # pretty names for metrics
    dimensions=[{"name": "ga:dimension1"},
        {"name": "ga:dimension2"},
        {"name": "ga:dimension4"},
        {"name": "ga:dimension5"},
        {"name": "ga:eventCategory"}],
    dimensions_col_names=["User ID",
        "Client ID",
        "Hit Timestamp",
        "Data Timestamp",
        "Event Category"] # pretty names for dimensions
    ):
    
    print "Get Google Analytics..."

    SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
    KEY_FILE_LOCATION = in_path 

    # Build the service object
    credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE_LOCATION, SCOPES)
    analytics = build('analytics', 'v4', credentials=credentials)

    # Check if the directory exists
    checkAndCreateDir(out_path)

    # Use the Analytics Service Object to query the Analytics Reporting API V4
    for k in date_info:
        info = {
            "reportRequests": [
                {
                    "viewId": view_id,
                    "dateRanges": [k],
                    "metrics": metrics,
                    "dimensions": dimensions,
                    "includeEmptyRows": True,
                    "pageSize": 10000
                }
            ] 
        }
        r = analytics.reports().batchGet(body=info).execute()
        # Parse rows and put them into a csv file
        file_name = "tracker-from-" + k["startDate"] + "-to-" + k["endDate"] + ".csv"
        with open(out_path + file_name, 'w') as out_file:
            out_file.write(",".join(dimensions_col_names) + "," + ",".join(metrics_col_names) + "\n")
            if "rows" not in r["reports"][0]["data"]:
                print "Error: no rows"
            else:
                rows = r["reports"][0]["data"]["rows"]
                print str(len(rows)) + " rows from " + k["startDate"] + " to " + k["endDate"]
                for p in rows:
                    line = ",".join([",".join(p["dimensions"]), p["metrics"][0]["values"][0]])
                    out_file.write(line + "\n")
                print "Google Analytics file created at " + out_path + file_name
        # Pause for some time
        time.sleep(1)

# Plot a grid of scatter plot pairs in X, with point colors representing binary labels
def plotClusterPairGrid(X, Y, out_p, w, h, title, is_Y_continuous,
    c_ls=[(0.5, 0.5, 0.5), (0.2275, 0.298, 0.7529), (0.702, 0.0118, 0.149), (0, 1, 0)], # color
    c_alpha=[0.1, 0.1, 0.2, 0.1], # color opacity
    c_bin=[0, 1], # color is mapped to index [Y<c_bin[0], Y==c_bin[0], Y==c_bin[1], Y>c_bin[1]]
    logger=None):

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

# Plot bar charts
# Note that x, y, title are all arrays
def plotBar(x, y, h, w, title, out_p):
    fig = plt.figure(figsize=(w*12, h*1.5))
    c = 1
    for i in range(0, h*w):
        ax = plt.subplot(h, w, i+1)
        plt.title(title[i], fontsize=14)
        plt.bar(range(0,len(x[i])), y[i], 0.6, color=(0.4,0.4,0.4), align="center")
        plt.xticks(range(0,len(x[i])), x[i])
        #for j in range(0, len(y[i])): ax.text(j, y[i][j], int(y[i][j]), color=(0.2,0.2,0.2), ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_p, dpi=150)
    fig.clf()
    plt.close()

# The input is a pandas time series
# replace is a dictionary, key is the original word, value is the replaced word
# exclude is an array, indicating words that we do not want
def textAnalysis(s, exclude=[], replace={}):
    sw = stopwords.words("english")
    s = s.str.lower()
    s = s.dropna().values
    s = " ".join(map(str, s))
    s = re.sub("[^0-9a-zA-Z]+", " ", s)
    s = s.split(" ")
    wnl = WordNetLemmatizer() # for lemmatisation
    res = []
    len_exclude = len(exclude)
    for k in s:
        k = wnl.lemmatize(wnl.lemmatize(k), "v")
        if k is not None and k != "" and k not in sw and not hasNumbers(k):
            if len_exclude > 0 and k in exclude: continue
            if k in replace: k = replace[k]
            res.append(k)
    return res

def dateIndexToMonthYear(index):
    month_txt = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return map("\n".join, zip(month_txt[index.month.values - 1], index.year.astype(str).values))

# This function converts a date object to epoch time
def dateToEpochtime(d):
    dt = datetime.combine(d, datetime.min.time())
    return datetimeToEpochtime(dt)

# This function groups a numpy array containing epoch time by date
def groupTime(epochtime, unit):
    raw_time = pd.to_datetime(epochtime, unit=unit)

    # Convert to US Eastern time
    raw_time = raw_time.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    #raw_time = raw_time.tz_localize(pytz.timezone("US/Eastern"), ambiguous="infer")

    # Group by date
    raw_time_df = pd.DataFrame(raw_time)
    raw_time_df["date"] = raw_time_df[0].apply(lambda x: x.date())
    gb = raw_time_df.groupby("date").groups

    # Format data
    data = {"data": [], "row_names": [], "index": [], "epochtime": []}
    sort_base = []
    for key in gb.keys():
        if key.year != 2018: continue
        sort_base.append(key)
        data["epochtime"].append(int(key.strftime('%s')))
        data["row_names"].append(key.strftime("%d %b %Y (%a)"))
        data["data"].append(epochtime[gb[key]].tolist())
        data["index"].append(gb[key].tolist())

    # Sort by epoch time
    idx = np.array(sort_base).argsort()
    data["epochtime"] = np.array(data["epochtime"])[idx].tolist()
    data["row_names"] = np.array(data["row_names"])[idx].tolist()
    data["data"] = np.array(data["data"])[idx].tolist()

    return data

# This function aggregates a numpy array containing epoch time
def aggregateTime(epochtime, unit, resample_method, format_method, **options):
    raw_time = pd.to_datetime(epochtime, unit=unit)

    # Convert to US Eastern time
    raw_time = raw_time.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    #raw_time = raw_time.tz_localize(pytz.timezone("US/Eastern"), ambiguous="infer")

    if "only_count_unique" in options:
        raw_time_series = pd.Series(options["only_count_unique"], index=raw_time)
        aggr = raw_time_series.resample(resample_method).apply(lambda x: x.nunique())
    else:
        raw_time_series = pd.Series(np.ones(len(epochtime), dtype=np.int), index=raw_time)
        aggr = raw_time_series.resample(resample_method).count()

    # Reindex the bins to involve the entire day
    d = raw_time[0].date()
    all_day_idx = pd.date_range(d, periods=24, freq=resample_method, tz="US/Eastern")
    aggr = aggr.reindex(all_day_idx, fill_value=0)

    # Format data
    keys = [d.strftime(format_method) for d in aggr.index]
    vals = aggr.values

    if "remove_zero_val" in options and options["remove_zero_val"] == True:
        idx = np.nonzero(vals)[0]
        keys = np.array(keys)[idx].tolist()
        vals = vals[idx]

    return {"data": map(list, zip(keys, vals.tolist())), "val_argmax": vals.argmax(), "val_max": vals.max()}

# Count the word frequency of a word array
def countWords(A):
    return Counter(A)

# Find if a string contains numbers
def hasNumbers(str):
    return bool(re.search(r'\d', str))

def plotScatter(df, x, y, title, out_p):
    ax = df.plot.scatter(x=x, y=y, figsize=(6, 6))
    fig = ax.get_figure()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_p, dpi=150)
    fig.clf()
    plt.close()

# Plot line charts
# df_all and title_all are all arrays
def plotLineCharts(df_all, title_all, h, w, out_p):
    fig = plt.figure(figsize=(w*12, h*2))
    c = 1
    for i in range(0, h*w):
        ax = plt.subplot(h, w, i+1)
        plt.title(title_all[i], fontsize=14)
        df_all[i].plot(ax=ax)
        #for j in range(0, len(y[i])): ax.text(j, y[i][j], int(y[i][j]), color=(0.2,0.2,0.2), ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_p, dpi=150)
    fig.clf()
    plt.close()

# Plot box charts
# df_all and title_all are all arrays
def plotBoxCharts(df_all, title_all, h, w, out_p):
    fig = plt.figure(figsize=(w*4, h*4))
    medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
    meanpointprops = dict(marker="D", markeredgecolor="firebrick", markerfacecolor="firebrick", markersize=7)
    c = 1
    for i in range(0, h*w):
        ax = plt.subplot(h, w, i+1)
        plt.title(title_all[i], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.boxplot(df_all[i], sym="", widths=0.6, labels=[df.name for df in df_all[i]],
            medianprops=medianprops, showmeans=True, meanprops=meanpointprops)

    plt.tight_layout()
    fig.savefig(out_p, dpi=150)
    fig.clf()
    plt.close()

# Find if the datetime object is timezone aware
def isDatetimeObjTzAware(dt):
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
