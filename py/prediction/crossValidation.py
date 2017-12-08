import numpy as npc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from trainModel import *
import os
from util import *
from datetime import datetime
from selectFeatures import *
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

# Cross validation
def crossValidation(
    df_X=None, # features
    df_Y=None, # labels
    in_p=None, # input path of features and labels
    out_p_root=None, # root directory for outputing files
    method="SVM", # see trainModel.py
    is_regr=False, # regression or classification,
    balance=False, # oversample or undersample training dataset
    only_day_time=False, # only use daytime data for training or not
    sequence_length=8, # length of data points (hours) to look back (only work for CRNN)
    num_folds=61, # number of folds for validation
    skip_folds=45, # skip first n folds (not enough data for training) 36, 45
    augment_data=True, # augment data or not
    select_feat=True, # select features or not
    logger=None):

    log("================================================================================", logger)
    log("================================================================================", logger)
    log("Inference using method = " + method, logger)
    log("balance = " + str(balance), logger)
    log("only_day_time = " + str(only_day_time), logger)
    log("is_regr = " + str(is_regr), logger)
    log("num_folds = " + str(num_folds), logger)
    log("skip_folds = " + str(skip_folds), logger)
    log("augment_data = " + str(augment_data), logger)
    log("select_feat = " + str(select_feat), logger)

    # Read features
    if df_X is None or df_Y is None:
        if in_p is not None:
            df_X = pd.read_csv(in_p[0])
            df_Y = pd.read_csv(in_p[1])
        else:
            log("ERROR: no input data, return None.")
            return None

    # Check if using day time only
    daytime_idx = None
    hd_start = 6
    hd_end = 18
    if only_day_time:
        df_hd = df_X["HourOfDay"]
        daytime_idx = ((df_hd>=hd_start)&(df_hd<=hd_end)).values

    # Convert to time series batches if using models that have time-series structure (e.g. RNN)
    # Dimension of X for RNN is (batch_size, sequence_length, feature_size)
    # Dimension of X for CNN is (batch_size, feature_size, sequence_length, 1)
    if "CRNN" in method or "ANCNN" in method:
        log("sequence_length = " + str(sequence_length), logger) # the number of data points to look back
        # Compute batches
        X, Y = computeTimeSeriesBatches(df_X.values, df_Y.values, sequence_length, index_filter=daytime_idx)
        # For CNN, we want to do 1D convolution on time-series images
        # Each image has dimention (sequence_length * 1), and has channel size equal to feature_size
        X = np.expand_dims(X.transpose(0,2,1), axis=3)
        # save the original data for pre-training an autoencoder
        if "ANCNN" in method:
            X_pretrain, _  = computeTimeSeriesBatches(df_X.values, None, sequence_length)
            X_pretrain = np.expand_dims(X_pretrain.transpose(0,2,1), axis=3)
    else:
        # For non-CNN methods, we need to perform feature selection for each cross validation fold
        # So do not convert the dataframe to numpy format yet
        if only_day_time:
            df_X, df_Y = df_X[daytime_idx], df_Y[daytime_idx]
        X, Y = df_X, df_Y

    # Validation folds
    # Notice that this is time-series prediction, we cannot use traditional cross-validation folds
    if only_day_time:
        tscv = TimeSeriesSplit(n_splits=num_folds, max_train_size=3000)
    else:
        tscv = TimeSeriesSplit(n_splits=num_folds, max_train_size=6000)

    # For pretraining the autoencoder, we always want to use all data
    if "ANCNN" in method:
        tscv_pretrain = TimeSeriesSplit(n_splits=num_folds, max_train_size=6000)
        tscv_pretrain_split = list(tscv_pretrain.split(X_pretrain))

    # Perform cross validation
    counter = 0
    train_all = {"X": [], "Y": [], "Y_pred": [], "Y_score": []}
    test_all = {"X": [], "Y": [], "Y_pred": [], "Y_score": []}
    metric_all = {"train": [], "test": []}
    for train_idx, test_idx in tscv.split(X, Y):
        if counter < skip_folds:
            counter += 1
            continue
        log("--------------------------------------------------------------", logger)
        log("Processing fold " + str(counter + 1) + ":", logger)
        # For non-CNN methods, we need to do feature selection and convert data to numpy format
        # For CNN methods, we need to augment time series data
        if len(X.shape) == 2:
            X_train, Y_train, X_test, Y_test = X.iloc[train_idx], Y.iloc[train_idx], X.iloc[test_idx], Y.iloc[test_idx]
            if select_feat:
                X_train, Y_train = selectFeatures(X_train, Y_train, is_regr=is_regr, logger=logger)
            X_test = X_test[X_train.columns]
            X_train, Y_train, X_test, Y_test = X_train.values, Y_train.values, X_test.values, Y_test.values
            if augment_data:
                log("Data augmentation is ignored for non-CNN methods", logger)
        else:
            X_train, Y_train = X[train_idx], Y[train_idx]
            if augment_data:
                log("Augment time series data...", logger)
                X_train, Y_train = augmentTimeSeriesData(X_train, Y_train)
            X_test, Y_test = X[test_idx], Y[test_idx]
        # Prepare training and testing set
        train = {"X": X_train, "Y": Y_train, "Y_pred": None}
        test = {"X": X_test, "Y": Y_test, "Y_pred": None}
        if "ANCNN" in method:
            tr_idx, te_idx = tscv_pretrain_split[counter]
            train["X_pretrain"] = X_pretrain[tr_idx]
            test["X_pretrain"] = X_pretrain[te_idx]
            #if augment_data:
            #    log("Augment time series data (for pre-training)...", logger)
            #    train["X_pretrain"], _ = augmentTimeSeriesData(train["X_pretrain"] , None)
        # Train model
        model = trainModel(train, test=test, method=method, is_regr=is_regr, logger=logger, balance=balance)
        # Evaluate model
        test["Y_pred"] = model.predict(test["X"])
        train["Y_pred"] = model.predict(train["X"])
        test_all["Y"].append(test["Y"]) 
        test_all["Y_pred"].append(test["Y_pred"])
        metric_i_test = computeMetric(test["Y"], test["Y_pred"], is_regr, aggr_axis=True)
        metric_all["test"].append(metric_i_test)
        train_all["Y"].append(train["Y"])
        train_all["Y_pred"].append(train["Y_pred"])
        metric_i_train = computeMetric(train["Y"], train["Y_pred"], is_regr, aggr_axis=True)
        metric_all["train"].append(metric_i_train)
        if not is_regr:
            test_all["Y_score"].append(model.predict_proba(test["X"]))
            train_all["Y_score"].append(model.predict_proba(train["X"]))
        if len(X.shape) == 2:
            train_all["X"].append(train["X"])
            test_all["X"].append(test["X"])
        else: # CNN case
            train_all["X"].append(train["X"][:,:,-1,:].squeeze())
            test_all["X"].append(test["X"][:,:,-1,:].squeeze())
        # Print result
        if not ("CRNN" in method or "CRMANN" in method):
            for m in metric_i_train:
                log("Training metrics: " + m, logger)
                log(metric_i_train[m], logger)
            for m in metric_i_test:
                log("Testing metrics: " + m, logger)
                log(metric_i_test[m], logger)
        counter += 1

    # Merge all evaluation data
    test_all["Y"] = np.concatenate(test_all["Y"], axis=0)
    test_all["Y_pred"] = np.concatenate(test_all["Y_pred"], axis=0)
    test_all["X"] = np.concatenate(test_all["X"], axis=0)
    train_all["Y"] = np.concatenate(train_all["Y"], axis=0)
    train_all["Y_pred"] = np.concatenate(train_all["Y_pred"], axis=0)
    train_all["X"] = np.concatenate(train_all["X"], axis=0)
    if not is_regr:
        train_all["Y_score"] = np.concatenate(train_all["Y_score"], axis=0)
        test_all["Y_score"] = np.concatenate(test_all["Y_score"], axis=0)

    # Evaluation
    log("================================================================================", logger)
    for i in range(0, len(metric_all["test"])):
        log("--------------------------------------------------------------", logger)
        log("For fold " + str(skip_folds+i+1) + ":", logger)
        for m in metric_all["train"][i]:
            log("Training metrics: " + m, logger)
            log(metric_all["train"][i][m], logger)
        for m in metric_all["test"][i]:
            log("Testing metrics: " + m, logger)
            log(metric_all["test"][i][m], logger)

    # Get true positives, false positives, true negatives, false negatives
    if not is_regr:
        hd = "HourOfDay"
        dw = "DayOfWeek"
        # For training data
        eva = evaluateData(train_all["Y"], train_all["Y_pred"], train_all["X"][:,-2:], col_names=df_X.columns[-2:].values)
        log("--------------------------------------------------------------", logger)
        log("True positive counts (training data):", logger)
        log(eva["tp"][hd].value_counts(), logger)
        log(eva["tp"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("True negative counts (training data):", logger)
        log(eva["tn"][hd].value_counts(), logger)
        log(eva["tn"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("False positive counts (training data):", logger)
        log(eva["fp"][hd].value_counts(), logger)
        log(eva["fp"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("False negative counts (training data):", logger)
        log(eva["fn"][hd].value_counts(), logger)
        log(eva["fn"][dw].value_counts(), logger)
        # For testing data
        eva = evaluateData(test_all["Y"], test_all["Y_pred"], test_all["X"][:,-2:], col_names=df_X.columns[-2:].values)
        log("--------------------------------------------------------------", logger)
        log("True positive counts (testing data):", logger)
        log(eva["tp"][hd].value_counts(), logger)
        log(eva["tp"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("True negative counts (testing data):", logger)
        log(eva["tn"][hd].value_counts(), logger)
        log(eva["tn"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("False positive counts (testing data):", logger)
        log(eva["fp"][hd].value_counts(), logger)
        log(eva["fp"][dw].value_counts(), logger)
        log("--------------------------------------------------------------", logger)
        log("False negative counts (testing data):", logger)
        log(eva["fn"][hd].value_counts(), logger)
        log(eva["fn"][dw].value_counts(), logger)

    # Evaluation for all data
    metric = {"train": [], "test": []}
    log("--------------------------------------------------------------", logger)
    log("For all training data:", logger)
    metric["train"] = computeMetric(train_all["Y"], train_all["Y_pred"], is_regr, aggr_axis=True)
    for m in metric["train"]:
        log("Metric: " + m, logger)
        log(metric["train"][m], logger)
    log("--------------------------------------------------------------", logger)
    log("For all testing data:", logger)
    metric["test"] = computeMetric(test_all["Y"], test_all["Y_pred"], is_regr, aggr_axis=True)
    for m in metric["test"]:
        log("Metric: " + m, logger)
        log(metric["test"][m], logger)

    # Evaluation for all data at daytime only
    metric_dt = {"train": [], "test": []}
    hd_val_train = train_all["X"][:,-1:].squeeze()
    hd_val_test = test_all["X"][:,-1:].squeeze()
    dt_idx_tr = (hd_val_train>=hd_start)&(hd_val_train<=hd_end)
    dt_idx_te = (hd_val_test>=hd_start)&(hd_val_test<=hd_end)
    log("--------------------------------------------------------------", logger)
    log("(Daytime only) For all training data:", logger)
    metric_dt["train"] = computeMetric(train_all["Y"][dt_idx_tr], train_all["Y_pred"][dt_idx_tr], is_regr, aggr_axis=True)
    for m in metric_dt["train"]:
        log("Metric: " + m, logger)
        log(metric_dt["train"][m], logger)
    log("--------------------------------------------------------------", logger)
    log("(Daytime only) For all testing data:", logger)
    metric_dt["test"] = computeMetric(test_all["Y"][dt_idx_te], test_all["Y_pred"][dt_idx_te], is_regr, aggr_axis=True)
    for m in metric_dt["test"]:
        log("Metric: " + m, logger)
        log(metric_dt["test"][m], logger)
    
    # Save plot
    if out_p_root is not None:
        if is_regr:
            out_p = out_p_root + "result/regression/"
            checkAndCreateDir(out_p)
            r2 = metric["test"]["r2"]
            mse = metric["test"]["mse"]
            Y_true = test_all["Y"]
            Y_pred = test_all["Y_pred"]
            predictionPlot(method, r2, mse, Y_true, Y_pred, out_p)
            residualPlot(method, r2, mse, Y_true, Y_pred, out_p)
            timeSeriesPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx_te)
        else:
            out_p = out_p_root + "result/classification/"
            checkAndCreateDir(out_p)
            metric["test"]["prf"].to_csv(out_p + method + "_clas_prf.csv")
            metric["test"]["cm"].to_csv(out_p + method + "_clas_cm.csv")
            Y_true = test_all["Y"]
            Y_score = test_all["Y_score"]
            prPlot(method, Y_true, Y_score, out_p)
            rocPlot(method, Y_true, Y_score, out_p)

def rocPlot(method, Y_true, Y_score, out_p):
    # Precision vs recall
    fig = plt.figure(figsize=(8, 8), dpi=150)
    fpr, tpr, threshold = roc_curve(Y_true, Y_score[:, -1])
    plt.step(fpr, tpr, "o", alpha=0.2, markersize=0, color=(0,0,1), where="post")
    plt.fill_between(fpr, tpr, alpha=0.2, color=(0,0,1), step="post")
    plt.plot([0,1], [0,1], "--", alpha=0.8, markersize=0, color=(0,0,1), lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_roc.png")
    # Precision vs thresholds
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(threshold, fpr, "-o", alpha=0.8, markersize=5, color=(0,0,1), lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("False Positive Rate")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_fpr_thr.png")
    # Recall vs thresholds
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(threshold, tpr, "-o", alpha=0.8, markersize=5, color=(1,0,0), lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("True Positive Rate")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_tpr_thr.png")

def prPlot(method, Y_true, Y_score, out_p):
    # Precision vs recall
    fig = plt.figure(figsize=(8, 8), dpi=150)
    precision, recall, threshold = precision_recall_curve(Y_true, Y_score[:, -1])
    plt.step(recall, precision, "o", alpha=0.2, markersize=0, color=(0,0,1), where="post")
    plt.fill_between(recall, precision, alpha=0.2, color=(0,0,1), step="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_pr.png")
    # Precision vs thresholds
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(threshold, precision[:-1], "-o", alpha=0.8, markersize=5, color=(0,0,1), lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_p_thr.png")
    # Recall vs thresholds
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(threshold, recall[:-1], "-o", alpha=0.8, markersize=5, color=(1,0,0), lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Method=" + method, fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_clas_r_thr.png")

def timeSeriesPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx):
    # Time vs Prediction (or True)
    fig = plt.figure(figsize=(200, 8), dpi=150)
    plt.plot(range(0, len(Y_true)), Y_true, "-o", alpha=0.8, markersize=3, color=(0,0,1), lw=1)
    plt.plot(range(0, len(Y_pred)), Y_pred, "-o", alpha=0.8, markersize=3, color=(1,0,0), lw=1)
    for i in range(0, len(dt_idx)):
        if dt_idx[i] == False:
            plt.axvspan(i-0.5, i+0.5, facecolor="0.2", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Smell value (blue=true, red=pred)")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    Y = list(Y_true) + list(Y_pred)
    plt.xlim(-1, len(Y_true)+1)
    plt.ylim(np.amin(Y)-0.5, np.amax(Y)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_time_true_pred.png")

def residualPlot(method, r2, mse, Y_true, Y_pred, out_p):
    # Histogram of Residual
    fig = plt.figure(figsize=(8, 8), dpi=150)
    res = Y_true - Y_pred
    plt.hist(res, bins=100)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    plt.xlim(np.amin(res)-1, np.amax(res)+1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_hist.png")
    # True vs Residual
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(Y_true, res, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("True smell value")
    plt.ylabel("Residual")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    plt.xlim(np.amin(Y_true)-0.5, np.amax(Y_true)+0.5)
    plt.ylim(np.amin(res)-0.5, np.amax(res)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_true.png")

def predictionPlot(method, r2, mse, Y_true, Y_pred, out_p):
    # True vs Prediction
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(Y_true, Y_pred, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("True smell value")
    plt.ylabel("Predicted smell value")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    plt.xlim(np.amin(Y_true)-0.5, np.amax(Y_true)+0.5)
    plt.ylim(np.amin(Y_pred)-0.5, np.amax(Y_pred)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_r2.png")
