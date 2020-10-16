import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from trainModel import trainModel
from util import log, checkAndCreateDir, computeMetric, evaluateData
from selectFeatures import selectFeatures
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import gc
import warnings
warnings.filterwarnings("ignore") # "error", "ignore", "always", "default", "module", or "once"


def crossValidation(
    df_X=None, # features
    df_Y=None, # labels
    df_C=None, # crowd feature, total smell values for the previous hour
    in_p=None, # input path of features and labels
    out_p_root=None, # root directory for outputing files
    method="ET", # see trainModel.py
    is_regr=False, # regression or classification,
    only_day_time=False, # only use daytime data for training or not
    num_folds=144, # number of folds for validation
    skip_folds=48, # skip first n folds (not enough data for training) 48
    select_feat=False, # False means do not select features, int means select n number of features
    hd_start=5, # definition of the starting time of "daytime", e.g. 6 means 6am
    hd_end=11, # definition of the ending time of "daytime", e.g. 14 means 2pm
    train_size=8000, # number of samples for training data
    event_thr=40, # the threshold of smell values to define an event, only used for is_regr=True
    pos_out=True, # always output positive values for regression or not
    logger=None):

    log("================================================================================", logger)
    log("================================================================================", logger)
    log("Cross validation using method = " + method, logger)
    log("only_day_time = " + str(only_day_time), logger)
    log("is_regr = " + str(is_regr), logger)
    log("num_folds = " + str(num_folds), logger)
    log("skip_folds = " + str(skip_folds), logger)
    log("select_feat = " + str(select_feat), logger)

    # Ouput path
    if out_p_root is not None:
        part = "regression" if is_regr else "classification"
        out_p = out_p_root + "result/" + part + "/method_" + method + "/"
        checkAndCreateDir(out_p)

    # Read features
    if df_X is None or df_Y is None:
        if in_p is not None:
            df_X = pd.read_csv(in_p[0])
            df_Y = pd.read_csv(in_p[1])
            df_C = pd.read_csv(in_p[2])
        else:
            log("ERROR: no input data, return None.")
            return None

    # Check if using day time only
    daytime_idx = None
    if only_day_time:
        df_hd = df_X["HourOfDay"]
        daytime_idx = ((df_hd>=hd_start)&(df_hd<=hd_end)).values

    # Perform feature selection for each cross validation fold
    if only_day_time:
        df_X, df_Y, df_C = df_X[daytime_idx], df_Y[daytime_idx], df_C[daytime_idx]
    X, Y, C = df_X, df_Y, df_C

    # Validation folds
    # Notice that this is time-series prediction, we cannot use traditional cross-validation folds
    if only_day_time:
        tscv = TimeSeriesSplit(n_splits=num_folds, max_train_size=train_size/2)
    else:
        tscv = TimeSeriesSplit(n_splits=num_folds, max_train_size=train_size)

    # Perform cross validation
    fold = 0
    train_all = {"X": [], "Y": [], "Y_pred": [], "Y_score": [], "C": []}
    test_all = {"X": [], "Y": [], "Y_pred": [], "Y_score": [], "C": []}
    metric_all = {"train": [], "test": []}
    for train_idx, test_idx in tscv.split(X, Y):
        if fold < skip_folds:
            fold += 1
            continue
        fold += 1
        log("--------------------------------------------------------------", logger)
        log("Processing fold " + str(fold) + " with method " + str(method) + ":", logger)
        # Do feature selection and convert data to numpy format
        X_train, Y_train, C_train = X.iloc[train_idx], Y.iloc[train_idx], C.iloc[train_idx]
        X_test, Y_test, C_test = X.iloc[test_idx], Y.iloc[test_idx], C.iloc[test_idx]
        if select_feat:
            X_train, Y_train = selectFeatures(X_train, Y_train, is_regr=is_regr, logger=logger, num_feat_rfe=select_feat)
        X_test = X_test[X_train.columns]
        X_train, Y_train, C_train = X_train.values, Y_train.values, C_train.values
        X_test, Y_test, C_test = X_test.values, Y_test.values, C_test.values
        # Prepare training and testing set
        train = {"X": X_train, "Y": Y_train, "Y_pred": None, "C": C_train}
        test = {"X": X_test, "Y": Y_test, "Y_pred": None, "C": C_test}
        # Train model
        model = trainModel(train, test=test, method=method, is_regr=is_regr, logger=logger)
        # Evaluate model
        if method in ["HCR", "CR"]: # the hybrid crowd classifier requires Y
            test["Y_pred"] = model.predict(test["X"], test["C"])
            train["Y_pred"] = model.predict(train["X"], train["C"])
        else:
            test["Y_pred"] = model.predict(test["X"])
            train["Y_pred"] = model.predict(train["X"])
        # For regression, check if want to always output positive values
        if is_regr and pos_out:
            test["Y_pred"][test["Y_pred"]<0] = 0
            train["Y_pred"][train["Y_pred"]<0] = 0
        test_all["Y"].append(test["Y"])
        test_all["Y_pred"].append(test["Y_pred"])
        metric_i_test = computeMetric(test["Y"], test["Y_pred"], is_regr, aggr_axis=True, event_thr=event_thr)
        metric_all["test"].append(metric_i_test)
        train_all["Y"].append(train["Y"])
        train_all["Y_pred"].append(train["Y_pred"])
        metric_i_train = computeMetric(train["Y"], train["Y_pred"], is_regr, aggr_axis=True, event_thr=event_thr)
        metric_all["train"].append(metric_i_train)
        if not is_regr:
            if method in ["HCR", "CR"]: # the hybrid crowd classifier requires Y
                test_all["Y_score"].append(model.predict_proba(test["X"], test["C"]))
                train_all["Y_score"].append(model.predict_proba(train["X"], train["C"]))
            else:
                test_all["Y_score"].append(model.predict_proba(test["X"]))
                train_all["Y_score"].append(model.predict_proba(train["X"]))
        train_all["X"].append(train["X"])
        test_all["X"].append(test["X"])
        # Print result
        for m in metric_i_train:
            log("Training metrics: " + m, logger)
            log(metric_i_train[m], logger)
        for m in metric_i_test:
            log("Testing metrics: " + m, logger)
            log(metric_i_test[m], logger)
        # Plot graph
        log("Print time series plots for fold " + str(fold), logger)
        hd_val_test = test["X"][:,-1:].squeeze()
        dt_idx_te = (hd_val_test>=hd_start)&(hd_val_test<=hd_end)
        timeSeriesPlot(method, test["Y"], test["Y_pred"], out_p, dt_idx_te, fold=fold)

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
    metric["train"] = computeMetric(train_all["Y"], train_all["Y_pred"], is_regr, aggr_axis=True, event_thr=event_thr)
    for m in metric["train"]:
        log("Metric: " + m, logger)
        log(metric["train"][m], logger)
    log("--------------------------------------------------------------", logger)
    log("For all testing data:", logger)
    metric["test"] = computeMetric(test_all["Y"], test_all["Y_pred"], is_regr, aggr_axis=True, event_thr=event_thr)
    for m in metric["test"]:
        log("Metric: " + m, logger)
        log(metric["test"][m], logger)

    # Evaluation for all data at daytime only
    metric_dt = {"train": [], "test": []}
    hd_val_train = train_all["X"][:,-1:].squeeze()
    hd_val_test = test_all["X"][:,-1:].squeeze()
    dt_idx_tr = (hd_val_train>=hd_start)&(hd_val_train<=hd_end)
    dt_idx_te = (hd_val_test>=hd_start)&(hd_val_test<=hd_end)
    train_all_dt = copy.deepcopy(train_all)
    if not is_regr:
        train_all_dt["Y"] = train_all_dt["Y"].astype(float)
        train_all_dt["Y_pred"] = train_all_dt["Y_pred"].astype(float)
    train_all_dt["Y"][~dt_idx_tr] = None
    train_all_dt["Y_pred"][~dt_idx_tr] = None
    test_all_dt = copy.deepcopy(test_all)
    if not is_regr:
        test_all_dt["Y"] = test_all_dt["Y"].astype(float)
        test_all_dt["Y_pred"] = test_all_dt["Y_pred"].astype(float)
    test_all_dt["Y"][~dt_idx_te] = None
    test_all_dt["Y_pred"][~dt_idx_te] = None
    log("--------------------------------------------------------------", logger)
    log("(Daytime only) For all training data with method " + str(method) +  ":", logger)
    metric_dt["train"] = computeMetric(train_all_dt["Y"], train_all_dt["Y_pred"],
            is_regr, aggr_axis=True, event_thr=event_thr)
    for m in metric_dt["train"]:
        log("Metric: " + m, logger)
        log(metric_dt["train"][m], logger)
    log("--------------------------------------------------------------", logger)
    log("(Daytime only) For all testing data with method " + str(method) + ":", logger)
    metric_dt["test"] = computeMetric(test_all_dt["Y"], test_all_dt["Y_pred"],
            is_regr, aggr_axis=True, event_thr=event_thr)
    for m in metric_dt["test"]:
        log("Metric: " + m, logger)
        log(metric_dt["test"][m], logger)

    # Save plot
    if out_p_root is not None:
        Y_true = test_all["Y"]
        Y_pred = test_all["Y_pred"]
        if is_regr:
            r2 = metric["test"]["r2"]
            mse = metric["test"]["mse"]
            r2_dt = metric_dt["test"]["r2"]
            mse_dt = metric_dt["test"]["mse"]
            log("Print prediction plots...", logger)
            predictionPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx_te, r2_dt, mse_dt)
            log("Print residual plots...", logger)
            residualPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx_te, r2_dt, mse_dt)
        else:
            Y_score = test_all["Y_score"]
            log("Print prediction recall plots...", logger)
            prPlot(method, Y_true, Y_score, out_p)
            log("Print roc curve plots...", logger)
            try:
                rocPlot(method, Y_true, Y_score, out_p)
            except Exception as e:
                log(str(type(e).__name__) + ": " + str(e), logger)
        #log("Print time series plots...", logger)
        #timeSeriesPlot(method, Y_true, Y_pred, out_p, dt_idx_te, w=40) # NOTE: this takes very long time

    # Release memory
    del df_X
    del df_Y
    del X
    del Y
    del X_train
    del X_test
    del train_all_dt
    del train_all
    del test_all_dt
    del test_all
    gc.collect()
    log("Done", logger)
    return True


def rocPlot(method, Y_true, Y_score, out_p):
    roc = round(roc_auc_score(Y_true, Y_score[:, -1]), 4)
    # Precision vs recall
    fig = plt.figure(figsize=(8, 8), dpi=150)
    fpr, tpr, threshold = roc_curve(Y_true, Y_score[:, -1])
    plt.step(fpr, tpr, "o", alpha=0.2, markersize=0, color=(0,0,1), where="post")
    plt.fill_between(fpr, tpr, alpha=0.2, color=(0,0,1), step="post")
    plt.plot([0,1], [0,1], "--", alpha=0.8, markersize=0, color=(0,0,1), lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Method=" + method + ", roc_auc=" + str(roc), fontsize=18)
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


def timeSeriesPlot(method, Y_true, Y_pred, out_p, dt_idx, fold="all", show_y_tick=False, w=18):
    if len(Y_true.shape) > 1: Y_true = np.sum(Y_true, axis=1)
    if len(Y_pred.shape) > 1: Y_pred = np.sum(Y_pred, axis=1)
    fig = plt.figure(figsize=(w, 4), dpi=150)

    # Time vs Ground Truth
    plt.subplot(2, 1, 1)
    plt.bar(range(0, len(Y_true)), Y_true, 1, alpha=0.8, color=(0.2, 0.53, 0.74), align="edge")
    plt.title("Crowdsourced smell events", fontsize=18)
    for i in range(0, len(dt_idx)):
        if dt_idx[i] == False: plt.axvspan(i, i+1, facecolor="0.2", alpha=0.5)
    if not show_y_tick: plt.yticks([], [])
    plt.xlim(0, len(Y_true))
    plt.ylim(np.amin(Y_true), np.amax(Y_true))
    plt.grid(False)

    # Time vs Prediction
    plt.subplot(2, 1, 2)
    plt.bar(range(0, len(Y_pred)), Y_pred, 1, alpha=0.8, color=(0.84, 0.24, 0.31), align="edge")
    plt.title("Predicted smell events", fontsize=18)
    for i in range(0, len(dt_idx)):
        if dt_idx[i] == False: plt.axvspan(i, i+1, facecolor="0.2", alpha=0.5)
    if not show_y_tick: plt.yticks([], [])
    plt.xlim(0, len(Y_pred))
    plt.ylim(np.amin(Y_pred), np.amax(Y_pred))
    plt.grid(False)

    # Save plot
    plt.tight_layout()
    fig.savefig(out_p + method + "_fold_" + str(fold) + "_regr_time.png")
    fig.clf()
    plt.close()


def residualPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx, r2_dt, mse_dt):
    if len(Y_true.shape) > 1: Y_true = np.sum(Y_true, axis=1)
    if len(Y_pred.shape) > 1: Y_pred = np.sum(Y_pred, axis=1)
    res = Y_true - Y_pred
    # Histogram of Residual
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.hist(res, bins=100)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    plt.xlim(np.amin(res)-1, np.amax(res)+1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_hist.png")
    # Histogram of Residual (daytime)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    res_dt = res[dt_idx]
    plt.hist(res_dt, bins=100)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("[Daytime] Method=" + method + ", r2=" + str(r2_dt) + ", mse=" + str(mse_dt), fontsize=18)
    plt.xlim(np.amin(res_dt)-1, np.amax(res_dt)+1)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_hist_dt.png")
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
    # True vs Residual (daytime)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    Y_true_dt = Y_true[dt_idx]
    plt.plot(Y_true_dt, res_dt, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("True smell value")
    plt.ylabel("Residual")
    plt.title("[Daytime] Method=" + method + ", r2=" + str(r2_dt) + ", mse=" + str(mse_dt), fontsize=18)
    plt.xlim(np.amin(Y_true_dt)-0.5, np.amax(Y_true_dt)+0.5)
    plt.ylim(np.amin(res_dt)-0.5, np.amax(res_dt)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_true_dt.png")
    # Pred vs Residual
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.plot(Y_pred, res, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("Predicted smell value")
    plt.ylabel("Residual")
    plt.title("Method=" + method + ", r2=" + str(r2) + ", mse=" + str(mse), fontsize=18)
    plt.xlim(np.amin(Y_pred)-0.5, np.amax(Y_pred)+0.5)
    plt.ylim(np.amin(res)-0.5, np.amax(res)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_pred.png")
    # Pred vs Residual (daytime)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    Y_pred_dt = Y_pred[dt_idx]
    plt.plot(Y_pred_dt, res_dt, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("Predicted smell value")
    plt.ylabel("Residual")
    plt.title("[Daytime] Method=" + method + ", r2=" + str(r2_dt) + ", mse=" + str(mse_dt), fontsize=18)
    plt.xlim(np.amin(Y_pred_dt)-0.5, np.amax(Y_pred_dt)+0.5)
    plt.ylim(np.amin(res_dt)-0.5, np.amax(res_dt)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_res_pred_dt.png")


def predictionPlot(method, r2, mse, Y_true, Y_pred, out_p, dt_idx, r2_dt, mse_dt):
    if len(Y_true.shape) > 1: Y_true = np.sum(Y_true, axis=1)
    if len(Y_pred.shape) > 1: Y_pred = np.sum(Y_pred, axis=1)
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
    # True vs Prediction (daytime)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    Y_true_dt = Y_true[dt_idx]
    Y_pred_dt = Y_pred[dt_idx]
    plt.plot(Y_true_dt, Y_pred_dt, "o", alpha=0.8, markersize=5, color=(0,0,1))
    plt.xlabel("True smell value")
    plt.ylabel("Predicted smell value")
    plt.title("[Daytime] Method=" + method + ", r2=" + str(r2_dt) + ", mse=" + str(mse_dt), fontsize=18)
    plt.xlim(np.amin(Y_true_dt)-0.5, np.amax(Y_true_dt)+0.5)
    plt.ylim(np.amin(Y_pred_dt)-0.5, np.amax(Y_pred_dt)+0.5)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(out_p + method + "_regr_r2_dt.png")
