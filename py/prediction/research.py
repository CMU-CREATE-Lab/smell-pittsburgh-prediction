import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from util import binary2Interval
from util import mergeInterval
from util import evalEventDetection


def scorer(model, x, y):
    """
    A customized scoring function to evaluate a classifier.

    Parameters
    ----------
    model : a sklearn model object
        The classifier model.
    x : pandas.DataFrame
        The feature matrix.
    y : pandas.Series
        The label vector.

    Returns
    -------
    dict of int or float
        A dictionary of evaluation metrics.
    """
    y_pred = model.predict(x)
    c = confusion_matrix(y, y_pred, labels=[0,1])
    p = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    a = accuracy_score(y, y_pred)
    s = evalEventDetection(y, y_pred, thr=1, h=1, round_to_decimal=2)
    return {"tn": c[0,0], "fp": c[0,1], "fn": c[1,0], "tp": c[1,1],
            "precision": p[0], "recall": p[1], "f1": p[2], "accuracy": a,
            "tp_e": s["TP"], "fp_e": s["FP"], "fn_e": s["FN"],
            "precision_e": s["precision"], "recall_e": s["recall"], "f1_e": s["f_score"]}


def train_and_evaluate(model, df_x, df_y, train_size=336, test_size=168):
    """
    Train and evaluate a model.

    Parameters
    ----------
    model : a sklearn model object
        The classifier model.
    df_x : pandas.DataFrame
        The dataframe with features.
    df_y : pandas.DataFrame
        The dataframe with labels.
    train_size : int
        Number of samples for training.
    test_size : int
        Number of samples for testing.
    """
    print("Use model", model)
    print("Perform cross-validation, please wait...")

    # Create time series splits for cross-validation.
    splits = []
    dataset_size = df_x.shape[0]
    for i in range(train_size, dataset_size, test_size):
        start = i - train_size
        end = i + test_size
        if (end >= dataset_size): break
        train_index = range(start, i)
        test_index = range(i, end)
        splits.append((list(train_index), list(test_index)))

    # Perform cross-validation.
    cv_res = cross_validate(model, df_x, df_y.squeeze(), cv=splits, scoring=scorer, verbose=2)

    # Print evaluation metrics.
    tp = np.sum(cv_res["test_tp"])
    fp = np.sum(cv_res["test_fp"])
    tn = np.sum(cv_res["test_tn"])
    fn = np.sum(cv_res["test_fn"])
    tp_e = np.sum(cv_res["test_tp_e"])
    fp_e = np.sum(cv_res["test_fp_e"])
    fn_e = np.sum(cv_res["test_fn_e"])
    print("="*40)
    print("For all the data points:")
    print("average precision:", round(np.mean(cv_res["test_precision"]), 2))
    print("average recall:", round(np.mean(cv_res["test_recall"]), 2))
    print("average f1-score:", round(np.mean(cv_res["test_f1"]), 2))
    print("average accuracy:", round(np.mean(cv_res["test_accuracy"]), 2))
    print("number of true positives:", np.sum(cv_res["test_tp"]))
    print("number of false positives:", np.sum(cv_res["test_fp"]))
    print("number of true negatives:", np.sum(cv_res["test_tn"]))
    print("number of false negatives:", np.sum(cv_res["test_fn"]))
    print("total precision:", round(tp/(tp+fp), 2))
    print("total recall:", round(tp/(tp+fn), 2))
    print("total f1-score:", round(2*tp/(2*tp+fp+fn), 2))
    print("total accuracy:", round((tp+tn)/(tp+tn+fp+fn), 2))
    print("-"*40)
    print("For only the events:")
    print("average precision:", round(np.mean(cv_res["test_precision_e"]), 2))
    print("average recall:", round(np.mean(cv_res["test_recall_e"]), 2))
    print("average f1-score:", round(np.mean(cv_res["test_f1_e"]), 2))
    print("number of true positives:", np.sum(cv_res["test_tp_e"]))
    print("number of false positives:", np.sum(cv_res["test_fp_e"]))
    print("number of false negatives:", np.sum(cv_res["test_fn_e"]))
    print("total precision:", round(tp_e/(tp_e+fp_e), 2))
    print("total recall:", round(tp_e/(tp_e+fn_e), 2))
    print("total f1-score:", round(2*tp_e/(2*tp_e+fp_e+fn_e), 2))
    print("="*40)


def compute_reward(df_y, discount=0.8, base=0.5):
    """
    Compute rewards using human preference.

    We prefer an event to be detected early.
    So if the model detects an event early, it should get a high reward.

    Question: what should be the design of a good reward function
    ...so that we can reach the goal of early event detection?

    Parameters
    ----------
    df_y : pandas.DataFrame
        The dataframe with labels.
    base : float
        The base value that will be used in computing the reward,
        ...which is the variable "b" in the following description of the discount parameter.
    discount : float
        The discount factor of the reward (should be between 0 and 1)
        For example, if the label array Y is [0, 1, 1, 1, 0] and we use discount=0.8,
        ...the reward array will be [-b, 1, 0.8, 0.64, -b] for the action that predicts Y=1
        ...and the reward will be [b, -1, -0.8, -0.64, b] for the action that predicts Y=0

    Returns
    -------
    pandas.DataFrame
        A dataframe with both the action and the rewards.
    """
    # First we need to compute the interval
    iv_list = mergeInterval(binary2Interval(df_y>=1), h=1)
    print("Total number of events:", len(iv_list))

    # Compute rewards
    r_y1 = np.ones(len(df_y))*-1*base
    r_y0 = np.ones(len(df_y))*base
    for iv in iv_list:
        running_r_y1 = 1
        running_r_y0 = -1
        for i in range(iv[0], iv[1]+1):
            # Compute the rewards when Y=1
            r_y1[i] = round(running_r_y1, 3)
            running_r_y1 *= discount
            # Compute the rewards when Y=0
            r_y0[i] = round(running_r_y0, 3)
            running_r_y0 *= discount

    # Sanity check
    df_y["r_y1"] = r_y1
    df_y["r_y0"] = r_y0
    print(df_y[10:50])

    return df_y


def main(argv):
    p = "data_main/"
    df_x = pd.read_csv(p + "X.csv")
    df_y = pd.read_csv(p + "Y.csv")
    df_y = compute_reward(df_y)
    model = ExtraTreesClassifier(n_estimators=200, max_features=60, min_samples_split=32, n_jobs=-2)
    train_and_evaluate(model, df_x, df_y["smell"], train_size=300, test_size=168)

if __name__ == "__main__":
    main(sys.argv)
