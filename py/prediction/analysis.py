import sys
import torch # need to import torch early to avoid an ImportError related to static TLS
from util import *
from getData import *
from computeFeatures import *
from plotFeatures import *
from crossValidation import *

def main(argv):
    p = "data_analysis/"
    mode = None
    if len(argv) >= 2:
        mode = argv[1]

    # Parameters
    is_regr = False # is regression or classification
    if mode == "run_all":
        get_data = True
        compute_features = True
        plot_features = False
        cross_validation = True
    elif mode == "test":
        get_data = False
        compute_features = True
        plot_features = False
        cross_validation = True
    else:
        get_data = False
        compute_features = False
        plot_features = False
        cross_validation = True

    # Get data
    # OUTPUT: raw esdr and smell data
    if get_data:
        getData(out_p=[p+"esdr.csv",p+"smell.csv"], start_dt=datetime(2016, 10, 6, 0), end_dt=datetime(2018, 2, 8, 0))

    # Compute features
    # INPUT: raw esdr and smell data
    # OUTPUT: features
    if compute_features:
        computeFeatures(in_p=[p+"esdr.csv",p+"smell.csv"], out_p=[p+"X.csv",p+"Y.csv"],
            is_regr=is_regr, f_hr=8, b_hr=3, thr=40, add_inter=False, add_roll=False, add_diff=False)

    # Plot features
    if plot_features:
        plotFeatures([p+"X.csv",p+"Y.csv"], p, is_regr=is_regr)

    # Cross validation
    # INPUT: features
    # OUTPUT: plots or metrics
    if cross_validation:
        #methods = ["ANCNN"]
        #methods = ["ET", "RF", "SVM", "RLR", "LR", "LA", "EN", "MLP", "KN", "DMLP"] # regression
        #methods = ["SVM", "RLR", "LR", "LA", "EN", "MLP", "KN", "DMLP"] # regression
        #methods = ["ET", "RF", "SVM", "LG", "MLP", "KN", "DMLP"] # classification
        #methods = ["ET"]
        methods = genMethodSetET()
        p_log = p + "log/"
        if is_regr: p_log += "regression/"
        else: p_log += "classification/"
        checkAndCreateDir(p_log)
        for m in methods:
            start_time_str = datetime.now().strftime("%Y-%d-%m-%H%M%S")
            logger = generateLogger(p_log + m + "-" + start_time_str + ".log", format=None)
            crossValidation(in_p=[p+"X.csv",p+"Y.csv"], out_p_root=p, method=m, is_regr=is_regr, logger=logger)

def genMethodSetET():
    m_all = []
    m = "ET"
    n_estimators = [100,200,400,800]
    max_features = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,None]
    min_samples_split = [2]
    for n in n_estimators:
        for mf in max_features:
            for mss in min_samples_split:
                m_all.append(m + "-" + str(n) + "-" + str(mf) + "-" + str(mss))
    return m_all

if __name__ == "__main__":
    main(sys.argv)
