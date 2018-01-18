import sys
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
    is_regr = True # is regression or classification
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
        getData(out_p=[p+"esdr.csv",p+"smell.csv"], start_dt=datetime(2016, 10, 6, 0), end_dt=datetime(2018, 1, 11, 0))

    # Compute features
    # INPUT: raw esdr and smell data
    # OUTPUT: features
    if compute_features:
        #computeFeatures(in_p=[p+"esdr.csv",p+"smell.csv"], out_p=[p+"X.csv",p+"Y.csv"],
        #    is_regr=is_regr, f_hr=8, b_hr=0, thr=40, add_inter=False, add_roll=False, add_diff=False)
        computeFeatures(in_p=[p+"esdr.csv", p+"smell.csv"], out_p=[p+"X.csv", p+"Y.csv"],
            is_regr=is_regr, f_hr=8, b_hr=2, thr=40, add_inter=True, add_roll=False, add_diff=False)

    # Plot features
    if plot_features:
        plotFeatures([p+"X.csv",p+"Y.csv"], p, is_regr=is_regr)

    # Cross validation
    # INPUT: features
    # OUTPUT: plots or metrics
    if cross_validation:
        #methods = ["ET", "RF", "SVM", "RLR", "LR", "LA", "EN", "MLP", "KN", "GP", "ANCNN", "DMLP"] # regression
        #methods = ["ET", "RF", "SVM", "LG", "MLP", "KN", "GP", "ANCNN", "DMLP"] # classification
        methods = ["DMLP"]
        #methods = ["ANCNN"]
        #methods = ["SVM"]
        p_log = p + "log/"
        checkAndCreateDir(p_log)
        for m in methods:
            start_time_str = datetime.now().strftime("%Y-%d-%m-%H%M%S")
            logger = generateLogger(p_log + m + "-" + start_time_str + ".log", format=None)
            crossValidation(in_p=[p+"X.csv",p+"Y.csv"], out_p_root=p, method=m, is_regr=is_regr, logger=logger)

if __name__ == "__main__":
    main(sys.argv)
