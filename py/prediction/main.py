import sys
import torch # need to import torch early to avoid an ImportError related to static TLS
from util import *
from getData import *
from analyzeData import *
from computeFeatures import *
from crossValidation import *

def main(argv):
    p = "data_main/"
    mode = None
    if len(argv) >= 2:
        mode = argv[1]

    # Parameters
    is_regr = False # is classification
    #is_regr = True # is regression
    start_dt = datetime(2016, 10, 9, 0, tzinfo=pytz.timezone("US/Eastern"))
    end_dt = datetime(2018, 4, 15, 0, tzinfo=pytz.timezone("US/Eastern"))
    get_data, analyze_data, compute_features, cross_validation = False, False, False, False
    if mode == "run_all":
        get_data = True
        compute_features = True
        cross_validation = True
    elif mode == "test":
        compute_features = True
        cross_validation = True
    elif mode == "analyze":
        analyze_data = True
    else:
        #get_data = True
        analyze_data = True
        #compute_features = True
        #cross_validation = True

    # Get data
    # OUTPUT: raw esdr and smell data
    if get_data:
        getData(out_p=[p+"esdr.csv",p+"smell.csv"], start_dt=start_dt, end_dt=end_dt)

    # Analyze data
    if analyze_data:
        analyzeData(in_p=[p+"esdr.csv",p+"smell.csv"], out_p_root=p)

    # Compute features
    # INPUT: raw esdr and smell data
    # OUTPUT: features and label
    if compute_features:
        computeFeatures(in_p=[p+"esdr.csv",p+"smell.csv"], out_p=[p+"X.csv",p+"Y.csv",p+"C.csv"],
            is_regr=is_regr, f_hr=8, b_hr=3, thr=40, add_inter=False, add_roll=False, add_diff=False)

    # Cross validation
    # INPUT: features
    # OUTPUT: plots or metrics
    if cross_validation:
        #methods = ["ANCNN"]
        #methods = ["ET", "RF", "SVM", "RLR", "LR", "LA", "EN", "MLP", "KN", "DMLP"] # regression
        #methods = ["ET", "RF", "SVM", "LG", "MLP", "KN", "DMLP", "HCR", "CR", "DT"] # classification
        methods = ["RF", "ET"] * 30
        #methods = genMethodSet()
        p_log = p + "log/"
        if is_regr: p_log += "regression/"
        else: p_log += "classification/"
        checkAndCreateDir(p_log)
        num_folds = (end_dt - start_dt).days / 7 # one fold represents a week
        for m in methods:
            start_time_str = datetime.now().strftime("%Y-%d-%m-%H%M%S")
            lg = generateLogger(p_log + m + "-" + start_time_str + ".log", format=None)
            crossValidation(in_p=[p+"X.csv",p+"Y.csv",p+"C.csv"], out_p_root=p,
                method=m, is_regr=is_regr, logger=lg, num_folds=num_folds, skip_folds=48)

def genMethodSet():
    m_all = []
    methods = ["ET", "RF"]
    n_estimators = [1000]
    #max_features = range(5,200,5) + [None]
    max_features = [30,60,90,120]
    min_samples_split = [2,4,8,16,32]
    for n in n_estimators:
        for mf in max_features:
            for mss in min_samples_split:
                for m in methods:
                    m_all.append(m + "-" + str(n) + "-" + str(mf) + "-" + str(mss))
    return m_all

if __name__ == "__main__":
    main(sys.argv)
