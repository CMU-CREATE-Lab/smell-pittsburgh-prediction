import sys
from util import checkAndCreateDir, generateLogger
from getData import getData
from preprocessData import preprocessData
from analyzeData import analyzeData
from computeFeatures import computeFeatures
from crossValidation import crossValidation
from datetime import datetime
import pytz


def main(argv):
    p = "data_main/"
    mode = None
    if len(argv) >= 2:
        mode = argv[1]

    # Parameters
    # NOTE: if is_regr is changed, you need to run the computeFeatures function again to generate new features
    is_regr = False # False for classification, True for regression
    smell_thr = 40 # threshold to define a smell event

    # The starting and ending date for the data used in the Smell PGH paper
    #start_dt = datetime(2016, 10, 31, 0, tzinfo=pytz.timezone("US/Eastern"))
    #end_dt = datetime(2018, 9, 30, 0, tzinfo=pytz.timezone("US/Eastern"))
    #region_setting = 0

    # The starting and ending data for the later second release of the dataset
    # (this version contains more zipcodes than the previous version)
    start_dt = datetime(2016, 10, 31, 0, tzinfo=pytz.timezone("US/Eastern"))
    end_dt = datetime(2022, 12, 11, 0, tzinfo=pytz.timezone("US/Eastern"))
    region_setting = 1

    # Set mode
    get_data, preprocess_data, analyze_data, compute_features, cross_validation = False, False, False, False, False
    if mode == "pipeline":
        get_data = True
        preprocess_data = True
        compute_features = True
        cross_validation = True
    elif mode == "data":
        get_data = True
    elif mode == "preprocess":
        preprocess_data = True
    elif mode == "feature":
        compute_features = True
    elif mode == "validation":
        cross_validation = True
    elif mode == "analyze":
        analyze_data = True
    else:
        get_data = True
        preprocess_data = True
        compute_features = True
        cross_validation = True

    # Get data
    # OUTPUT: raw esdr and raw smell data
    if get_data:
        getData(out_p=[p+"esdr_raw/",p+"smell_raw.csv"], start_dt=start_dt, end_dt=end_dt, region_setting=region_setting)

    # Preprocess data
    # INPUT: raw esdr and raw smell data
    # OUTPUT: preprocessed esdr and smell data
    if preprocess_data:
        preprocessData(in_p=[p+"esdr_raw/",p+"smell_raw.csv"], out_p=[p+"esdr.csv",p+"smell.csv"])

    # Analyze data
    if analyze_data:
        analyzeData(in_p=[p+"esdr.csv",p+"smell.csv"], out_p_root=p, start_dt=start_dt, end_dt=end_dt)

    # Compute features
    # INPUT: preprocessed esdr and smell data
    # OUTPUT: features and labels
    if compute_features:
        computeFeatures(in_p=[p+"esdr.csv",p+"smell.csv"], out_p=[p+"X.csv",p+"Y.csv",p+"C.csv"],
            is_regr=is_regr, f_hr=8, b_hr=3, thr=smell_thr, add_inter=False, add_roll=False, add_diff=False)

    # Cross validation
    # INPUT: features
    # OUTPUT: plots or metrics
    if cross_validation:
        #methods = ["ET", "RF", "SVM", "RLR", "LR", "LA", "EN", "MLP", "KN", "DMLP"] # regression
        #methods = ["ET", "RF", "SVM", "LG", "MLP", "KN", "DMLP", "HCR", "CR", "DT"] # classification
        methods = ["ET"] # default for extra trees
        #methods = genModelSet(is_regr)
        p_log = p + "log/"
        if is_regr: p_log += "regression/"
        else: p_log += "classification/"
        checkAndCreateDir(p_log)
        num_folds = int((end_dt - start_dt).days / 7) # one fold represents a week
        for m in methods:
            start_time_str = datetime.now().strftime("%Y-%d-%m-%H%M%S")
            lg = generateLogger(p_log + m + "-" + start_time_str + ".log", format=None)
            crossValidation(in_p=[p+"X.csv",p+"Y.csv",p+"C.csv"], out_p_root=p, event_thr=smell_thr,
                method=m, is_regr=is_regr, logger=lg, num_folds=num_folds, skip_folds=48, train_size=8000)


def genModelSet(is_regr):
    m_all = []
    methods = ["ET", "RF"]
    if is_regr:
        n_estimators = [200]
    else:
        n_estimators = [1000]
    max_features = [30,60,90,120,150,180]
    min_samples_split = [2,4,8,16,32,64,128]
    for n in n_estimators:
        for mf in max_features:
            for mss in min_samples_split:
                for m in methods:
                    m_all.append(m + "-" + str(n) + "-" + str(mf) + "-" + str(mss))
    return m_all


if __name__ == "__main__":
    main(sys.argv)
