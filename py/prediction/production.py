import sys
from util import *
from getData import *
from computeFeatures import *
from selectFeatures import *
from trainModel import *
import joblib
from datetime import timedelta
import os
import subprocess

def main(argv):
    mode = None
    if len(argv) >= 2:
        mode = argv[1]

    if mode == "train":
        train()
    elif mode == "predict":
        predict()
    else:
        print "Use 'python main.py [mode]'; mode can be 'train' or 'predict'"

def train():
    p = "data_production/"

    # Set logger
    logger = generateLogger(p+"log.log")
    log("--------------------------------------------------------------------------", logger)
    log("---------------------------------  Train  --------------------------------", logger)

    # Get data
    end_dt = datetime.now() - timedelta(hours=24)
    start_dt = end_dt - timedelta(hours=6000)
    log("Get data from " + str(start_dt) + " to " + str(end_dt), logger)
    df_esdr, df_smell = getData(start_dt=start_dt, end_dt=end_dt, logger=logger)

    # Compute features
    df_X, df_Y = computeFeatures(df_esdr=df_esdr, df_smell=df_smell, f_hr=8, b_hr=12, thr=80,
        add_inter=True, add_roll=True, add_diff=True, logger=logger, out_p_mean=p+"mean.csv", out_p_std=p+"std.csv")

    # Select features
    df_X, df_Y = selectFeatures(df_X, df_Y, logger=logger, out_p=p+"feat_selected.csv")

    # Train, save, and evaluate model
    model = trainModel({"X": df_X, "Y": df_Y}, method="SVM", out_p=p+"SVM_model.pkl", logger=logger)
    metric = computeMetric(df_Y, model.predict(df_X), False)
    for m in metric:
        log(metric[m], logger)

def predict():
    p = "data_production/"

    # Set logger
    logger = generateLogger(p+"log.log")
    log("--------------------------------------------------------------------------", logger)
    log("--------------------------------  Predict  -------------------------------", logger)

    # Get data for previous b_hr hours
    b_hr = 12
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=b_hr+1)
    log("Get data from " + str(start_dt) + " to " + str(end_dt), logger)
    df_esdr, _ = getData(start_dt=start_dt, end_dt=end_dt, logger=logger)
    if len(df_esdr) < b_hr+1:
        log("ERROR: Length of esdr is less than " + str(b_hr+1) + " hours", logger)
        log("Length of esdr = " + str(len(df_esdr)), logger)
        return
    
    # Compute features
    df_X, _ = computeFeatures(df_esdr=df_esdr, f_hr=8, b_hr=12, thr=80,
        add_inter=True, add_roll=True, add_diff=True, logger=logger, in_p_mean=p+"mean.csv", in_p_std=p+"std.csv")
    if len(df_X) != 1:
        log("ERROR: Length of X is not 1", logger)
        log("Length of X = " + str(len(df_X)), logger)
        return

    # Select features
    df_feat_selected = pd.read_csv(p+"feat_selected.csv")
    df_X = df_X[df_feat_selected.columns]

    # Load model
    log("Load model...", logger)
    model = joblib.load(p+"SVM_model.pkl")

    # Predict result
    y_pred = model.predict(df_X)[0]
    log("Prediction for " + str(end_dt) + " is " + str(y_pred), logger)
    nst_p = p + "notification_sent_times.csv"
    if y_pred == 1:
        # Read the push notification file
        if isFileHere(nst_p):
            df_nst = pd.read_csv(nst_p, parse_dates=["DateTime"])
            last_date = df_nst["DateTime"].dt.date.iloc[-1]
            current_date = end_dt.date()
            if current_date == last_date:
                # We already sent push notifications to users today, do not send it again until next day
                log("Ignore this prediction because we already sent a push notification today", logger)
                return
        else:
            df_nst = pd.DataFrame(data=[], columns=["DateTime"])
        # Send push notification to users
        #os.system('cd /var/www/rails-apps/smellpgh/staging/current/ ; bundle exec rake firebase_push_notification:send["/topics/SmellReports","Smell Prediction (beta testing)","There may be a smell event in Pittsburgh today - be sure to submit a smell report if you notice it. Tap to learn more!","smell_prediction"] RAILS_ENV=staging >> /home/yenchiah/smell-pittsburgh-prediction-production/py/prediction/data_production/push.log 2>&1')
        log("A push notification was sent to users", logger)
        df_nst = df_nst.append({"DateTime": end_dt}, ignore_index=True)
        df_nst.to_csv(nst_p, index=False)

if __name__ == "__main__":
    main(sys.argv)
