"""
The production code for predicting smell events and sending push notifications
(sending push notifications requires the rake script in the smell-pittsburgh-rails repository)
"""


import sys
from util import log, generateLogger, computeMetric, isFileHere
import pandas as pd
from getData import getData
from preprocessData import preprocessData
from computeFeatures import computeFeatures
#from selectFeatures import selectFeatures
from trainModel import trainModel
import joblib
from datetime import timedelta, datetime
import os
import subprocess

# The flag to determine the server type
SERVER = "staging"
#SERVER = "production"

# The flag for enabling the rake call to send push notifications
#ENABLE_RAKE_CALL = False
ENABLE_RAKE_CALL = True

# The path for storing push notification data
DATA_PATH = "data_production/"


def main(argv):
    mode = None
    if len(argv) >= 2:
        mode = argv[1]

    if mode == "train":
        train()
    elif mode == "predict":
        predict()
    else:
        print("Use 'python main.py [mode]'; mode can be 'train' or 'predict'")


def train(f_hr=8, b_hr=3, thr=40, method="HCR"):
    """
    Train the machine learning model for predicting smell events
    
    Input:
        f_hr: the number of hours to look further and compute responses (Y),
            ...which is the sum of smell ratings (that are larger than 3) over the future f_hr hours
        b_hr: the number of hours to look back and compute features (X),
            ...which are the sensor readings (on ESDR) over the past b_hr hours
        thr: the threshold for binning the smell value into two classes (for classification)
        method: the method used in the "trainModel.py" file
    """
    p = DATA_PATH

    # Set logger
    logger = generateLogger(p+"log.log")
    log("--------------------------------------------------------------------------", logger)
    log("---------------------------------  Train  --------------------------------", logger)

    # Get and preprocess data
    end_dt = datetime.now() - timedelta(hours=24)
    start_dt = end_dt - timedelta(hours=8000)
    log("Get data from " + str(start_dt) + " to " + str(end_dt), logger)
    df_esdr_array_raw, df_smell_raw = getData(start_dt=start_dt, end_dt=end_dt, logger=logger)
    df_esdr, df_smell = preprocessData(df_esdr_array_raw=df_esdr_array_raw, df_smell_raw=df_smell_raw, logger=logger)

    # Compute features
    df_X, df_Y, df_C = computeFeatures(df_esdr=df_esdr, df_smell=df_smell, f_hr=f_hr, b_hr=b_hr, thr=thr, is_regr=False,
        add_inter=False, add_roll=False, add_diff=False, logger=logger, out_p_mean=p+"mean.csv", out_p_std=p+"std.csv")

    # Select features
    # NOTE: currently, the best model uses all the features
    #df_X, df_Y = selectFeatures(df_X, df_Y, logger=logger, out_p=p+"feat_selected.csv")

    # Train, save, and evaluate model
    # NOTE: to know more about the model, see the "HybridCrowdClassifier.py" file
    model = trainModel({"X": df_X, "Y": df_Y, "C": df_C}, method=method, out_p=p+"model.pkl", logger=logger)
    metric = computeMetric(df_Y, model.predict(df_X, df_C), False)
    for m in metric:
        log(metric[m], logger)


def predict(f_hr=8, b_hr=3, thr=40):
    """
    Predict smell events using the trained machine learning model
    
    For the description of the input arguments, see the docstring in the train() function
    """
    p = DATA_PATH

    # Set logger
    logger = generateLogger(p+"log.log")
    log("--------------------------------------------------------------------------", logger)
    log("--------------------------------  Predict  -------------------------------", logger)

    # Get data for previous b_hr hours
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=b_hr+1)
    log("Get data from " + str(start_dt) + " to " + str(end_dt), logger)
    df_esdr_array_raw, df_smell_raw = getData(start_dt=start_dt, end_dt=end_dt, logger=logger)
    df_esdr, df_smell = preprocessData(df_esdr_array_raw=df_esdr_array_raw, df_smell_raw=df_smell_raw, logger=logger)
    if len(df_esdr) < b_hr+1:
        log("ERROR: Length of esdr is less than " + str(b_hr+1) + " hours", logger)
        log("Length of esdr = " + str(len(df_esdr)), logger)
        return

    # Compute features
    df_X, _, df_C = computeFeatures(df_esdr=df_esdr, df_smell=df_smell, f_hr=f_hr, b_hr=b_hr, thr=thr, is_regr=False,
        add_inter=False, add_roll=False, add_diff=False, logger=logger, in_p_mean=p+"mean.csv", in_p_std=p+"std.csv")
    if len(df_X) != 1:
        log("ERROR: Length of X is not 1", logger)
        log("Length of X = " + str(len(df_X)), logger)
        return

    # Select features
    # NOTE: currently, the best model uses all the features
    #df_feat_selected = pd.read_csv(p+"feat_selected.csv")
    #df_X = df_X[df_feat_selected.columns]

    # Load model
    log("Load model...", logger)
    model = joblib.load(p+"model.pkl")

    # Predict result
    # For the hybrid crowd classifier
    # if pred==0, no event
    # if pred==1, event predicted by the base estimator
    # if pred==2, event detected by the crowd
    # if pred==3, event both predicted by the base estimator and detected by the crowd
    y_pred = model.predict(df_X, df_C)[0]
    log("Prediction for " + str(end_dt) + " is " + str(y_pred), logger)

    # Send the predictive push notification
    if y_pred in (1, 3): pushType1(end_dt, logger)

    # Send the crowd-based notification
    # NOTE: comment out the next line when migrating its function to the rails server
    if y_pred in (2, 3): pushType2(end_dt, logger)


def pushType1(end_dt, logger):
    """
    Send type 1 push notification (predicted by the classifier)

    Input:
        end_dt: the ending time for getting the ESDR data that is used for prediction (which is the current time)
        logger: the python logger created by the generateLogger() function
    """
    p = DATA_PATH

    # Read the push notification file
    nst_p = p + "notification_sent_times.csv"
    if isFileHere(nst_p):
        # If the file that stores the notification sending time exist,
        # ...load the times and check if we already sent the notification at the same date before
        df_nst = pd.read_csv(nst_p, parse_dates=["DateTime"])
        last_date = df_nst["DateTime"].dt.date.iloc[-1] # the last date that we send the notification
        current_date = end_dt.date()
        if current_date == last_date: # check if the dates (only year, month, and day) match
            # We already sent push notifications to users today, do not send it again until next day
            log("Ignore this prediction because we already sent a push notification today", logger)
            return
    else:
        # If the file did not exist, create a new pandas Dataframe to store the time when we send notifications
        df_nst = pd.DataFrame(data=[], columns=["DateTime"])

    # Send push notification to users
    if ENABLE_RAKE_CALL:
        # NOTE: Python by default uses the sh terminal but we want it to use bash,
        # ...because "source" and "bundle" only works for bash on the Hal11 machine
        # ...(on the sh terminal we will want to use "." instead of "source")
        cmd = 'source /etc/profile ; cd /var/www/rails-apps/smellpgh/' + SERVER + '/current/ ; bundle exec rake firebase_push_notification:send_prediction["/topics/SmellReports"] RAILS_ENV=' + SERVER + ' >> /var/www/smell-pittsburgh-prediction/py/prediction/data_production/push.log 2>&1'
        subprocess.call(["bash", "-c", cmd])

    # Create a CSV file that writes the time when the system send the push notification
    log("A prediction push notification was sent to users", logger)
    df_nst = df_nst.append({"DateTime": end_dt}, ignore_index=True) # append a row to the pandas Dataframe
    df_nst.to_csv(nst_p, index=False) # re-write the Dataframe to a CSV file


def pushType2(end_dt, logger):
    """
    Send type 2 push notification (verified by the crowd)
    
    Input:
        end_dt: the ending time for getting the ESDR data that is used for prediction (which is the current time) 
        logger: the python logger created by the generateLogger() function
    """
    p = DATA_PATH

    # Read the crowd push notification file
    cvnst_p = p + "crow_verified_notification_sent_times.csv"
    if isFileHere(cvnst_p):
        # If the file that stores the notification sending time exist,
        # ...load the times and check if we already sent the notification at the same date before
        df_cvnst = pd.read_csv(cvnst_p, parse_dates=["DateTime"])
        last_date = df_cvnst["DateTime"].dt.date.iloc[-1] # the last date that we send the notification
        current_date = end_dt.date()
        if current_date == last_date: # check if the dates (only year, month, and day) match
            # We already sent crowd-verified push notifications to users today, do not send it again until next day
            log("Ignore this crowd-verified event because we already sent a push notification today", logger)
            return
    else:
        # If the file did not exist, create a new pandas Dataframe to store the time when we send notifications
        df_cvnst = pd.DataFrame(data=[], columns=["DateTime"])

    # Send crowd-verified push notification to users
    if ENABLE_RAKE_CALL:
        # NOTE: Python by default uses the sh terminal but we want it to use bash,
        # ...because "source" and "bundle" only works for bash on the Hal11 machine
        # ...(on the sh terminal we will want to use "." instead of "source")
        cmd = 'source /etc/profile ; cd /var/www/rails-apps/smellpgh/' + SERVER + '/current/ ; bundle exec rake firebase_push_notification:send_prediction_type2["/topics/SmellReports"] RAILS_ENV=' + SERVER + ' >> /var/www/smell-pittsburgh-prediction/py/prediction/data_production/crow_verified_push.log 2>&1'
        subprocess.call(["bash", "-c", cmd])

    # Create a CSV file that writes the time when the system send the push notification
    log("A crowd-verified push notification was sent to users", logger)
    df_cvnst = df_cvnst.append({"DateTime": end_dt}, ignore_index=True) # append a row to the pandas Dataframe
    df_cvnst.to_csv(cvnst_p, index=False) # re-write the Dataframe to a CSV file


if __name__ == "__main__":
    main(sys.argv)
