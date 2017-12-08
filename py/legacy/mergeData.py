import numpy as np
import pandas as pd
from util import *
from sklearn import preprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
import re
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import Binarizer
import numbers

# Merge all data together into a dataset
# INPUT: seperated files
# OUTPUT: dataset, transformed dataset, sample_weights, plots
def mergeData(file_path_in, file_path_out, is_regression):
    print "Merge data..."

    # Check if directory exist
    for p in file_path_out:
        checkAndCreateDir(p)

    # Merge edsr data
    df_esdr = mergeEsdrData(file_path_in[0])

    # Aggregate ESDR data
    b_hr = 4 # how many hours to look back
    df_esdr = aggregateEsdrData(df_esdr, b_hr)
    idx = df_esdr["EpochTime"].values

    # Aggregate smell data
    b_hr = 4 # how many hours to look back
    f_hr = [-2, 2] # how many hours to look further
    bin_smell = None if is_regression else [10] # bin smell reports into labels or not
    df_smell, df_smell_raw, bow_smell = aggregateSmellData(file_path_in[1], idx, b_hr, f_hr, bin_smell, 3, 5)
    df_bow_smell = pd.DataFrame.from_dict(bow_smell, orient="index").reset_index()
    df_bow_smell.columns = ["word", "count"]

    # Merge esdr, smell, and tracker data
    df = pd.merge_ordered(df_esdr, df_smell, on="EpochTime", how="outer", fill_method=None)
    df = pd.merge_ordered(df, df_tracker, on="EpochTime", how="outer", fill_method=None)
    df = df.dropna().reset_index(drop=True)

    # Sort by epoch time
    df = df.sort_values("EpochTime")

    # Drop data points before Oct 6th 2016 (the app released date)
    df = df[df["EpochTime"] >= 1475726400].reset_index(drop=True)

    # Compute columns of days of the week and hours of the day
    df_datetime = pd.to_datetime(df["EpochTime"], unit="s")
    df_hd = df_datetime.dt.hour
    df_dw = df_datetime.dt.dayofweek

    # Compute sample weights
    df_w, df_freq = computeSampleWeights(df_smell_raw, df_hd, df_dw)

    # Drop the epochtime column
    df.drop("EpochTime", axis=1, inplace=True)

    # Prevent extreme small values
    df[df < 1e-6] = 0
    df_w[df_w < 1e-6] = 0

    # Transformed data points
    df_tran = pd.DataFrame(preprocessing.robust_scale(df), columns=df.columns)
    df_tran = df_tran.round(6)
    df_tran["NumberOfSmellReports"] = df["NumberOfSmellReports"]
   
    # Add days of week and hours of day
    df["DayOfWeek"] = df_dw
    df["HourOfDay"] = df_hd
    df_tran["DayOfWeek"] = df_dw
    df_tran["HourOfDay"] = df_hd

    # Write dataframe into a csv file
    df.to_csv(file_path_out[0])
    df_tran.to_csv(file_path_out[1])
    df_w.to_csv(file_path_out[2])
    df.corr().to_csv(file_path_out[3])
    df_freq.to_csv(file_path_out[4])
    df_bow_smell.to_csv(file_path_out[5])
    print "Dataset created at " + file_path_out[0]
    print "Transformed dataset created at " + file_path_out[1]
    print "Sample weights created at " + file_path_out[2]
    print "Original correlations created at " + file_path_out[3]
    print "Frequency of data points created at " + file_path_out[4]
    print "Bag of words for smell description created at " + file_path_out[5]

def computeSampleWeights(df_smell_raw, df_hd, df_dw):
    hd = "HourOfDay"
    dw = "DayOfWeek"

    # Compute hours of day and days of week
    df_t = pd.to_datetime(df_smell_raw["EpochTime"], unit="s")
    df_1 = df_t.dt.hour.rename(hd).reset_index(drop=True)
    df_2 = df_t.dt.dayofweek.rename(dw).reset_index(drop=True) 
    df = pd.concat([df_1, df_2], join="outer", axis=1)

    # Group data and fill in missing keys
    df_freq = df.groupby([hd, dw]).size()
    df_freq = df_freq.reindex(itertools.product(range(0,24), range(0,7)), fill_value=0)
    df_freq = df_freq.rename("count")

    # Compute weight
    df_w = df_freq.loc[zip(df_hd, df_dw)].reset_index(drop=True).rename("weight")
    df_w = df_w / df_w.max()

    return df_w.to_frame(), df_freq.reset_index()

def mergeEsdrData(file_path_in):
    # Read all csv files into pandas data frames
    file_name_in_all = getAllFileNamesInFolder(file_path_in)
    dfs = [pd.read_csv(file_path_in + f) for f in file_name_in_all if ".csv" in f]
    
    # Merge all pandas data frames
    df_merged = dfs.pop(0).sort_values("EpochTime")
    while len(dfs) != 0:
        df_merged = pd.merge_ordered(df_merged, dfs.pop(0), on="EpochTime", how="outer", fill_method=None)

    # Fill some NaN fields and drop rows
    #df_merged = df_merged.fillna(method="ffill", limit=1).dropna()
    df_merged = df_merged.dropna().reset_index(drop=True)

    return df_merged

def aggregateEsdrData(df_esdr, b_hr):
    # Convert epochtime to datetime
    df_tmp = df_esdr.copy(deep=True)
    df_epochtime = df_esdr["EpochTime"]
    df_tmp["DateTime"] = pd.to_datetime(df_epochtime, unit="s", utc=True)
    df_tmp.drop("EpochTime", axis=1, inplace=True)

    # Aggregate data
    b_hr_all = range(1, b_hr + 1)
    df_all = [df_epochtime]
    for b_hr in b_hr_all:
        # Add the previous readings
        df_previous = df_tmp.shift(b_hr)
        df_previous.drop("DateTime", axis=1, inplace=True)
        df_previous.columns += ".previous." + str(b_hr) + ".hour"
        df_all.append(df_previous)
        if b_hr <= 1: continue
        # Perform rolling mean and max
        # Because the government-operated sensor stations report data per hour, we want to
        # make sure that we get at least [b_hr] readings (e.g. for 3 hours, we want 3 readings)
        df_roll_max = df_tmp.rolling(str(b_hr) + "h", on="DateTime", min_periods=b_hr).max()
        df_roll_mean = df_tmp.rolling(str(b_hr) + "h", on="DateTime", min_periods=b_hr).mean()
        df_roll_std = df_tmp.rolling(str(b_hr) + "h", on="DateTime", min_periods=b_hr).std()
        df_roll_max.drop("DateTime", axis=1, inplace=True)
        df_roll_mean.drop("DateTime", axis=1, inplace=True) 
        df_roll_std.drop("DateTime", axis=1, inplace=True) 
        df_roll_max.columns += ".max.of.last." + str(b_hr) + ".hours"
        df_roll_mean.columns += ".mean.of.last." + str(b_hr) + ".hours"
        df_roll_std.columns += ".std.of.last." + str(b_hr) + ".hours"
        df_all.append(df_roll_max)
        df_all.append(df_roll_mean)
        df_all.append(df_roll_std)
    
    # Combine dataframes
    df_tmp.drop("DateTime", axis=1, inplace=True)
    df_tmp.columns += ".current"
    df_all.append(df_tmp)
    df = pd.concat(df_all, join="outer", axis=1)

    # Fill some NaN fields and drop rows
    df = df.dropna().reset_index(drop=True)

    return df

def aggregateSmellData(file_path_in, idx, b_hr, f_hr, bin_smell, min_smell_value, max_smell_value):
    # Read csv file into a pandas data frame
    df = pd.read_csv(file_path_in)
   
    # Select only the reports within the range of min_smell_value and max_smell_value
    df = df[(df["SmellValue"]>=min_smell_value)&(df["SmellValue"]<=max_smell_value)].reset_index(drop=True)

    # This is for finding words related to industrial smell
    bow_smell = bagOfWords(df["SmellDescription"])
    #bow_smell = dict([(k,v) for k, v in bow_smell.items() if v > 300])
    #for k, v in bow_smell.items(): print k, v
    keywords = ["sulfer", "coking", "ammonia", "propane",
        "smog", "chemically", "methane", "sooty", "gasoline",
        "acid", "poisonous", "sulphurous", "rot", "stink",
        "sufer", "particulate", "smoky", "smoke", "oily",
        "toxins", "sour", "tar", "sulpherous", "chemical",
        "clairton", "sulfury", "soot", "sulfure", "toxic",
        "acetone", "noxous", "smelting", "industrial", "metal"
        "sufurous", "sulfuric", "sulphur", "sulfurous", "slag",
        "coal", "metallic", "mills", "coke", "insustrial",
        "scorched", "chemicsls", "plume", "sulfur", "steel",
        "claritin", "acidic", "thomson", "hydrocarbon", "crude",
        "edgar", "acrid", "polution", "pollution", "sulphuric",
        "indutrial", "voc", "indusrial", "petrochemical", "ozone",
        "industry", "sulfar", "noxious", "sulpher", "induatrial",
        "aromatic", "clariton", "None"]

    keywords_exclude = ["wood", "Wood", "car", "Car", "trash",
        "Trash", "vehicle", "Vehicle", "paint", "Paint"
        "garbage", "Garbage", "sewer", "Sewer", "sewage",
        "Sewage"]

    # Select only the reports that are related to industrial smell
    #df = df[df["SmellDescription"].str.contains("|".join(keywords))]
    #df = df[~df["SmellDescription"].str.contains("|".join(keywords_exclude))]

    # Sort by epoch time
    df = df.sort_values("EpochTime")
    df_t = df["EpochTime"]
    
    # Aggregate data
    dict_all = {"EpochTime": idx}
    if b_hr is not None:
        b_hr_all = range(1, b_hr + 1)
        for b_hr in b_hr_all:
            # Count the number of smell reports in previous [b_hr] hours
            b_val = countByIndex(idx, df_t.values, -b_hr*3600)
            b_key = "NumberOfSmellReports.of.last." + str(b_hr) + ".hours"
            dict_all[b_key] = b_val

    # Count the number of smell reports in future f_hr hours
    if f_hr is not None:
        if isinstance(f_hr, numbers.Number):
            f_val = countByIndex(idx, df_t.values, f_hr*3600)
        elif type(f_hr) == list:
            f_val = countByIndex(idx, df_t.values, [f_hr[0]*3600, f_hr[1]*3600])
        if bin_smell is not None:
            f_val = binTransform(f_val, bin_smell)
        dict_all["NumberOfSmellReports"] = f_val

    # Create dataframe
    df_smell = pd.DataFrame(dict_all)

    return df_smell, df, bow_smell

# Usage:
# df_tracker = aggregateTrackerData(file_path_in[2], idx, 4)
def aggregateTrackerData(file_path_in, idx, b_hr):
    # Read all csv files into one pandas data frames
    file_name_in_all = getAllFileNamesInFolder(file_path_in)
    df = pd.concat([pd.read_csv(file_path_in + file_name_in) for file_name_in in file_name_in_all])
    
    # Select the user ID that is not "undefined" and does not begin with "BA"
    df = df[(df["UserID"] != "undefined") & ~(df["UserID"].str.startswith("BA"))]

    # Select the rows that the hit and data timestamps are on the same day
    df = df[(df["DataTimestamp"] != "-1") & (df["DataTimestamp"] != "undefined")]
    tz = "US/Eastern"
    df_hit_dt = pd.to_datetime(df["HitTimestamp"],unit="ms").dt.tz_localize("UTC").dt.tz_convert(tz)
    df_data_dt = pd.to_datetime(df["DataTimestamp"],unit="ms").dt.tz_localize("UTC").dt.tz_convert(tz)
    df["HitTimeString"] = df_hit_dt.dt.strftime("%Y-%m-%d")
    df["DataTimeString"] = df_data_dt.dt.strftime("%Y-%m-%d")
    df = df[df["HitTimeString"] == df["DataTimeString"]]

    # Sort by timestamp and convert it to epochtime in seconds
    df = df.sort_values("HitTimestamp")
    df = (df["HitTimestamp"]/1000).round().astype(int)
   
    # Aggregate data
    b_hr_all = range(1, b_hr + 1)
    dict_all = {"EpochTime": idx}
    for b_hr in b_hr_all:
        # Count the number of GA hits in previous [b_hr] hours
        b_val = countByIndex(idx, df.values, -b_hr*3600)
        b_key = "GoogleAnalyticsHitCounts.of.last." + str(b_hr) + ".hours"
        dict_all[b_key] = b_val

    return pd.DataFrame(dict_all)

# Count the frequency based on epochtime offset (e.g. how many smell reports in a previous hour)
# INPUT:
# - idx_ref: the epochtime index that we want to reference from
# - idx_data: the epochtime index that want to count the frequency
# - offset: the epochtime offset in seconds
def countByIndex(idx_ref, idx_data, offset):
    result = np.zeros(idx_ref.size)
    c = 0
    for i in idx_ref:
        if isinstance(offset, numbers.Number):
            if offset > 0:
                result[c] = np.sum(np.logical_and(idx_data>=i, idx_data<=i+offset))
            elif offset < 0:
                result[c] = np.sum(np.logical_and(idx_data>=i+offset, idx_data<=i))
        elif type(offset) == list:
            result[c] = np.sum(np.logical_and(idx_data>=i+offset[0], idx_data<=i+offset[1]))
        c += 1
    return result

# Convert a pandas dataframe to bag of words
def bagOfWords(df):
    # Preprocessing
    line = " ".join(df)
    line = re.sub("[^a-zA-Z]", " ", line) # replace non-letters
    line = re.sub("[ ]+", " ", line) # replace multiple white space
    line = [line.lower()] # to lower case
    
    # Bag of words
    model = CountVectorizer(stop_words="english")
    model.fit_transform(line)

    return model.vocabulary_ 

# Bin and transform data
def binTransform(X, b):
    if type(b) == int:
        hist, bin_edges = np.histogram(X, bins=b)
        return np.digitize(X, bin_edges)
    elif type(b) == list:
        return np.digitize(X, b)
