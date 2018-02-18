import pandas as pd
import numpy as np
from util import *
from datetime import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer

# Get data
# OUTPUT: raw esdr and smell data
def getData(
    out_p=None, # output file path
    start_dt=datetime(2016, 10, 6, 0), # starting date for the data
    end_dt=datetime(2018, 1, 25, 0), # ending data for the data
    logger=None):

    log("Get data...", logger)

    # Get and save ESDR data
    source = [
        [
            {"feed": "1", "channel": "PM25B_UG_M3"},
            {"feed": "1", "channel": "PM25T_UG_M3"}
        ],
        [{"feed": "1", "channel": "SO2_PPM,H2S_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}],
        [{"feed": "26", "channel": "OZONE_PPM,PM25B_UG_M3,PM10B_UG_M3"}],
        [{"feed": "27", "channel": "NO_PPB,NOY_PPB,CO_PPB,SO2_PPB"}],
        [{"feed": "28", "channel": "H2S_PPM,SO2_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}],
        [{"feed": "29", "channel": "PM10_UG_M3,PM25_UG_M3"}],
        [{"feed": "3", "channel": "SO2_PPM,SONICWD_DEG,SONICWS_MPH,SIGTHETA_DEG,PM10B_UG_M3"}],
        [{"feed": "23", "channel": "CO_PPM,PM10_UG_M3"}],
        [
            {"feed": "11067", "channel": "CO_PPB,NO2_PPB,NOX_PPB,NO_PPB,PM25T_UG_M3,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"},
            {"feed": "43", "channel": "CO_PPB,NO2_PPB,NOX_PPB,NO_PPB,PM25T_UG_M3,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}
        ],
        [{"feed": "3506", "channel": "PM2_5,OZONE"}],
        [{"feed": "5975", "channel": "PM2_5"}],
        [{"feed": "3508", "channel": "PM2_5"}],
        [{"feed": "24", "channel": "PM10_UG_M3"}]
    ]
    start_time = datetimeToEpochtime(start_dt) / 1000 # ESDR uses seconds
    end_time = datetimeToEpochtime(end_dt) / 1000 # ESDR uses seconds
    esdr_data = getEsdrData(source, start_time=start_time, end_time=end_time)
    df_esdr = mergeEsdrData(esdr_data)
    
    # Get smell reports (datetime object in "DateTime" column is in UTC tzinfo)
    df_smell = getSmellReports(start_time=start_time, end_time=end_time)
    df_smell = aggregateSmellData(df_smell)
    
    # Sync DateTime column in esdr and smell data
    if df_smell is not None:
        df_smell = pd.merge_ordered(df_esdr["DateTime"].to_frame(), df_smell, on="DateTime", how="left", fill_method=None)
        df_smell = df_smell.fillna(0)
    
    # Check directory and save file
    if out_p is not None:
        for p in out_p: checkAndCreateDir(p)
        df_esdr.to_csv(out_p[0], index=False)
        df_smell.to_csv(out_p[1], index=False)
        log("ESDR data created at " + out_p[0], logger)
        log("Smell data created at " + out_p[1], logger)
    return df_esdr, df_smell

def mergeEsdrData(data):
    # Resample data
    df = resampleData(data.pop(0)).reset_index()
    while len(data) != 0:
        df = pd.merge_ordered(df, resampleData(data.pop(0)).reset_index(), on="DateTime", how="outer", fill_method=None)

    # Fill NaN with -1
    df = df.fillna(-1)
    return df

def aggregateSmellData(df):
    if df is None: return None

    # Bag of words
    #bow = bagOfWords(df["smell_description"])
    
    # Select only the reports that are related to industrial smell
    #keywords_exclude = [
    #    "car","Car","trash","Trash","vehicle","Vehicle","paint",
    #    "Paint","garbage","Garbage","sewer","Sewer","sewage","Sewage"]
    #select_smell = ~df["smell_description"].str.contains("|".join(keywords_exclude)).fillna(False)
    #df = df[select_smell]
    
    # Select only the reports within the range of 3 and 5
    df = df[(df["smell_value"]>=3)&(df["smell_value"]<=5)]
    
    # If empty, return None
    if df.empty:
        return None

    # Group by zipcode and output a vector with zipcodes
    # TODO: need to merge the reports submitted by the same user in an hour
    data = []
    for z, df_z in df.groupby("zipcode"):
        # Select only smell values
        df_z = df_z["smell_value"]
        # Resample data
        df_z = resampleData(df_z, method="sum")
        df_z.name = z
        data.append(df_z)
    
    # Merge all
    df = data.pop(0).reset_index()
    while len(data) != 0:
        df = pd.merge_ordered(df, data.pop(0).reset_index(), on="DateTime", how="outer", fill_method=None)

    # Fill NaN with 0
    df = df.fillna(0)
    
    return df

def resampleData(df, method=None):
    df = df.copy(deep=True)
    df = epochtimeIdxToDatetime(df).resample("1h", label="right")
    if method == "sum":
        return df.sum()
    elif method == "count":
        return df.count()
    else:
        return df.mean()

# Convert a pandas dataframe to bag of words
def bagOfWords(df):
    # Preprocessing
    line = " ".join(df.fillna(""))
    line = re.sub("[^a-zA-Z]", " ", line) # replace non-letters
    line = re.sub("[ ]+", " ", line) # replace multiple white space
    line = [line.lower()] # to lower case

    # Bag of words
    model = CountVectorizer(stop_words="english")
    model.fit_transform(line)
    return model.vocabulary_
