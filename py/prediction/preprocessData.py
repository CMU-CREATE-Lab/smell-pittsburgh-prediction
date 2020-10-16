import pandas as pd
from util import log, getAllFileNamesInFolder, checkAndCreateDir, epochtimeIdxToDatetime
import re
from sklearn.feature_extraction.text import CountVectorizer


"""
Preprocess data
INPUT: raw esdr and smell data
OUTPUT: preprocessed esdr and smell data
"""
def preprocessData(
    df_esdr_array_raw=None, # raw esdr dataframes
    df_smell_raw=None, # raw smell dataframe
    in_p=None, # input file path
    out_p=None, # output file path
    logger=None):

    log("Preprocess data...", logger)

    if df_esdr_array_raw is None or df_smell_raw is None:
        if in_p is not None:
            df_esdr_array_raw = []
            for f in getAllFileNamesInFolder(in_p[0]):
                if ".csv" in f:
                    df_esdr_array_raw.append(pd.read_csv(in_p[0] + f, index_col="EpochTime"))
            df_smell_raw = pd.read_csv(in_p[1], index_col="EpochTime")
        else:
            if df_esdr_array_raw is None:
                log("ERROR: no data, return None.", logger)
                return None

    # Merge esdr data
    df_esdr = mergeEsdrData(df_esdr_array_raw)

    # Aggregate smell reports (datetime object in "DateTime" column is in UTC tzinfo)
    df_smell = aggregateSmellData(df_smell_raw)

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
    df = resampleData(data.pop(0), method="mean").reset_index()
    while len(data) != 0:
        df = pd.merge_ordered(df, resampleData(data.pop(0), method="mean").reset_index(),
            on="DateTime", how="outer", fill_method=None)

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
    # TODO: need to merge the reports submitted by the same user in an hour with different weights
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


def resampleData(df, method="mean", rule="60Min"):
    df = df.copy(deep=True)
    # Because we want data from the past, so label need to be "right"
    df = epochtimeIdxToDatetime(df).resample(rule, label="right")
    if method == "sum":
        return df.sum()
    elif method == "count":
        return df.count()
    elif method == "mean":
        return df.mean()
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
