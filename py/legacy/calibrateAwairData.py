import pandas as pd
import numpy as np
from util import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as dates

def calibrateAwairData(file_path_in, file_path_out):
    print "Calibrate Awair sensor data..."

    # Get all files
    file_name_in_all = getAllFileNamesInFolder(file_path_in[0])

    # Check output directory
    for i in file_path_out:
        checkAndCreateDir(i)

    # Calibrate each file
    b_days = 7 # number of days to look back
    for f in file_name_in_all:
        if not ".csv" in f: continue
        df = pd.read_csv(file_path_in[0] + f)
        df_calibrated = rollSubtractMin(df, b_days)

        # Output file
        base_name = getBaseName(f, False)
        file_full_path = file_path_out[0] + "calibrated_" + base_name + ".csv"
        df_calibrated.to_csv(file_full_path, index=False)
        print "Calibrated Awair data saved at " + file_full_path

        # Merge data
        df_compare = pd.concat([df, df_calibrated.drop("EpochTime", axis=1)], join="outer", axis=1)

        # Output graphs
        df_compare.index = pd.to_datetime(df_compare["EpochTime"], unit="s")
        df_compare.drop("EpochTime", axis=1, inplace=True)
        p = df_compare.plot(subplots=True, figsize=(200, 4), fontsize=15, rot=0)
        for ax in plt.gcf().axes:
            ax.legend(loc=1, fontsize=15)
            #ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1), interval=1))
            ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
            ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%a'))
            ax.xaxis.grid(True, which="minor")
            #ax.yaxis.grid()
            ax.xaxis.set_major_locator(dates.MonthLocator())
            ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b %Y'))
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig = p[0].get_figure()
        fig.tight_layout()
        graph_full_path = file_path_out[1] + "calibrated_" + base_name + ".png"
        fig.savefig(graph_full_path, dpi=150)
        print "Calibrated Awair graph saved at " + graph_full_path

# For each data point, find the minimum value of the previous [b_days] days, and subtract it
def rollSubtractMin(df, b_days):
    df_epochtime = df["EpochTime"]
    df = df.copy(deep=True)
    df["DateTime"] = pd.to_datetime(df_epochtime, unit="s", utc=True)
    df.drop("EpochTime", axis=1, inplace=True)
    df_roll_min = df.rolling(str(b_days) + "d", on="DateTime").min()
    df_roll_min.drop("DateTime", axis=1, inplace=True)
    df.drop("DateTime", axis=1, inplace=True)
    df_calibrated = df - df_roll_min
    df_calibrated.columns = "calibrated." + df_calibrated.columns
    df_calibrated = pd.concat([df_epochtime, df_calibrated], join="outer", axis=1)
    return df_calibrated

# Find the minimum value based on epochtime offset (e.g. what is the min value for the previous week?)
# INPUT:
# - idx_ref: the epochtime index that we want to reference from
# - idx_data: the epochtime index that want to find the minimum value
# - data: the data
# - offset: the epochtime offset in seconds
def minByIndex(idx_ref, idx_data, data, offset):
    result = np.zeros(idx_ref.size)
    c = 0
    for i in idx_ref:
        if offset > 0:
            result[c] = np.amin(data[np.logical_and(idx_data>=i, idx_data<=i+offset)])
        elif offset < 0:
            result[c] = np.amin(data[np.logical_and(idx_data>=i+offset, idx_data<=i)])
        c += 1
    return result

def getBaseName(path, **options):
    """Get the base name of a file path"""
    with_extension = options["with_extension"] if "with_extension" in options else False
    do_strip = options["do_strip"] if "do_strip" in options else True
    base_name = os.path.basename(path)
    if with_extension:
        return base_name
    base_name_no_ext = os.path.splitext(base_name)[0]
    if do_strip:
        return base_name_no_ext.strip()
    return base_name_no_ext
