import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibrateAwairData import *

import sys
sys.path.append("../analysis/")
from mergeData import *
from util import *

# This function is used for testing the calibration algorithm for AWAIR sensors
def main():
    p = "../../data/"
    file_path_in = [
        p + "calibration/awair/",
        p + "raw/smell/smell-reports.csv"]
    file_path_out = p + "calibration/"

    # Check output directory
    checkAndCreateDir(file_path_out)

    # Read and merge file
    df = mergeEsdrData(file_path_in[0])
    start_dt = dt.datetime(2016, 11, 21, 0) # year, month, day, hour
    start_t = datetimeToEpochtime(start_dt) / 1000 # ESDR uses seconds
    df = df[df["EpochTime"]>=start_t].reset_index(drop=True)
    b_days = 7
    df_cal = rollSubtractMin(df, b_days)
    df_t = df["EpochTime"]
    df.drop("EpochTime", axis=1, inplace=True)
    df_cal.drop("EpochTime", axis=1, inplace=True)

    # Compute differences
    a = df.columns
    b = df_cal.columns
    c = "absolute.difference.of.("
    df_diff_ori = pd.DataFrame(df.iloc[:,1]-df.iloc[:,0], columns=[c+a[0]+").and.("+a[1]+")"]).abs()
    df_diff_cal = pd.DataFrame(df_cal.iloc[:,1]-df_cal.iloc[:,0], columns=[c+b[0]+").and.("+b[1]+")"]).abs()
    df_diff_1 = pd.concat([df_diff_ori, df_diff_cal], join="outer", axis=1)
    df_diff_2 = pd.DataFrame(df.values-df_cal.values, columns=df.columns).abs()
    df_diff_2.columns = [c+a[0]+").and.("+b[0]+")", c+a[1]+").and.("+b[1]+")"]

    # Aggregate smell data
    b_hr = None # how many hours to look back
    f_hr = [-1.5, 1.5] # how many hours to look further
    df_s_3, _, _ = aggregateSmellData(file_path_in[1], df_t.values, b_hr, f_hr, None, 3, 3)
    df_s_4, _, _ = aggregateSmellData(file_path_in[1], df_t.values, b_hr, f_hr, None, 4, 4)
    df_s_5, _, _ = aggregateSmellData(file_path_in[1], df_t.values, b_hr, f_hr, None, 5, 5)
    df_s_345, _, _ = aggregateSmellData(file_path_in[1], df_t.values, b_hr, f_hr, None, 3, 5)
    a = "NumberOfSmellReports"
    a_new = a + ".between.last.90.mins.and.future.90.mins.with.smell.rating."
    df_s_3.rename(columns={a: a_new + "3"}, inplace=True)
    df_s_4.rename(columns={a: a_new + "4"}, inplace=True)
    df_s_5.rename(columns={a: a_new + "5"}, inplace=True)
    df_s_345.rename(columns={a: a_new + "3.and.4.and.5"}, inplace=True)
    df_s_3 = df_s_3[a_new + "3"]
    df_s_4 = df_s_4[a_new + "4"]
    df_s_5 = df_s_5[a_new + "5"]
    df_s_345 = df_s_345[a_new + "3.and.4.and.5"]
    df_smell = pd.concat([df_s_3, df_s_4, df_s_5], join="outer", axis=1)

    # Output correlation
    df_awair_1 = pd.concat([df.iloc[:,0], df_cal.iloc[:,0]], join="outer", axis=1)
    df_awair_2 = pd.concat([df.iloc[:,1], df_cal.iloc[:,1]], join="outer", axis=1)
    corr_full_path = file_path_out + "calibrated_corr.csv"
    pd.concat([df_s_345, df_awair_1, df_awair_2], join="outer", axis=1).corr().to_csv(corr_full_path)
    print "Calibrated Awair correlation saved at " + corr_full_path
    
    # Output graph
    df_awair_1.index = pd.to_datetime(df_t, unit="s")
    df_awair_2.index = pd.to_datetime(df_t, unit="s")
    df_smell.index = pd.to_datetime(df_t ,unit="s")
    df_diff_1.index = pd.to_datetime(df_t ,unit="s")
    df_diff_2.index = pd.to_datetime(df_t ,unit="s")
    fig, axes = plt.subplots(5, 1, figsize=(200, 14), gridspec_kw = {"height_ratios":[2,1,1,1,1]})
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    df_smell.plot.area(ax=ax1, fontsize=15, rot=0, sharex=True, alpha=0.8, color=["#FF7E00","#FF0000","#7E0023"])
    #df_awair.plot.area(stacked=False, subplots=True, ax=ax2, fontsize=15, rot=0, sharex=True)
    df_awair_1.plot(ax=ax2, fontsize=15, rot=0, sharex=True, alpha=0.8)
    df_awair_2.plot(ax=ax3, fontsize=15, rot=0, sharex=True, alpha=0.8)
    df_diff_2.plot(ax=ax4, fontsize=15, rot=0, sharex=True, alpha=0.8)
    df_diff_1.plot(ax=ax5, fontsize=15, rot=0, sharex=True, alpha=0.8)
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
    fig.tight_layout()
    graph_full_path = file_path_out + "calibrated_debug.png"
    fig.savefig(graph_full_path, dpi=150)
    print "Calibrated Awair graph saved at " + graph_full_path

if __name__ == "__main__":
    main()
