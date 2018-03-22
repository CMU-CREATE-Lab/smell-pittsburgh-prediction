import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pandas.plotting import parallel_coordinates
from joblib import Parallel, delayed
import gc
from itertools import combinations
from util import *
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from selectFeatures import *
import itertools

# Plot features
# INPUT: features
# OUTPUT: plots
def plotFeatures(
    in_p, # features and original data (in pandas dataframe format)
    out_p_root, # root directory for outputing graphs and files
    is_regr=False, # regression or classification
    plot_time=True, # plot time series data
    plot_hist=True, # plot histogram
    plot_pair=False, # plot pairs (scatter plots)
    plot_freq=True, # plot the frequency of smell reports
    plot_corr=True, # plot correlation matrix
    plot_pca=True, # plot principle components
    logger=None):

    log("Plot features...", logger)

    # Create file out folders
    out_p = [
        out_p_root + "plot/time/",
        out_p_root + "plot/hist/",
        out_p_root + "plot/pair/",
        out_p_root + "plot/"]

    # Create folder for saving files
    for f in out_p:
        checkAndCreateDir(f)

    # Read features
    df = pd.read_csv(in_p)

    # Select features before plotting
    label = "smell_value"
    df_Y = df[label].copy(deep=True)
    df_X = df.drop([label], axis=1).copy(deep=True)
    df_X, df_Y = selectFeatures(df_X, df_Y, is_regr)
    df = df_X.join(df_Y)

    # Histogram or time-series of a single feature, or pairs of features
    with Parallel(n_jobs=-2) as parallel:
        if plot_time:
            log("Plot time series...", logger)
            h = "Time series of "
            parallel(delayed(plotTime)(df[v], v, h, out_p[0], False) for v in df.columns)
        if plot_hist:
            log("Plot histograms...", logger)
            h = "Histogram of "
            parallel(delayed(plotHist)(df[v], v, h, out_p[1], False) for v in df.columns)
        if plot_pair:
            log("Plot pairs...", logger)
            p = combinations(df.columns, 2)
            h = ""
            parallel(delayed(plotPair)(df[[v1,v2]], v1, v2, h, out_p[2], False) for (v1, v2) in p)

    # Plot the frequency of smell reports by days of week and hours of day
    if plot_freq:
        log("Plot frequency...", logger)
        df_freq = computeFrequency(df)
        fig, ax1 = plt.subplots(1, 1, figsize=(19, 6))
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("right", size="2%", pad=0.4)
        df_hd = df_freq["HourOfDay"].values
        df_dw = df_freq["DayOfWeek"].values
        df_c = df_freq["smell_value"].values
        mat = np.zeros((7,24))
        for hd, dw, c in zip(df_hd, df_dw, df_c):
            mat[(dw, hd)] = c
        df_tmp = pd.DataFrame(data=mat)

        y_l = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        x_l = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00",
            "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
            "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00"]
        sns.heatmap(df_tmp,ax=ax1,cbar_ax=ax2,cmap=None,linewidths=0.1,annot=True,fmt="g",xticklabels=x_l,yticklabels=y_l)

        for item in ax1.get_yticklabels():
            item.set_rotation(0)
        for item in ax1.get_xticklabels():
            item.set_rotation(0)

        ax1.set_ylabel("Day of week", fontsize=14)
        ax1.set_xlabel("Hour of day", fontsize=14)
        plt.suptitle("Distribution of Smell Reports over Time", fontsize=20)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        fig.savefig(out_p[3] + "frequency.png", dpi=150)
        fig.clf()
        plt.close()

    # Plot correlation matrix
    if plot_corr:
        log("Plot correlation...", logger)
        fig, ax = plt.subplots(figsize=(10, 8))
        df_corr = df.corr()
        df_corr.to_csv(out_p[3] + "correlation.csv")
        im = ax.imshow(df_corr, cmap=plt.get_cmap("brg"), interpolation="nearest",vmin=-1, vmax=1)
        fig.colorbar(im)
        fig.tight_layout()
        plt.suptitle("Correlation matrix", fontsize=18)
        fig.subplots_adjust(top=0.92)
        fig.savefig(out_p[3] + "correlation.png", dpi=150)
        fig.clf()
        plt.close()

    # Plot dimension reduction
    if plot_pca:
        log("Plot PCA...", logger)
        plotPCA(df, label, out_p[3], is_regr, use_kernel=False)
        plotPCA(df, label, out_p[3], is_regr, use_kernel=True)

def plotPCA(df, label, out_p, is_regr, use_kernel=False):
        n_c = 8
        if use_kernel:
            pca = KernelPCA(n_components=n_c, kernel="rbf")
        else:
            pca = PCA(n_components=n_c)
        dff = df.drop(label, axis=1)
        X = pca.fit_transform(dff)
        fig = plt.figure(figsize=(18, 11), dpi=150)
        Y = df[label]
        if use_kernel:
            r = pca.lambdas_
            r = np.round(r/sum(r), 3)
        else:
            r = np.round(pca.explained_variance_ratio_, 3)
        dot_size = 15
        dot_cmap = "brg"
        w = 7
        h = 4
        c = 1
        for i in range(0,n_c):
            for j in range(0,n_c):
                if j <= i:
                    continue
                else:
                    plt.subplot(h, w, c)
                    if is_regr:
                        plt.scatter(X[:,i], X[:,j], c=Y, s=dot_size, alpha=0.2, cmap=dot_cmap)
                    else:
                        c0_i = (Y == 0)
                        c1_i = (Y == 1)
                        plt.scatter(X[c0_i,i], X[c0_i,j], c="b", s=dot_size, alpha=0.1, cmap=dot_cmap)
                        plt.scatter(X[c1_i,i], X[c1_i,j], c="r", s=dot_size, alpha=0.2, cmap=dot_cmap)
                    plt.xlabel("Component " + str(i) + " (" + str(r[i]) + ")")
                    plt.ylabel("Component " + str(j) + " (" + str(r[j]) + ")")
                    c += 1
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        if use_kernel:
            plt.suptitle("Kernel PCA = " + str(r), fontsize=22)
            fig.savefig(out_p + "kernel_pca.png")
        else:
            plt.suptitle("PCA = " + str(r), fontsize=22)
            fig.savefig(out_p + "pca.png")
        fig.clf()
        plt.close()

def plotTime(df_v, v, title_head, out_p, is_data_transformed):
    print title_head + v
    fig = plt.figure(figsize=(40, 8), dpi=150)
    df_v.plot(alpha=0.5, title=title_head+v)
    if is_data_transformed:
        out_p += "time===transformed===" + v + ".png"
    else:
        out_p += "time===" + v + ".png"
    fig.tight_layout()
    fig.savefig(out_p)
    fig.clf()
    plt.close()
    gc.collect()

def plotHist(df_v, v, title_head, out_p, is_data_transformed):
    print title_head + v
    fig = plt.figure(figsize=(8, 8), dpi=150)
    df_v.plot.hist(alpha=0.5, bins=30, title=title_head+v)
    if is_data_transformed:
        out_p += "transformed===" + v + ".png"
    else:
        out_p += v + ".png"
    fig.tight_layout()
    fig.savefig(out_p)
    fig.clf()
    plt.close()
    gc.collect()

def plotPair(df_v, v1, v2, title_head, out_p, is_data_transformed):
    print title_head + v1 + " === " + v2
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(df_v[v1], df_v[v2], s=10, alpha=0.4)
    plt.title(title_head + v1 + " === " + v2)
    plt.xlabel(v1)
    plt.ylabel(v2)
    if is_data_transformed:
        out_p += "transformed===" + v1 + "===" + v2 + ".png"
    else:
        out_p += v1 + "===" + v2 + ".png"
    fig.tight_layout()
    fig.savefig(out_p)
    fig.clf()
    plt.close()
    gc.collect()

def computeFrequency(df):
    df = df.copy(deep=True)
    df_freq = df.groupby(["HourOfDay", "DayOfWeek"]).mean()
    df_freq = np.round(df_freq, 2)
    return df_freq.reset_index()
