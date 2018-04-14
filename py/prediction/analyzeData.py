import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc
from util import *
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from computeFeatures import *
from Interpreter import *
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import SpectralEmbedding
from copy import deepcopy
from crossValidation import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
from collections import Counter
import re

# Analyze data
def analyzeData(
    in_p=None, # input path for raw esdr and smell data
    out_p_root=None, # root directory for outputing files
    logger=None):

    log("Analyze data...", logger)
    out_p = out_p_root + "analysis/"
    checkAndCreateDir(out_p)

    # Plot features
    #plotFeatures(in_p, out_p_root, logger)

    # Plot aggregated smell data
    #plotAggrSmell(in_p, out_p, logger)

    # Plot dimension reduction
    #plotLowDimensions(in_p, out_p, logger)

    # Correlational study
    #corrStudy(in_p, out_p, logger=logger)

    # Interpret model
    #interpretModel(in_p, out_p, logger=logger)

def interpretModel(in_p, out_p, logger):
    # Load time series data
    df_esdr = pd.read_csv(in_p[0], parse_dates=True, index_col="DateTime")
    df_smell = pd.read_csv(in_p[1], parse_dates=True, index_col="DateTime")

    # Select variables based on prior knowledge
    print "Select variables based on prior knowledge..."
    want = [
        "3.feed_26.OZONE_PPM", # Lawrenceville ACHD
        "3.feed_26.SONICWS_MPH",
        "3.feed_26.SONICWD_DEG",
        "3.feed_26.SIGTHETA_DEG",
        "3.feed_28.H2S_PPM", # Liberty ACHD
        "3.feed_28.SIGTHETA_DEG",
        "3.feed_28.SONICWD_DEG",
        "3.feed_28.SONICWS_MPH",
        "3.feed_23.PM10_UG_M3", # Flag Plaza ACHD
        "3.feed_11067.SIGTHETA_DEG..3.feed_43.SIGTHETA_DEG", # Parkway East ACHD
        "3.feed_11067.SONICWD_DEG..3.feed_43.SONICWD_DEG",
        "3.feed_11067.SONICWS_MPH..3.feed_43.SONICWS_MPH"
    ]
    df_esdr_cp = df_esdr
    df_esdr = pd.DataFrame()
    for col in df_esdr_cp.columns:
        if col in want:
            print "\t" + col
            df_esdr[col] = df_esdr_cp[col]
    
    # Check if time series variables are stationary using Dickey-Fuller test
    if False:
        for col in df_esdr.columns:
            r = adfuller(df_esdr[col], regression="ctt")
            print "p-value: %.3f -- %s" % (r[1], col)

    # Compute cross-correlation between variables
    if False:
        L = len(df_esdr.columns)
        for i in range(0, L-1):
            col_i = df_esdr.columns[i]
            for j in range(i+1, L):
                col_j = df_esdr.columns[j]
                x_i, x_j = df_esdr[col_i], df_esdr[col_j]
                pair = col_i + " === " + col_j
                max_lag = 3
                cc = computeCrossCorrelation(x_i, x_j, max_lag=max_lag)
                all_lag = np.array(range(0, max_lag*2 + 1)) - max_lag
                max_cr_idx = np.argmax(abs(cc))
                max_cr_val = cc[max_cr_idx]
                max_cr_lag = all_lag[max_cr_idx]
                pair = col_i + " === " + col_j
                print "(max_corr=%.3f, lag=%d)  %s" % (max_cr_val, max_cr_lag, pair)
                #r = grangercausalitytests(df_esdr[[col_i, col_j]], maxlag=max_lag, verbose=False)
                #for key in r.keys(): print "\t (ts, p_value, dof, lag) = %.2f, %.3f, %d, %d" % r[key][0]['params_ftest']
                #print "\t(corr, p_value) = %.2f, %.2f" % pearsonr(x_i, x_j)
    
    # Interpret data
    df_esdr = df_esdr.reset_index()
    df_smell = df_smell.reset_index()
    df_X, df_Y, df_C = computeFeatures(df_esdr=df_esdr, df_smell=df_smell, f_hr=8, b_hr=2, thr=40, is_regr=False,
        add_inter=True, add_roll=False, add_diff=False, logger=logger)
    model = Interpreter(df_X=df_X, df_Y=df_Y, out_p=out_p, logger=logger)
    df_Y = model.getFilteredLabels()
    df_X = model.getSelectedFeatures()
    for m in ["LG", "DT"]:
        start_time_str = datetime.now().strftime("%Y-%d-%m-%H%M%S")
        lg = generateLogger(out_p + "log/" + m + "-" + start_time_str + ".log", format=None)
        crossValidation(df_X=df_X, df_Y=df_Y, df_C=df_C, out_p_root=out_p, method=m, is_regr=False, logger=lg)

def computeCrossCorrelation(x, y, max_lag=None):
    n = len(x)
    xo = x - x.mean()
    yo = y - y.mean()
    cv = np.correlate(xo, yo, "full") / n
    cc = cv / (np.std(x) * np.std(y))
    if max_lag > 0:
        cc = cc[n-1-max_lag:n+max_lag]
    return cc

# Correlational study
def corrStudy(in_p, out_p, logger):
    log("Compute correlation of lagged X and current Y...", logger)
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=0, thr=40, is_regr=True,
         add_inter=False, add_roll=False, add_diff=False, logger=logger)
    Y = df_Y.squeeze()
    Y = Y - Y.mean()
    max_t_lag = 25 # the maximum time lag
    df_corr = pd.DataFrame()
    for c in df_X.columns:
        if c in ["Day", "DayOfWeek", "HourOfDay"]: continue
        s = []
        X = df_X[c]
        for i in range(0, max_t_lag+1):
            s.append(np.round(Y.corr(X.shift(i)), 3))
        df_corr[c] = pd.Series(data=s)
    df_corr = df_corr.round(3)
    df_corr.to_csv(out_p + "corr_with_time_lag.csv")

    # Plot graph
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(df_corr, cmap=plt.get_cmap("RdBu"), interpolation="nearest",vmin=-0.6, vmax=0.6)
    plt.ylabel("Time lag (hours)")
    fig.colorbar(im)
    fig.tight_layout()
    plt.suptitle("Correlation with time lag", fontsize=18)
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_p + "corr_with_time_lag.png", dpi=150)
    fig.clf()
    plt.close()

def plotAggrSmell(in_p, out_p, logger):
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=None, b_hr=0, thr=40, is_regr=True,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    
    # Plot the distribution of smell values by days of week and hours of day
    plotDayHour(df_X, df_Y, out_p, logger)

def plotDayHour(df_X, df_Y, out_p, logger):
    log("Plot the distribution of smell over day and hour...", logger)
    df = pd.DataFrame()
    df["HourOfDay"] = df_X["HourOfDay"]
    df["DayOfWeek"] = df_X["DayOfWeek"]
    df["smell"] = df_Y["smell"]
    df = df.groupby(["HourOfDay", "DayOfWeek"]).mean()
    df = df.round(2).reset_index()

    df_hd = df["HourOfDay"].values
    df_dw = df["DayOfWeek"].values
    df_c = df["smell"].values
    mat = np.zeros((7,24))
    for hd, dw, c in zip(df_hd, df_dw, df_c):
        mat[(dw, hd)] = c

    y_l = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    x_l = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00",
        "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
        "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    df_day_hour = pd.DataFrame(data=mat, columns=x_l, index=y_l)
    df_day_hour.to_csv(out_p + "smell_day_hour.csv")

    fig, ax1 = plt.subplots(1, 1, figsize=(19, 6))
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="2%", pad=0.4)
    sns.heatmap(df_day_hour, ax=ax1, cbar_ax=ax2, cmap="Blues", vmin=0, vmax=7,
        linewidths=0.1, annot=True, fmt="g", xticklabels=x_l, yticklabels=y_l)

    for item in ax1.get_yticklabels():
        item.set_rotation(0)
    for item in ax1.get_xticklabels():
        item.set_rotation(0)

    ax1.set_ylabel("Day of week", fontsize=14)
    ax1.set_xlabel("Hour of day", fontsize=14)
    plt.suptitle("Distribution of smell reports over Time", fontsize=20)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_p + "smell_day_hour.png", dpi=150)
    fig.clf()
    plt.close()

def plotFeatures(in_p, out_p_root, logger):
    plot_time_hist_pair = True
    plot_corr = True

    # Create file out folders
    out_p = [
        out_p_root + "analysis/time/",
        out_p_root + "analysis/hist/",
        out_p_root + "analysis/pair/",
        out_p_root + "analysis/"]

    # Create folder for saving files
    for f in out_p:
        checkAndCreateDir(f)

    # Compute features
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=0, thr=40, is_regr=True,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    df_Y = pd.to_numeric(df_Y.squeeze())

    # Plot feature histograms, or time-series, or pairs of (feature, label)
    if plot_time_hist_pair:
        with Parallel(n_jobs=-1) as parallel:
            # Plot time series
            log("Plot time series...", logger)
            h = "Time series "
            parallel(delayed(plotTime)(df_X[v], h, out_p[0]) for v in df_X.columns)
            plotTime(df_Y, h, out_p[0])
            # Plot histograms
            log("Plot histograms...", logger)
            h = "Histogram "
            parallel(delayed(plotHist)(df_X[v], h, out_p[1]) for v in df_X.columns)
            plotHist(df_Y, h, out_p[1])
            # Plot pairs of (feature, label)
            log("Plot pairs...", logger)
            h = "Pairs "
            parallel(delayed(plotPair)(df_X[v], df_Y, h, out_p[2]) for v in df_X.columns)

    # Plot correlation matrix
    if plot_corr:
        log("Plot correlation matrix of predictors...", logger)
        plotCorrMatirx(df_X, out_p[3])

    log("Finished plotting features", logger)

def plotTime(df_v, title_head, out_p):
    v = df_v.name
    print title_head + v
    fig = plt.figure(figsize=(40, 8), dpi=150)
    df_v.plot(alpha=0.5, title=title_head)
    fig.tight_layout()
    fig.savefig(out_p + "time===" + v + ".png")
    fig.clf()
    plt.close()
    gc.collect()

def plotHist(df_v, title_head, out_p, bins=30):
    v = df_v.name
    print title_head + v
    fig = plt.figure(figsize=(8, 8), dpi=150)
    df_v.plot.hist(alpha=0.5, bins=bins, title=title_head)
    plt.xlabel(v)
    fig.tight_layout()
    fig.savefig(out_p + v + ".png")
    fig.clf()
    plt.close()
    gc.collect()

def plotPair(df_v1, df_v2, title_head, out_p):
    v1, v2 = df_v1.name, df_v2.name
    print title_head + v1 + " === " + v2
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(df_v1, df_v2, s=10, alpha=0.4)
    plt.title(title_head)
    plt.xlabel(v1)
    plt.ylabel(v2)
    fig.tight_layout()
    fig.savefig(out_p + v1 + "===" + v2 + ".png")
    fig.clf()
    plt.close()
    gc.collect()

# Plot correlation matrix of (x_i, x_j) for each vector x_i and vector x_j in matrix X
def plotCorrMatirx(df, out_p):
    # Compute correlation matrix
    df_corr = df.corr().round(3)
    df_corr.to_csv(out_p + "corr_matrix.csv")
    # Plot graph
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(df_corr, cmap=plt.get_cmap("RdBu"), interpolation="nearest",vmin=-1, vmax=1)
    fig.colorbar(im)
    fig.tight_layout()
    plt.suptitle("Correlation matrix", fontsize=18)
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_p + "corr_matrix.png", dpi=150)
    fig.clf()
    plt.close()

def plotLowDimensions(in_p, out_p, logger):
    plot_PCA = True
    plot_RTE = True
    plot_SE = True

    is_regr = False
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=3, thr=40, is_regr=is_regr,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    X = df_X.values
    Y = df_Y.squeeze().values

    if plot_PCA:
        log("Plot PCA...", logger)
        plotPCA(X, Y, out_p, is_regr=is_regr)
        log("Plot Kernel PCA...", logger)
        plotKernelPCA(X, Y, out_p, is_regr=is_regr)

    if plot_RTE:
        log("Plot Random Trees Embedding...", logger)
        plotRandomTreesEmbedding(X, Y, out_p, is_regr=is_regr)
    
    if plot_SE:
        log("Plot Spectral Embedding...", logger)
        plotSpectralEmbedding(X, Y, out_p, is_regr=is_regr)
    
    log("Finished plotting dimensions", logger)

def plotSpectralEmbedding(X, Y, out_p, is_regr=False):
    X, Y = deepcopy(X), deepcopy(Y)
    pca = PCA(n_components=10)
    X = pca.fit_transform(X)
    sm = SpectralEmbedding(n_components=4, eigen_solver="arpack", n_neighbors=10, n_jobs=-1)
    X = sm.fit_transform(X)
    title = "Spectral Embedding"
    out_p += "spectral_embedding.png"
    plotClusterPairGrid(X, Y, out_p, 3, 2, title, is_regr)

def plotRandomTreesEmbedding(X, Y, out_p, is_regr=False):
    X, Y = deepcopy(X), deepcopy(Y)
    hasher = RandomTreesEmbedding(n_estimators=1000, max_depth=5, min_samples_split=2, n_jobs=-1)
    X = hasher.fit_transform(X)
    pca = TruncatedSVD(n_components=4)
    X = pca.fit_transform(X)
    title = "Random Trees Embedding"
    out_p += "random_trees_embedding.png"
    plotClusterPairGrid(X, Y, out_p, 3, 2, title, is_regr)

def plotKernelPCA(X, Y, out_p, is_regr=False):
    X, Y = deepcopy(X), deepcopy(Y)
    pca = KernelPCA(n_components=4, kernel="rbf", n_jobs=-1)
    X = pca.fit_transform(X)
    r = pca.lambdas_
    r = np.round(r/sum(r), 3)
    title = "Kernel PCA, eigenvalue = " + str(r)
    out_p += "kernel_pca.png"
    plotClusterPairGrid(X, Y, out_p, 3, 2, title, is_regr)

def plotPCA(X, Y, out_p, is_regr=False):
    X, Y = deepcopy(X), deepcopy(Y)
    pca = PCA(n_components=4)
    X = pca.fit_transform(X)
    r = np.round(pca.explained_variance_ratio_, 3)
    title = "PCA, eigenvalue = " + str(r)
    out_p += "pca.png"
    plotClusterPairGrid(X, Y, out_p, 3, 2, title, is_regr)
