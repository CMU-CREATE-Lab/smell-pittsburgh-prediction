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
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from computeFeatures import *
from ForestInterpreter import *
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

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

    # Plot the distribution of smell reports by days of week and hours of day
    #plotDayHour(in_p, out_p, logger)

    # Plot dimension reduction
    plotLowDimensions(in_p, out_p, logger)

    # Correlational study
    #corrStudy(in_p, out_p, logger=logger)

    # Interpret model
    #interpretModel(in_p, out_p, logger=logger)

# Correlational study
def corrStudy(in_p, out_p, logger):
    log("Compute correlation of lagged X and current Y...", logger)
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=0, thr=40, is_regr=True,
         add_inter=False, add_roll=False, add_diff=False, logger=logger)
    Y = df_Y.squeeze()
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

# Interpret the model
def interpretModel(in_p, out_p, logger):
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=3, thr=40, is_regr=False,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
    model = ForestInterpreter(df_X=df_X, df_Y=df_Y, logger=logger)

def plotLowDimensions(in_p, out_p, logger):
    plot_PCA = False
    plot_manifold = True
    is_regr = False
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=8, b_hr=3, thr=40, is_regr=is_regr,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)

    if plot_PCA:
        log("Plot PCA...", logger)
        plotPCA(df_X, df_Y, out_p, is_regr=is_regr, use_kernel=False)
        log("Plot Kernel PCA...", logger)
        plotPCA(df_X, df_Y, out_p, is_regr=is_regr, use_kernel=True)

    if plot_manifold:
        log("Plot manifold...", logger)
        plotManifold(df_X, df_Y, out_p, is_regr=is_regr)
    
    log("Finished", logger)

def plotManifold(df_X, df_Y, out_p, is_regr=False):
    # Use PCA to reduce dimensions first (speed up manifold learning and reduce noise)
    pca = PCA(n_components=10)
    X = pca.fit_transform(df_X)

    # Apply manifold learning
    n_c = 3
    #model = Isomap(n_neighbors=10, n_components=n_c, max_iter=500, n_jobs=-1)
    model = TSNE(n_components=n_c, perplexity=50.0, init="pca", n_iter=500, verbose=1)
    X = model.fit_transform(X)
    Y = df_Y.squeeze().values
     
    dot_cmap = "RdBu"
    dot_size = 20
    w = 3
    h = 1
    c = 1
    if not is_regr:
        c0 = (Y == 0)
        c1 = (Y == 1)
    fig = plt.figure(figsize=(18, 6), dpi=150)
    for i in range(0, n_c-1):
        for j in range(i+1, n_c):
            plt.subplot(h, w, c)
            if is_regr:
                plt.scatter(X[:,i], X[:,j], c=Y, s=dot_size, alpha=0.2, cmap=dot_cmap)
            else:
                plt.scatter(X[c0,i], X[c0,j], c="b", s=dot_size, alpha=0.1, cmap=dot_cmap)
                plt.scatter(X[c1,i], X[c1,j], c="r", s=dot_size, alpha=0.2, cmap=dot_cmap)
            plt.xlabel("Component " + str(i))
            plt.ylabel("Component " + str(j))
            c += 1
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.suptitle("Dimension reduction using manifold learning", fontsize=22)
    fig.savefig(out_p + "manifold.png")
    fig.clf()
    plt.close()

def plotDayHour(in_p, out_p, logger):
    df_X, df_Y, _ = computeFeatures(in_p=in_p, f_hr=None, b_hr=0, thr=40, is_regr=True,
        add_inter=False, add_roll=False, add_diff=False, logger=logger)
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

def plotPCA(df_X, df_Y, out_p, is_regr=False, use_kernel=False):
    n_c = 5
    if use_kernel:
        pca = KernelPCA(n_components=n_c, kernel="rbf")
    else:
        pca = PCA(n_components=n_c)
    X = pca.fit_transform(df_X)
    Y = df_Y.squeeze()
    if use_kernel:
        r = pca.lambdas_
        r = np.round(r/sum(r), 3)
    else:
        r = np.round(pca.explained_variance_ratio_, 3)

    dot_size = 15
    dot_cmap = "brg"
    w = 5
    h = 2
    c = 1
    if not is_regr:
        c0 = (Y == 0)
        c1 = (Y == 1)
    fig = plt.figure(figsize=(30, 12), dpi=150)
    for i in range(0, n_c-1):
        for j in range(i+1, n_c):
            plt.subplot(h, w, c)
            if is_regr:
                plt.scatter(X[:,i], X[:,j], c=Y, s=dot_size, alpha=0.2, cmap=dot_cmap)
            else:
                plt.scatter(X[c0,i], X[c0,j], c="b", s=dot_size, alpha=0.1, cmap=dot_cmap)
                plt.scatter(X[c1,i], X[c1,j], c="r", s=dot_size, alpha=0.2, cmap=dot_cmap)
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

def plotHist(df_v, title_head, out_p):
    v = df_v.name
    print title_head + v
    fig = plt.figure(figsize=(8, 8), dpi=150)
    df_v.plot.hist(alpha=0.5, bins=30, title=title_head)
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
