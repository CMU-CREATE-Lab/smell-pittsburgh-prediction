from util import *
import pandas as pd
import json

def readInfo(p):
    with open(p) as f:
        data = f.readlines()
    if "time series" in data[-1]:
        tail = data[-13]
    if "residual" in data[-1]:
        tail = data[-7]
    else:
        tail = data[-12]
    tail = tail.strip().replace("'", "\"")
    return pd.read_json("[" + tail + "]", orient="records")

def main2():
    path = "data_main/analysis/log/result/"
    cols = ["TP","FP","FN","precision","recall","fscore"]
    dt = []
    for name in getAllFileNamesInFolder(path):
        if "DT" in name:
            dt.append(readInfo(path + name))
    df_dt = pd.concat(dt) # Decision Tree
    print df_dt
    print "----------------"
    print "Decision Tree"
    print df_dt.describe()

def evaluate(path, rule):
    cols = ["TP","FP","FN","precision","recall","fscore"]
    d = []
    for name in getAllFileNamesInFolder(path):
        if rule in name:
            d.append(readInfo(path + name))
    df = pd.concat(d) # Decision Tree
    print df
    print df.describe()

def main():
    print "------------------------------------------------------"
    print "------------------------------------------------------"
    print "Classification ExtraTrees"
    evaluate("data_main/log/classification/result/", "ET")
    print "------------------------------------------------------"
    print "------------------------------------------------------"
    print "Classification Random Forest"
    evaluate("data_main/log/classification/result/", "RF")
    print "------------------------------------------------------"
    print "------------------------------------------------------"
    print "Regression ExtraTrees"
    evaluate("data_main/log/regression/result/", "ET")
    print "------------------------------------------------------"
    print "------------------------------------------------------"
    print "Regresssion Random Forest"
    evaluate("data_main/log/regression/result/", "RF")
    print "------------------------------------------------------"
    print "------------------------------------------------------"
    print "Decision Tree"
    evaluate("data_main/analysis/log/result/", "DT")

main()
