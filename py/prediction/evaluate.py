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

def readInfo2(p):
    with open(p) as f:
        data = f.readlines()
    info = {}
    for i in range(len(data)):
        if "Train a decision tree" in data[i]:
            a = data[i+11].split("\n")[0] # training error of the tree
            b = data[i+19].split("\n")[0] # most important feature
            c = data[i+20].split("\n")[0] # second important feature
            info["train_all"] = json.loads(a.replace("\'","\""))
            b = map(str.strip, b.split("--"))
            c = map(str.strip, c.split("--"))
            info["train_all"]["f1"] = b[1]
            info["train_all"]["f1_importance"] = round(float(b[0]), 3)
            info["train_all"]["f2"] = c[1]
            info["train_all"]["f2_importance"] = round(float(c[0]), 3)
        elif "(Daytime only) For all training data with method DT" in data[i]:
            # Training performance for cross-validation
            info["train_cv"] = json.loads(data[i+2].split("\n")[0].replace("\'","\""))
        elif "(Daytime only) For all testing data with method DT" in data[i]:
            # Testing performance for cross-validation
            info["test_cv"] = json.loads(data[i+2].split("\n")[0].replace("\'","\""))
    return info

def main2():
    path = "data_main/analysis/experiment/"
    dt = []
    train_cv = []
    test_cv = []
    for name in listdir(path):
        try:
            info = readInfo2(path + name + "/DT-" + name + ".log")
            dt.append(info["train_all"])
            train_cv.append(info["train_cv"])
            test_cv.append(info["test_cv"])
            corr = pd.read_csv(path + name + "/corr_inference.csv")
            print "-"*10
            print "Most important feature:"
            print info["train_all"]["f1"]
            print corr[info["train_all"]["f1"]]
            print "Second important feature:"
            print info["train_all"]["f2"]
            print corr[info["train_all"]["f2"]]
            print "-"*10
        except Exception as e:
            print(e)
            continue
    df_dt = pd.DataFrame(data=dt) # Decision Tree
    df_train_cv = pd.DataFrame(data=train_cv) # Training performance for cross-validation
    df_test_cv = pd.DataFrame(data=test_cv) # Testing performance for cross-validation
    print df_dt
    print "----------------"
    print "Decision Tree"
    print df_dt.describe()
    print "----------------"
    print "Unique for the most important feature"
    print df_dt.groupby("f1").count()["FN"]
    print "----------------"
    print "Unique for the second important feature"
    print df_dt.groupby("f2").count()["FN"]
    print "----------------"
    print "Training performance for cross-validation"
    print df_train_cv.describe()
    print "----------------"
    print "Testing performance for cross-validation"
    print df_test_cv.describe()

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

main2()
