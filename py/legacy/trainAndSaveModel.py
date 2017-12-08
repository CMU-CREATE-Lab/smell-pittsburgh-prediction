from util import *
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import tree
from sklearn.svm import SVC

def trainAndSaveModel(file_path_in, file_path_out, is_regression):
    print "====================================================================="
    print "Train and save model"
    print "--------------------------------------------------------------"
    
    is_tree = True

    # Check directories
    for p in file_path_out:
        checkAndCreateDir(p)

    # Read features
    df = pd.read_csv(file_path_in[0])

    # The label of response
    label_resp = "smell_value"

    # Build model
    df_Y = df[label_resp].copy()
    df_X = df[[c for c in df.columns if c != label_resp]].copy()
    if is_regression:
        model = ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    else:
        model = SVC(max_iter=5000)
    model.fit(df_X,df_Y)

    # Export tree graph
    # for http://webgraphviz.com/
    if False:
        i = 0
        for t in model.estimators_:
            with open(file_path_out[0] + "tree_" + str(i) + ".dot", "w") as f:
                tree.export_graphviz(t,
                        out_file=f,
                        feature_names=df_X.columns,
                        class_names=["no", "yes"],
                        max_depth=5,
                        filled=True)
            i += 1
