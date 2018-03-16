from util import *
import numpy as np
import copy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree

# This class builds and interprets the ExtraTrees model
class ForestInterpreter(object):
    def __init__(self,
        df_X=None, # the predictors, a pandas dataframe
        df_Y=None, # the responses, a pandas dataframe
        logger=None):
        
        df_Y = df_Y.squeeze()
        self.df_X = df_X
        self.df_Y = df_Y
        self.path = {} # decision paths indexed by tree_id and path_id
        self.logger = logger

        # Fit the ExtraTrees model
        self.log("Fit ExtraTrees model..")
        #model = ExtraTreesClassifier(n_estimators=1000, max_features=20, min_samples_split=2, random_state=0, n_jobs=-1)
        model = RandomForestClassifier(n_estimators=100, max_features=20, min_samples_split=2, random_state=0, n_jobs=-1)
        #model = ExtraTreesClassifier(n_estimators=100, max_features=20, min_samples_split=2, random_state=0, n_jobs=-1)
        model.fit(df_X, df_Y)
        self.model = model

        # Find all predictors with responese 1
        df_X = df_X[df_Y == 1].reset_index(drop=True)

        # Get all decision paths of df_X (labels are already 1)
        sample_id = 0
        for i in range(0, len(model)):
            tree = model[i]
            Y_pred = tree.predict(df_X)
            leave_id = tree.apply(df_X)
            leaf_id = leave_id[sample_id]
            print "-------------------------------"
            print "Tree id : " + str(i)
            if Y_pred[sample_id] != 1:
                print "The predicted result is not 1"
                continue
            else:
                print "Leaf id : " + str(leaf_id)
            value = tree.tree_.value # class value of the leaf node
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            node_indicator = tree.decision_path(df_X)
            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
            for node_id in node_index:
                if df_X.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                #print("decision id node %s : %s (= %s) %s %s"
                #print("%20s : %s (= %s) %s %s" % (
                #    value[node_id],
                #    df_X.columns[feature[node_id]],
                #    np.round(df_X.iloc[sample_id, feature[node_id]], 3),
                #    threshold_sign,
                #    np.round(threshold[node_id], 3)))
            #break

    def printDecisionPath(self, tree_id, leaf_id):
        # need to print the decision path based on tree_id and leaf_id
        return

    def reportPerformance(self):
        self.log("Report training performance...")
        metric = computeMetric(self.df_Y, self.model.predict(self.df_X), False)
        for m in metric:
            self.log(metric[m])

    def reportFeatureImportance(self):
        self.log("Report 50% feature importance...") 
        feat_ims = np.array(self.model.feature_importances_)
        sorted_ims_idx = np.argsort(feat_ims)[::-1]
        feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
        feat_names = self.df_X.columns.copy()
        feat_names = feat_names[sorted_ims_idx]
        c = 0
        for (fi, fn) in zip(feat_ims, feat_names):
            self.log("{0:.5f}".format(fi) + " -- " + str(fn))
            c += fi
            if c > 0.5: break

    def tree_to_code(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        print "def tree({}):".format(", ".join(feature_names))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print "{}if {} <= {}:".format(indent, name, threshold)
                recurse(tree_.children_left[node], depth + 1)
                print "{}else:  # if {} > {}".format(indent, name, threshold)
                recurse(tree_.children_right[node], depth + 1)
            else:
                print "{}return {}".format(indent, tree_.value[node])

        recurse(0, 1)

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
