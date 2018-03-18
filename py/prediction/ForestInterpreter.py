from util import *
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.cluster import SpectralClustering

# This class builds and interprets the ExtraTrees model
class ForestInterpreter(object):
    def __init__(self,
        df_X=None, # the predictors, a pandas dataframe
        df_Y=None, # the responses, a pandas dataframe
        logger=None):
        
        self.df_X = df_X
        self.df_Y = df_Y
        self.logger = logger

        # Fit the predictive model
        self.log("Fit predictive model..")
        self.model = RandomForestClassifier(n_estimators=10, max_features=30, min_samples_split=2, random_state=0, n_jobs=-1)
        self.model.fit(df_X, df_Y.squeeze())
        #self.reportPerformance()
        #self.reportFeatureImportance()

        # Build the decision paths
        self.dp_rules, self.dp_samples = self.extractDecisionPath()
        #for k in self.dp_rules:
        #    print "--------------------------------------------------"
        #    print k, self.dp_samples[k]
        #    print self.dp_rules[k]

        # Compute the similarity matrix
        self.sm, self.sm_cols = self.computeSimilarityMatrix()
        print self.sm[self.sm>0.1]
        print self.sm_cols

        # Cluster decision paths based on the similarity matrix
        sc = SpectralClustering(n_clusters=8, affinity="precomputed", n_init=100, random_state=0)
        sc.fit(self.sm)

    def computeSimilarityMatrix(self):
        self.log("Compute the similarity matrix of decision paths...")
        keys = self.dp_samples.keys()
        values = self.dp_samples.values()
        values = map(set, values)
        L = len(values)
        sm = np.empty([L, L])
        sm[:] = np.nan

        # Loop and build the similarity matrix
        for i in range(0, L):
            for j in range(0, L):
                # For diagonal entries, just use 0
                if i == j:
                    sm[i,j] = 0
                    continue
                # If it is already computed, just use the old value
                if not np.isnan(sm[j,i]):
                    sm[i,j] = sm[j,i]
                    continue
                # Compute weighted Jaccard Similarity of two sample sets i and j
                set_i, set_j = values[i], values[j]
                intersect_len = len(set_i & set_j)
                union_len = len(set_i | set_j)
                similarity = float(intersect_len) / float(union_len) # the normal Jaccard Similarit
                similarity *= intersect_len # weighted by the number of common samples
                sm[i,j] = similarity
                #if sm[i,j] > 10:
                #    print "-----------------------"
                #    print sm[i,j]
                #    print self.dp_rules[keys[i]]
                #    print self.dp_rules[keys[j]]

        # Scale the entire similaity matrix to range 0 and 1
        min_sm = np.min(sm)
        max_sm = np.max(sm)
        sm = (sm - min_sm) / (max_sm - min_sm)
        sm = np.round(sm, 4)

        # Return the similarity matrix and the columns (indicating path_id)
        return sm, keys
    
    def extractDecisionPath(self):
        self.log("Extract decision paths...")

        # Store all decision path rules
        # key: (tree_id, leaf_id), a tuple
        # value: decision rule of the path, a string
        dp_rules = {}

        # Store all decision path samples
        # key: (tree_id, leaf_id), a tuple
        # value: sample ids for the path, a string
        dp_samples = {}

        # Find all predictors with responese 1
        df = self.df_X[self.df_Y.squeeze() == 1].reset_index(drop=True)

        # Get all decision paths of all samples with label 1
        # Loop all trees in model, i is the sample id
        for i in range(0, len(self.model)):
            self.log("Process tree id : %s" %(i))
            tree = self.model[i]
            Y_pred = tree.predict(df)
            leave_id = tree.apply(df) # leaf node id for all samples
            value = tree.tree_.value # class label of the leaf node
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            node_indicator = tree.decision_path(df)
            # Loop all samples with label 1, j is the sample id
            for j in range(0, len(df)):
                if Y_pred[j] != 1: continue # skip if the predicted label is not one
                dp_id = (i, leave_id[j])
                # Extract decision rules
                if dp_id not in dp_rules: # onlt compute the path when it does not exist
                    node_index = node_indicator.indices[node_indicator.indptr[j]:node_indicator.indptr[j+1]]
                    rule = ""
                    for node_id in node_index:
                        if df.iloc[j, feature[node_id]] <= threshold[node_id]:
                            threshold_sign = "<="
                        else:
                            threshold_sign = ">"
                        rule += ("%20s : %s (= %s) %s %s\n" % (
                            value[node_id],
                            df.columns[feature[node_id]],
                            np.round(df.iloc[j, feature[node_id]], 3),
                            threshold_sign,
                            np.round(threshold[node_id], 3)))
                    dp_rules[dp_id] = rule
                # Update the samples
                dp_samples[dp_id] = dp_samples.get(dp_id, []) + [j]

        # return result
        return dp_rules, dp_samples

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
