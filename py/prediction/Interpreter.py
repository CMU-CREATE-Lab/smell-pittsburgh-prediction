from util import *
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import _tree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from selectFeatures import *
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift

# This class builds and interprets the model
class Interpreter(object):
    def __init__(self,
        df_X=None, # the predictors, a pandas dataframe
        df_Y=None, # the responses, a pandas dataframe
        out_p=None, # the path for saving graphs
        use_forest=True,
        logger=None):
       
        df_X, df_Y = deepcopy(df_X), deepcopy(df_Y)
        self.df_X, self.df_Y = df_X, df_Y
        self.out_p = out_p
        self.logger = logger
        
        if use_forest:
            # Fit the predictive model
            self.log("Fit predictive model..")
            F = RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_split=2, random_state=0, n_jobs=-1)
            #F = ExtraTreesClassifier(n_estimators=1000, max_features=30, min_samples_split=2, random_state=0, n_jobs=-1)
            F.fit(df_X, df_Y.squeeze())

            # Build the decision paths of samples with label 1
            # self.dp_rules contains all decision paths of positive labels
            # self.dp_samples contains all sample ids for all decision paths
            # self.df_X_pos contains samples with positive labels
            # self.df_X_pos_idx contains the index of samples with positive labels
            self.log("Extract decision paths...")
            self.dp_rules, self.dp_samples, self.df_X_pos, self.df_X_pos_idx = self.extractDecisionPath(F)

            # Compute the similarity matrix of samples having label 1
            self.log("Compute the similarity matrix of samples with label 1...")
            self.sm = self.computeSampleSimilarity()

            # Cluster samples with label 1 based on the similarity matrix
            self.log("Cluster samples with label 1...")
            self.cluster = self.clusterSamplesWithPositiveLabels()
        else:
            df_X_pos = df_X[df_Y["smell"]==1]
            df_X_pos_idx = df_X_pos.index.values
            df_X_pos = df_X_pos.reset_index(drop=True)
            X = KernelPCA(n_components=10, kernel="rbf", n_jobs=-1).fit_transform(df_X_pos)
            cluster = MeanShift(n_jobs=-1, bandwidth=0.25).fit_predict(X)
            cluster[cluster>0] = -1
            self.cluster, self.df_X_pos, self.df_X_pos_idx = cluster, df_X_pos, df_X_pos_idx
            print "Unique cluster ids : %s" % np.unique(cluster)
            print "Total number of positive samples : %s" % len(self.df_X_pos)
            for c_id in np.unique(cluster):
                if c_id < 0:
                    print "%s samples are not clustered" % (len(cluster[cluster==c_id]))
                else:
                    print "Cluster %s has %s samples" % (c_id, len(cluster[cluster==c_id]))
            if self.out_p is not None:
                self.plotClusters(cluster, self.out_p)
            qc = silhouette_score(X, cluster)
            print "Silhouette Coefficient: %0.3f" % qc

        # Set the label for samples outside the cluster to zero
        df_X_pos_idx_c0 = self.df_X_pos_idx[self.cluster==0] # select the largest cluster
        df_X_neg_idx_c0 = ~self.df_X.index.isin(df_X_pos_idx_c0) # select samples that are not in the cluster
        self.df_Y["smell"].iloc[df_X_neg_idx_c0] = 0 # select labels of samples that are not in the cluster to zero

        # Feature selection
        self.df_X, self.df_Y = selectFeatures(df_X=self.df_X, df_Y=self.df_Y,
            method="RFE", is_regr=False, num_rfe_feat=30, num_rfe_loop=10)
        
        # Train a L1 logistic regression on the selected cluster
        print "Train a logistic regression model..."
        lr = LogisticRegression(random_state=0, penalty="l1", C=0.01)
        lr.fit(self.df_X, self.df_Y.squeeze())
        self.reportPerformance(lr)
        self.reportCoefficient(lr)

        # Train a decision tree classifier on the selected cluster
        print "Train a decision tree..."
        dt = DecisionTreeClassifier(random_state=0, min_samples_split=20, max_depth=6)
        dt.fit(self.df_X, self.df_Y.squeeze())
        self.reportPerformance(dt)
        self.reportFeatureImportance(dt)
        self.exportTreeGraph(dt)

    def reportCoefficient(self, model):
        for (c, fn) in zip(np.squeeze(model.coef_), self.df_X.columns.values):
            if c > 0.00001: self.log("{0:.5f}".format(c) + " -- " + str(fn))

    def getFilteredLabels(self):
        return self.df_Y

    def getSelectedFeatures(self):
        return self.df_X

    def exportTreeGraph(self, tree):
        # For http://webgraphviz.com/
        with open(self.out_p + "decision_tree.dot", "w") as f:
            export_graphviz(tree,
                out_file=f,
                feature_names=self.df_X.columns,
                class_names=["no", "yes"],
                max_depth=5,
                filled=True)

    def clusterSamplesWithPositiveLabels(self):
        c = DBSCAN(metric="precomputed", min_samples=30, eps=0.75, n_jobs=-1) # for Random Forest
        #c = DBSCAN(metric="precomputed", min_samples=30, eps=0.9, n_jobs=-1) # for ExtraTrees
        dist = 1.0 - self.sm # DBSCAN uses distance instead of similarity
        cluster = c.fit_predict(dist)

        # Print cluster information
        print "Unique cluster ids : %s" % np.unique(cluster)
        print "Total number of positive samples : %s" % len(self.df_X_pos)
        for c_id in np.unique(cluster):
            if c_id < 0:
                print "%s samples are not clustered" % (len(cluster[cluster==c_id]))
            else:
                print "Cluster %s has %s samples" % (c_id, len(cluster[cluster==c_id]))
        if self.out_p is not None:
            self.plotClusters(cluster, self.out_p)
        
        # Merge clusters
        print "Merge clusters..."
        cluster[cluster>0] = 0

        # Evaluate the quality of the cluster
        qc = silhouette_score(dist, cluster, metric="precomputed")
        print "Silhouette Coefficient: %0.3f" % qc
        return cluster

    def plotClusters(self, cluster, out_p):
        # Visualize the clusters using PCA
        print "Plot PCA of positive labels..."
        pca = PCA(n_components=4)
        X = pca.fit_transform(deepcopy(self.df_X_pos.values))
        r = np.round(pca.explained_variance_ratio_, 3)
        title = "PCA of positive labels, eigenvalue = " + str(r)
        out_p_tmp = out_p + "pca_positive_labels.png"
        c_ls = [(0.5, 0.5, 0.5), (1, 0, 0), (0, 0, 1)]
        c_alpha = [0.1, 0.2, 0.1]
        c_bin=[0]
        plotClusterPairGrid(X, cluster, out_p_tmp, 3, 2, title, False, c_ls=c_ls, c_alpha=c_alpha, c_bin=c_bin)

        # Visualize the clusters using kernel PCA
        print "Plot Kernel PCA of positive labels..."
        pca = KernelPCA(n_components=4, kernel="rbf", n_jobs=-1)
        X = pca.fit_transform(deepcopy(self.df_X_pos.values))
        r = pca.lambdas_
        r = np.round(r/sum(r), 3)
        title = "Kernel PCA of positive labels, eigenvalue = " + str(r)
        out_p_tmp = out_p + "kernel_pca_positive_labels.png"
        plotClusterPairGrid(X, cluster, out_p_tmp, 3, 2, title, False, c_ls=c_ls, c_alpha=c_alpha, c_bin=c_bin)

    def computeSampleSimilarity(self):
        L = len(self.df_X_pos)
        sm = np.zeros([L, L])

        # Loop and build the similarity matrix for samples with label 1
        for k in self.dp_samples:
            s = self.dp_samples[k]
            for i in range(0, len(s)):
                for j in range(i+1, len(s)):
                    sm[s[i], s[j]] += 1
                    sm[s[j], s[i]] += 1
        
        return self.scaleMatrix(sm) 

    def computeDecisionPathSimilarity(self):
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
                if sm[i,j] > 15:
                    print "-----------------------"
                    print sm[i,j]
                    print keys[i]
                    print self.dp_rules[keys[i]]
                    print keys[j]
                    print self.dp_rules[keys[j]]

        # Return the similarity matrix and the columns (indicating path_id)
        return self.scaleMatrix(sm), keys

    def scaleMatrix(self, m):
        # Scale the entire matrix to range 0 and 1
        min_m = np.min(m)
        max_m = np.max(m)
        m = (m - min_m) / (max_m - min_m)
        m = np.round(m, 4)
        return m

    def extractDecisionPath(self, model):
        # Store all decision path rules
        # key: (tree_id, leaf_id), a tuple
        # value: decision rule of the path, a string
        dp_rules = {}

        # Store all decision path samples
        # key: (tree_id, leaf_id), a tuple
        # value: sample ids for the path, a string
        dp_samples = {}

        # Find all predictors with responese 1
        df = self.df_X[self.df_Y["smell"] == 1]
        df_idx = df.index
        df = df.reset_index(drop=True)

        # Get all decision paths of all samples with label 1
        # Loop all trees in model, i is the sample id
        for i in range(0, len(model)):
            if i % 5 == 0: self.log("Process tree id : %s" %(i))
            tree = model[i]
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
        return dp_rules, dp_samples, df, df_idx.values

    def reportPerformance(self, model):
        print "Report performance for all data..."
        metric = computeMetric(self.df_Y, model.predict(self.df_X), False)
        for m in metric: self.log(metric[m])

        print "Report performance for daytime data..."
        hd_start, hd_end = 8, 18
        hd = self.df_X["HourOfDay"]
        dt_idx = (hd>=hd_start)&(hd<=hd_end)
        metric = computeMetric(self.df_Y[dt_idx], model.predict(self.df_X[dt_idx]), False)
        for m in metric: self.log(metric[m])

    def reportFeatureImportance(self, model, thr=0.9):
        feat_ims = np.array(model.feature_importances_)
        sorted_ims_idx = np.argsort(feat_ims)[::-1]
        feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
        feat_names = self.df_X.columns.copy()
        feat_names = feat_names[sorted_ims_idx]
        c = 0
        for (fi, fn) in zip(feat_ims, feat_names):
            self.log("{0:.5f}".format(fi) + " -- " + str(fn))
            c += fi
            if c > thr: break

    def tree_to_code(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print "def tree({}):".format(", ".join(feature_names))

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
