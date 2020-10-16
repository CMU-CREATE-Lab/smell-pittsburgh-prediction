from util import plotClusterPairGrid, computeMetric
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#from sklearn.linear_model import LogisticRegression
from selectFeatures import selectFeatures
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift
from scipy.stats import pointbiserialr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
This class builds and interprets the model
"""
class Interpreter(object):
    def __init__(self,
        df_X=None, # the predictors, a pandas dataframe
        df_Y=None, # the responses, a pandas dataframe
        out_p=None, # the path for saving graphs
        use_forest=True,
        logger=None):

        df_X = deepcopy(df_X)
        df_Y = deepcopy(df_Y)

        # We need to run the random forest in the unsupervised mode
        # Consider the original data as class 1
        # Create a synthetic second class of the same size that will be labeled as class 2
        # The synthetic class is created by sampling at random from the univariate distributions of the original data
        n = len(df_X)
        synthetic = {}
        for c in df_X.columns:
            synthetic[c] = df_X[c].sample(n=n, replace=True).values
        df_synthetic = pd.DataFrame(data=synthetic)
        df_XX = pd.concat([df_X, df_synthetic])
        df_YY = pd.concat([df_Y.applymap(lambda x: 1), df_Y.applymap(lambda x: -1)])

        self.df_XX, self.df_YY = df_XX, df_YY
        self.df_X, self.df_Y = df_X, df_Y
        self.out_p = out_p
        self.logger = logger

        if use_forest:
            # Fit the predictive model
            self.log("Fit predictive model..")
            n_trees = 1000
            F = RandomForestClassifier(n_estimators=n_trees, max_features=0.15, n_jobs=-1, min_samples_split=4)
            F.fit(df_XX, df_YY.squeeze())
            self.reportPerformance(F, df_XX, df_YY)
            self.reportFeatureImportance(F, df_XX, thr=0.3)

            # Build the decision paths of samples with label 1
            # self.dp_rules contains all decision paths of positive labels
            # self.dp_samples contains all sample ids for all decision paths
            # self.df_X_pos contains samples with positive labels
            # self.df_X_pos_idx contains the index of samples with positive labels
            self.log("Extract decision paths...")
            self.dp_rules, self.dp_samples, self.df_X_pos, self.df_X_pos_idx = self.extractDecisionPath(F)

            # Compute the similarity matrix of samples having label 1
            self.log("Compute the similarity matrix of samples with label 1...")
            self.sm = self.computeSampleSimilarity(n_trees)

            # Cluster samples with label 1 based on the similarity matrix
            self.log("Cluster samples with label 1...")
            self.cluster = self.clusterSamplesWithPositiveLabels()

            # TODO: Refine the cluster by using k-means on the original feature space
        else:
            df_X_pos = df_X[df_Y["smell"]==1]
            df_X_pos_idx = df_X_pos.index.values
            df_X_pos = df_X_pos.reset_index(drop=True)
            X = KernelPCA(n_components=10, kernel="rbf", n_jobs=-1).fit_transform(df_X_pos)
            cluster = MeanShift(n_jobs=-1, bandwidth=0.25).fit_predict(X)
            cluster[cluster>0] = -1
            self.cluster, self.df_X_pos, self.df_X_pos_idx = cluster, df_X_pos, df_X_pos_idx
            self.log("Unique cluster ids : %s" % np.unique(cluster))
            self.log("Total number of positive samples : %s" % len(self.df_X_pos))
            for c_id in np.unique(cluster):
                if c_id < 0:
                    self.log("%s samples are not clustered" % (len(cluster[cluster==c_id])))
                else:
                    self.log("Cluster %s has %s samples" % (c_id, len(cluster[cluster==c_id])))
            qc = silhouette_score(X, cluster)
            self.log("Silhouette Coefficient: %0.3f" % qc)

        # Set the label for samples outside the cluster to zero
        df_X_pos_idx_c0 = self.df_X_pos_idx[self.cluster==0] # select the largest cluster
        df_X_neg_idx_c0 = ~self.df_X.index.isin(df_X_pos_idx_c0) # select samples that are not in the cluster
        self.df_Y["smell"].astype(int).iloc[df_X_neg_idx_c0] = -1 # set labels of samples that are not in the cluster to -1
        #if self.out_p is not None:
        #    self.plotClusters(self.df_X, self.df_Y, self.out_p)
        self.df_Y["smell"].iloc[df_X_neg_idx_c0] = 0 # set labels of samples that are not in the cluster to zero

        # Feature selection
        self.df_X, self.df_Y = selectFeatures(df_X=self.df_X, df_Y=self.df_Y,
            method="RFE", is_regr=False, num_feat_rfe=30, step_rfe=50)

        # Plot point biserial correlation
        df_corr_info = pd.DataFrame()
        df_corr = pd.DataFrame()
        Y = self.df_Y.squeeze().values
        n = len(self.df_X)
        for c in self.df_X.columns:
            if c in ["Day", "DayOfWeek", "HourOfDay"]: continue
            r, p = pointbiserialr(Y, self.df_X[c])
            df_corr_info[c] = pd.Series(data=(np.round(r,3), np.round(p,5), n))
            df_corr[c] = pd.Series(data=np.round(r,3))
        # The first row is the correlation
        # The second row is the p-value
        # The third row is the numbe of samples
        df_corr_info.to_csv(out_p+"corr_inference.csv")
        #self.plotCorrelation(df_corr, out_p+"corr_inference.png")

        # Format feature names
        #self.df_X.columns = [c.replace("*", "\n*") for c in self.df_X.columns]

        # Train a L1 logistic regression on the selected cluster
        #self.log("Train a logistic regression model...")
        #lr = LogisticRegression(penalty="l1", C=1)
        #lr.fit(self.df_X, self.df_Y.squeeze())
        #self.reportPerformance(lr)
        #self.reportCoefficient(lr)

        # Train a decision tree classifier on the selected cluster
        self.log("Train a decision tree...")
        dt = DecisionTreeClassifier(min_samples_split=10, max_depth=8, min_samples_leaf=5)
        dt.fit(self.df_X, self.df_Y.squeeze())
        self.reportPerformance(dt, self.df_X, self.df_Y)
        self.reportFeatureImportance(dt, self.df_X)
        dt = self.selectDecisionTreePaths(dt)
        self.exportTreeGraph(dt)

    def plotCorrelation(self, df_corr, out_p):
        # Plot graph
        tick_font_size = 16
        label_font_size = 20
        title_font_size = 32

        fig, ax1 = plt.subplots(1, 1, figsize=(28, 5))
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("right", size="2%", pad=0.4)
        sns.heatmap(df_corr, ax=ax1, cbar_ax=ax2, cmap="RdBu", vmin=-0.6, vmax=0.6,
            linewidths=0.1, annot=True, fmt="g", xticklabels=False, yticklabels="auto")

        ax1.tick_params(labelsize=tick_font_size)
        ax2.tick_params(labelsize=tick_font_size)
        ax1.set_ylabel("", fontsize=label_font_size)
        ax1.set_xlabel("Selected predictors", fontsize=label_font_size)
        plt.suptitle("Biserial correlation of predictors and response (smell events)", fontsize=title_font_size)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(out_p, dpi=150)
        fig.clf()
        plt.close()

    # Only select paths that lead to positive labels
    def selectDecisionTreePaths(self, dt):
        tree = dt.tree_
        self.dt_children_left = tree.children_left
        self.dt_children_right = tree.children_right
        #feature = tree.feature
        #threshold = tree.threshold
        self.dt_value = tree.value
        self.dt_want_pruned = np.zeros(shape=tree.node_count, dtype=bool)

        # Perform post-order DFS to mark nodes that need to be pruned
        self.log("Pruning nodes...")
        self.postorderTraversal(0, 0)
        return dt

    # i is the current node id
    # p is the parent node id
    def postorderTraversal(self, i, depth):
        if self.dt_children_left[i] != -1:
            self.postorderTraversal(self.dt_children_left[i], depth+1)
        if self.dt_children_right[i] != -1:
            self.postorderTraversal(self.dt_children_right[i], depth+1)
        # Check if this node want to be pruned (num_neg>num_pos or num_neg+num_pos<thr)
        # num_neg>num_pos means number of negative labels larger than positive labels
        # num_neg+num_pos<thr means sample size too small
        left, right = self.dt_children_left[i], self.dt_children_right[i]
        v = self.dt_value[i][0]
        thr = 40
        if v[0] > v[1] or v[0] + v[1] < thr:
            if left != -1 and right != -1: # not a leaf node
                # Only want to be pruned if all childs want to be pruned
                if self.dt_want_pruned[left] and self.dt_want_pruned[right]:
                    self.dt_want_pruned[i] = True
                    self.dt_children_left[i] = -1
                    self.dt_children_right[i] = -1
            else: # a leaf node
                self.dt_want_pruned[i] = True

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
                class_names=None,
                max_depth=4,
                precision=2,
                impurity=False,
                rounded=True,
                filled=True)

    def clusterSamplesWithPositiveLabels(self):
        c = DBSCAN(metric="precomputed", min_samples=30, eps=0.965, n_jobs=-1) # for Random Forest
        dist = 1.0 - self.sm # DBSCAN uses distance instead of similarity
        cluster = c.fit_predict(dist)

        # Clean clusters
        self.log("Select only the largest cluster")
        cluster[cluster>0] = -1

        # Print cluster information
        self.log("Unique cluster ids : %s" % np.unique(cluster))
        self.log("Total number of positive samples : %s" % len(self.df_X_pos))
        for c_id in np.unique(cluster):
            if c_id < 0:
                self.log("%s samples are not clustered" % (len(cluster[cluster==c_id])))
            else:
                self.log("Cluster %s has %s samples" % (c_id, len(cluster[cluster==c_id])))

        # Evaluate the quality of the cluster
        if len(np.unique(cluster)) > 1:
            qc1 = silhouette_score(dist, cluster, metric="precomputed") # on the distance space
            self.log("Silhouette coefficient on the distance space: %0.3f" % qc1)
            qc2 = silhouette_score(self.df_X_pos, cluster) # on the original space
            self.log("Silhouette coefficient on the original space: %0.3f" % qc2)
        return cluster

    def plotClusters(self, df_X, df_Y, out_p):
        # Visualize the clusters using PCA
        self.log("Plot PCA of positive labels...")
        pca = PCA(n_components=3)
        X = pca.fit_transform(deepcopy(df_X.values))
        r = np.round(pca.explained_variance_ratio_, 3)
        title = "PCA of positive labels, eigenvalue = " + str(r)
        out_p_tmp = out_p + "pca_positive_labels.png"
        c_ls = [(0.5, 0.5, 0.5), (0, 0, 1), (1, 0, 0)]
        c_alpha = [0.1, 0.2, 0.1]
        c_bin=[0]
        plotClusterPairGrid(X, df_Y, out_p_tmp, 3, 1, title, False, c_ls=c_ls, c_alpha=c_alpha, c_bin=c_bin)

        # Visualize the clusters using kernel PCA
        self.log("Plot Kernel PCA of positive labels...")
        pca = KernelPCA(n_components=3, kernel="rbf", n_jobs=-1)
        X = pca.fit_transform(deepcopy(df_X.values))
        r = pca.lambdas_
        r = np.round(r/sum(r), 3)
        title = "Kernel PCA of positive labels, eigenvalue = " + str(r)
        out_p_tmp = out_p + "kernel_pca_positive_labels.png"
        plotClusterPairGrid(X, df_Y, out_p_tmp, 3, 1, title, False, c_ls=c_ls, c_alpha=c_alpha, c_bin=c_bin)

    def computeSampleSimilarity(self, n_trees):
        L = len(self.df_X_pos)
        sm = np.zeros([L, L])

        # Loop and build the similarity matrix for samples with label 1
        for k in self.dp_samples:
            s = self.dp_samples[k]
            for i in range(0, len(s)):
                for j in range(i+1, len(s)):
                    sm[s[i], s[j]] += 1
                    sm[s[j], s[i]] += 1

        return sm / n_trees

    def extractDecisionPath(self, model, extract_rule=False):
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
            if extract_rule:
                value = tree.tree_.value # class label of the leaf node
                feature = tree.tree_.feature
                threshold = tree.tree_.threshold
                node_indicator = tree.decision_path(df)
            # Loop all samples with label 1, j is the sample id
            for j in range(0, len(df)):
                if Y_pred[j] != 1: continue # skip if the predicted label is not one
                dp_id = (i, leave_id[j])
                # Extract decision rules
                if extract_rule and dp_id not in dp_rules: # onlt compute the path when it does not exist
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

    def reportPerformance(self, model, df_X, df_Y):
        self.log("Report performance for all data...")
        metric = computeMetric(df_Y, model.predict(df_X), False)
        for m in metric: self.log(metric[m])

        self.log("Report performance for daytime data...")
        hd_start, hd_end = 8, 18
        hd = self.df_X["HourOfDay"]
        dt_idx = (hd>=hd_start)&(hd<=hd_end)
        metric = computeMetric(df_Y[dt_idx], model.predict(df_X[dt_idx]), False)
        for m in metric: self.log(metric[m])

    def reportFeatureImportance(self, model, df_X, thr=0.9):
        feat_ims = np.array(model.feature_importances_)
        sorted_ims_idx = np.argsort(feat_ims)[::-1]
        feat_ims = np.round(feat_ims[sorted_ims_idx], 5)
        feat_names = df_X.columns.copy()
        feat_names = feat_names[sorted_ims_idx]
        c = 0
        for (fi, fn) in zip(feat_ims, feat_names):
            self.log("{0:.5f}".format(fi) + " -- " + str(fn))
            c += fi
            if c > thr: break

    def log(self, msg):
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)
