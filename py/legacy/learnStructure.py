import numpy as np
import pandas as pd
import sklearn.covariance as sklcov
import networkx as nx
from networkx.readwrite import json_graph
import json
from util import *

# Run graphical Lasso to learn the structure of the graphical model
# INPUT: transformed dataset, sample weights
# OUTPUT: graph structure, precision matrix
def learnStructure(file_path_in, file_path_out, use_sample_weight):
    print "Run graphical Lasso..."
    round_to = 6 # round the graph data to a decimal
    label_x = "NumberOfSmellReports" # the label of variable x that we want to inference

    # Check if directories exits
    for p in file_path_out:
        checkAndCreateDir(p)

    # Read the datset
    df = pd.read_csv(file_path_in[0])
    df = df[df.columns[1:]] # drop the index column
    col_names = df.columns
    print col_names
    
    # Read the sample weights
    if use_sample_weight:
        df_w = pd.read_csv(file_path_in[1])
        df_w = df_w[df_w.columns[1:]] # drop the index column

    # Compute covariance
    if use_sample_weight:
        ts_mu = computeWeightedMean(df, df_w) # note that this is a pandas time series object
        df_cov = computeWeightedCov(df, df_w, ts_mu)
    else:
        df_cov = df.cov()

    # Run Graphical Lasso
    #model = sklcov.GraphLassoCV(cv=5, max_iter=1000, alphas=20) # used for choosing alpha
    model = sklcov.GraphLasso(alpha=3.5, max_iter=2000) # for transformed dataset with sample weights
    model.fit(df_cov)
    #print model.get_params(), model.cv_alphas_, model.alpha_

    # Get the precision matrix
    prec = model.get_precision()
    prec[abs(prec) < 0.001] = 0
    prec = np.round(prec, round_to)

    # Construct graph
    prec_triu = np.triu(prec, 1) # Get the upper triangle matrix without diagonal
    rows, cols = np.nonzero(prec_triu)
    rows = rows.tolist()
    cols = cols.tolist()
    G = nx.Graph()
    print "Number of edges: " + str(len(rows))
    while len(rows) != 0:
        i = rows.pop(0)
        j = cols.pop(0)
        print "Edge: " + col_names[i] + " === " + col_names[j]
        G.add_edge(col_names[i], col_names[j], precision=round(prec[i,j],6))
    
    # Add the diagonal of the prexision matrix and the mean to the graph
    for (node, value) in zip(col_names, np.diag(prec)):
        if G.has_node(node):
            nx.set_node_attributes(G, "precision", {node: value})

    # Find the largest connected component
    #GC = max(nx.connected_component_subgraphs(G), key=len)

    # Find the connected component that contains the smell reports node
    for g in nx.connected_component_subgraphs(G):
        if g.has_node(label_x):
            GC = g
            break
    
    # Export the graph structure to json for d3.js visualization
    with open(file_path_out[0], "w") as out_file:
        json.dump(json_graph.node_link_data(GC), out_file)
    print "Graphical model created at " + file_path_out[0]

    # Export the precision matrix in the format of pandas dataframe
    df_prec = pd.DataFrame(data=prec, columns=col_names)
    df_prec.to_csv(file_path_out[1])
    print "Precision matrix created at " + file_path_out[1]

def computeWeightedMean(df, df_w):
    w = df_w.values
    ts_mu = (w*df).sum() / w.sum() # this is a pandas time series object
    return ts_mu

def computeWeightedCov(df, df_w, ts_mu):
    w = df_w.values
    mu = ts_mu.values
    df_x = df - mu
    df_cov = (w*df_x).T.dot(df_x) / (w.sum()-1)
    return df_cov
