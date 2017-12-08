import numpy as np
import copy
import networkx as nx

# Inference p(x) by using Gaussian loopy belief propagation
# INPUT:
# - df_train: the training set
#   (type: pandas data frame)
# - df_test: the testing set
#   (type: same as df_train)
# - label_x: the label of variable x that we want to inference
#   (type: a string of the column name of the pandas dataframe)
# - G: the graphical model
#   (type: a networkx undirected graph)
# OUTPUT:
# - x_pred_all: the predicted value of x in the testing set
#   (type: an array of numbers)
# - x_true_all: the true value of x in the testing set
#   (type: same as x_pred)
def GaLBP_main(df_train, df_test, label_x, G):
    debug = False

    # Compute mean
    mu_node = df_train.mean()

    # Preprocessing for speeding up GaLBP
    mu_xx = mu_node[label_x] # empirical mean of variable x that we want to inference
    A_node = nx.get_node_attributes(G, "precision") # parameter of node potentials (precision)
    A_node = copy.deepcopy(A_node)
    A_edge_tmp = nx.get_edge_attributes(G, "precision") # parameter edge potentials (precision)
    A_edge = {} # parameter edge potentials (precision)
    for (u, v) in A_edge_tmp:
        val = A_edge_tmp[u, v]
        A_edge[v, u] = val
        A_edge[u, v] = val

    # Initialize messages passing along each edge at both directions
    P_msg = {} # edge potentials (precision)
    mu_msg = {} # edge potentials (mean)
    for e in G.edges():
        P_msg[e[0], e[1]] = 0
        P_msg[e[1], e[0]] = 0
        mu_msg[e[0], e[1]] = 0
        mu_msg[e[1], e[0]] = 0

    # Inference p(x)
    x_true_all = []
    x_pred_all = []
    c = 0
    for idx, row in df_test.iterrows():
        c += 1
        x_true = np.round(row[label_x], 6) # true value of x
        x_true_all.append(x_true)
        y_node = row.drop(label_x).to_dict() # parameter of node potentials (values of given variables y)
        x_pred = GaLBP(G, P_msg, mu_msg, A_edge, A_node, y_node, label_x, mu_xx) # compute p(x)
        x_pred = np.round(x_pred, 6)
        x_pred_all.append(x_pred)
        if c % 200 == 0:
            print "Processed " + str(c) + " data points"
        if debug:
            print "(idx, pred, true) = (" + str(idx) + ", " + str(x_pred) + ", " + str(x_true) + ")"

    # Return
    return x_pred_all, x_true_all

# The Gaussian loopy belief propagation algorithm
# See "Gaussian Belief Propagation: Theory and Application" for details
# See "Approximate Inference in Gaussian Graphical Models" for details
# The algorithm uses a Gaussian Markov random field (Gaussian graphical model)
# x_i is the value of the node x that we want to inference
# Edge potential = exp(-0.5*x_i*A_ij*x_j) for A_ij in A_edge
# Node potential = exp(-0.5*A_ii*x_i^2 + y_i*x_i) for A_ii in A_node, y_i in y_node
# INPUT:
# - G: the graphical model estimated by graphical Lasso
#   (type: a networkx Graph)
# - P_msg: parameter (precision) of initial message passing along each edge at both directions
#   (type: a dictionary with each key being a tuple of two nodes i and j
#          directions between nodes matter, tuple (i,j) means i -> j)
# - mu_msg: parameter (mean) of initial message passing along each edge at both directions
#   (type: same as P_msg)
# - A_edge: parameter (precision) of edge potentials
#   (type: a dictionary with each key being a tuple of two nodes, no directions)
# - A_node: parameter (precision) of node potentials
#   (type: a dictionary with each key being a node)
# - y_node: parameter (values of given variables y) of node potentials
#   (same as A_node)
# - label_x: the label of variable x that we want to inference
#   (type: a string)
# - mu_x: the empirical mean of variable x that we want to inference
#   (type: a number)
# OUTPUT:
# - x: the predicted marginal p(x)
#   (type: a number)
def GaLBP(G, P_msg, mu_msg, A_edge, A_node, y_node, label_x, mu_xx):
    debug = False
    max_iter = 20 # The iteration for looping all nodes in the graph
    epsilon = 0.0001 # the convergence threshold
    P_msg = copy.deepcopy(P_msg)
    mu_msg = copy.deepcopy(mu_msg)

    for c in range(0, max_iter):
        # Iteration for checking convergence of messages for all messages in the graph
        # We loop this until convergence
        if debug:
            print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print "@@@@@@@@@@@@@@@@@@@@@ Outer iteration " + str(c) + " @@@@@@@@@@@@@@@@@@@@@@@@@@@"
            print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        converged = True
        for i in G:
            # Iteration for scheduling message passing k -> i -> j
            # We need to pick a node i from all nodes in a certain order
            N_i = G[i] # neighbors of node i
            A_ii = A_node[i] # precision of node potential function
            if i == label_x:
                mu_ii = mu_xx
                #continue # IMPORTANT: do not run LBP on the node that we want to inference
            else:
                mu_ii = y_node[i] / A_ii # mean of node potential function
            for j in N_i:
                # Iteration for switching node j from neighbors of node i
                # The union set of node k and j is N_i, neighbors of node i
                # Each loop sets j to a different node in N_i, and k to N_i exclude j
                A_ij = A_edge[i,j]
                if len(N_i) != 1:
                    # This means that node i has more than one neighbors
                    # , so we need to collect messages from k
                    sum_P_ki_exclude_j = 0
                    sum_P_mu_ki_exclude_j = 0
                    for k in N_i:
                        if k != j: # exclude j from N_i
                            P_ki = P_msg[k,i]
                            sum_P_ki_exclude_j += P_ki
                            sum_P_mu_ki_exclude_j += P_ki * mu_msg[k,i]
                    # Compute P_ij, the precision passed from node i to neighbor j
                    P_ij = -np.power(A_ij, 2) / (A_ii + sum_P_ki_exclude_j)
                    # Compute mu_ij, the mean passed from node i to neighbor j
                    mu_ij = (A_ii * mu_ii + sum_P_mu_ki_exclude_j) / A_ij
                else:
                    # This means node i has only one neighbor
                    # , so we just pass message without collecting from k
                    P_ij = -np.power(A_ij, 2) / A_ii
                    mu_ij = (A_ii * mu_ii) / A_ij
                # Check if the message passing from node i to neighbors j converges
                if False:
                #if True:
                #if converged == True:
                    diff_P = abs(P_ij - P_msg[i,j])
                    diff_mu = abs(mu_ij - mu_msg[i,j])
                    if diff_P > epsilon or diff_mu > epsilon:
                        converged = False
                    if debug:
                        print "for node i = " + i + " -> for node j = " + j
                        print "    P_ij = " + str(round(P_ij,4)) + ", diff_P = " + str(round(diff_P,4))
                        print "    mu_ij = " + str(round(mu_ij,4)) + ", diff_mu = " + str(round(diff_mu,))
                # Update messages
                P_msg[i,j] = P_ij
                mu_msg[i,j] = mu_ij
        # We keep running this process until convergence
        if debug:
            print "--------------- Iteration " + str(c) + " converged? " + str(converged) + " -----------------" 
        # We keep running this process until convergence
        if converged:
            break

    # Compute the marginal mean and precision of node x
    sum_P_kx = 0
    sum_P_mu_kx = 0
    for k in G[label_x]: # for every node k in the neighbors of node x
        P_kx = P_msg[k,label_x]
        sum_P_kx += P_kx
        sum_P_mu_kx += P_kx * mu_msg[k,label_x]
    P_xx = A_node[label_x]
    P_x = P_xx + sum_P_kx # the marginal precision
    mu_x = (P_xx * mu_xx + sum_P_mu_kx) / P_x # the marginal mean
    if debug:
        print "------------------------------------------------------"
        print "Marginal precision: P_x = " + str(P_x)
        print "Marginal mean: mu_x = " + str(mu_x)
        print "------------------------------------------------------"

    return round(mu_x, 6)
