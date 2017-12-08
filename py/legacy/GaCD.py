import numpy as np
import numpy.matlib
import copy
from learnStructure import *

# Inference p(x) by computing multivariate Gaussian conditional distribution
# INPUT:
# - df_train: the training set
#   (type: pandas data frame)
# - df_test: the testing set
#   (type: same as df_train)
# - label_x: the label of variable x that we want to inference
#   (type: a string of the column name of the pandas dataframe)
# OUTPUT:
# - x_pred_all: the predicted value of x in the testing set
#   (type: an array of numbers)
# - x_true_all: the true value of x in the testing set
#   (type: same as x_pred)
def GaCD_main(df_train, df_test, label_x, df_prec):
    debug = False

    label_x_loc = df_train.columns.get_loc(label_x) # integer of where the label is

    # Separate the mean of y and x
    mu = df_train.mean().copy().values # copy to prevent from modifying it
    mu[[label_x_loc,0]] = mu[[0,label_x_loc]] # permute rows
    mu_x = mu[0] # mean of the variable that we want to inference
    mu_y = mu[1:] # mean of the variables that have given values

    # Separate the covariance matrix of y and x
    prec = df_prec.copy().values # copy to prevent from modifying it
    S = np.linalg.inv(prec)
    S[[label_x_loc,0],:] = S[[0,label_x_loc],:] # permute rows
    S[:,[label_x_loc,0]] = S[:,[0,label_x_loc]] # permute columns
    S_xx = S[label_x_loc, label_x_loc]
    S_xy = S[0,1:]
    S_yx = S[1:,0]
    S_yy = S[1:,1:]
    S_yy_inv = np.linalg.pinv(S_yy)
    S_yx_dot_S_yy_inv = S_yx.dot(S_yy_inv)

    # Inference p(x)
    x_true_all = []
    x_pred_all = []
    c = 0
    for idx, row in df_test.iterrows():
        c += 1
        x_true = np.round(row[label_x], 6) # true value of x
        x_true_all.append(x_true)
        row_cp = row.copy().values # copy to prevent from modifying it
        row_cp[[label_x_loc,0]] = row_cp[[0,label_x_loc]] # permute rows
        y = row_cp[1:] # values of given variables
        x_pred = GaCD(mu_x, S_yx_dot_S_yy_inv, y, mu_y)
        x_pred = np.round(x_pred, 6)
        x_pred_all.append(x_pred)
        if c % 200 == 0:
            print "Processed " + str(c) + " data points"
        if debug:
            print "(idx, pred, true) = (" + str(idx) + ", " + str(x_pred) + ", " + str(x_true) + ")"

    # Return
    return x_pred_all, x_true_all

# Compute the multivariate Gaussian conditional distribution p(x|y)
def GaCD(mu_x, S_yx_dot_S_yy_inv, y, mu_y):
    x_pred = mu_x + S_yx_dot_S_yy_inv.dot(y - mu_y)
    return x_pred
