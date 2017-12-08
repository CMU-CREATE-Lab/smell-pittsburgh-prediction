import numpy as np
from util import *
from CRnnLearner import CRnnLearner
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

class CRMAnnLearner(object):
    def __init__(self,
            test=None,
            is_regr=False,
            logger=None):

        # Set testing dataset
        self.test = test
       
        # Set hyper-parameters
        self.is_regr = is_regr

        # set the logger
        self.logger = logger

    # X: input predictors in numpy format
    # Y: input response in numpy format
    def fit(self, X, Y):
        # First stage many to many regression
        sequence_length = int(X.shape[2] / 2)
        train_1 = {
            "X": X[:, 1:, :, :],
            "Y": X[:, 0, sequence_length:, :].squeeze()
        }
        test_1 = {
            "X": self.test["X"][:, 1:, :, :],
            "Y": self.test["X"][:, 0, sequence_length:, :].squeeze()
        }
        regressor = CRnnLearner(test=test_1, logger=self.logger, is_regr=True)
        regressor.fit(train_1["X"], train_1["Y"])
        
        # Add errors back to the features
        train_1_err = train_1["Y"] - regressor.predict(train_1["X"])
        train_1_err = np.expand_dims(np.expand_dims(train_1_err, axis=1), axis=3)
        train_2 = {
            "X": np.append(X[:, :, sequence_length:, :], train_1_err, axis=1),
            "Y": Y
        }
        test_1_err = test_1["Y"] - regressor.predict(test_1["X"]) 
        test_1_err = np.expand_dims(np.expand_dims(test_1_err, axis=1), axis=3)
        test_2 = {
            "X": np.append(self.test["X"][:, :, sequence_length:, :], test_1_err, axis=1),
            "Y": self.test["Y"]
        }

        # Second stage of classification
        classifier = CRnnLearner(test=test_2, logger=self.logger, is_regr=self.is_regr)
        classifier.fit(train_2["X"], train_2["Y"])
        
        # Save models
        self.regressor = regressor
        self.classifier = classifier

        return self

    def predict(self, X):
        sequence_length = int(X.shape[2] / 2)
        X_1 = X[:, 1:, :, :]
        Y_1 = X[:, 0, sequence_length:, :].squeeze()
        err = Y_1 - self.regressor.predict(X_1)
        err = np.expand_dims(np.expand_dims(err, axis=1), axis=3)
        X_2 = np.append(X[:, :, sequence_length:, :], err, axis=1)
        return self.classifier.predict(X_2)

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
