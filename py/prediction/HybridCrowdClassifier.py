from util import *
import numpy as np
import copy

# A hybrid model that combines crowdsourcing and machine learning classifier
class HybridCrowdClassifier(object):
    def __init__(self,
            base_estimator=None, # if no estimator is specified, only use crowdsourcing
            logger=None):
        self.base_estimator = None#base_estimator

    def fit(self, X, Y):
        if self.base_estimator is not None:
            self.base_estimator.fit(X, Y)

    # Be careful that this function only works for binary classification
    def predict(self, X, Y_previous, threshold=0.5):
        # Use crowd to predict result
        pred = Y_previous
        
        # Use the model to predict result and merge them
        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
            pred_model = (prob_model[:,1] > threshold).astype(int)
            pred = pred | pred_model

        return pred

    def predict_proba(self, X, Y_previous):
        # Replace 0 with [1.0, 0.0] and 1 with [0.0 1.0]
        prob = copy.deepcopy(Y_previous)
        prob[prob == 0] = [1.0, 0.0]
        prob[prob == 1] = [0.0, 1.0]
        print prob

        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
        print Y_previous
        print prob_model
        return prob_model

    def save(self, out_path):
        return None

    def load(self, in_path):
        return None

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
