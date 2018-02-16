from util import *
import numpy as np
import copy

# A hybrid model that combines crowdsourcing and machine learning classifier
class HybridCrowdClassifier(object):
    def __init__(self,
            base_estimator=None, # if no estimator is specified, only use crowdsourcing
            logger=None):
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        if self.base_estimator is not None:
            self.base_estimator.fit(X, Y)

    # Three situations for prediction: 0, 1, 2
    # 0 means no event
    # 1 means the event predictedd by the base estimator
    # 2 means the event noticed by the crowd
    def predict(self, X, Y_previous, threshold=0.5):
        # Use crowd to predict result
        pred = Y_previous * 2
        
        # Use the model to predict result and merge them
        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
            pred_model = (prob_model[:,1] > threshold).astype(int)
            pred = np.maximum(pred, pred_model)
        
        return pred

    def predict_proba(self, X, Y_previous):
        # Replace 0 with [1.0, 0.0] and 1 with [0.0 1.0]
        L = len(Y_previous)
        prob = np.array([[1.0, 0.0]]*L)
        prob[Y_previous==1] = [0.0, 1.0]

        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
            prob[Y_previous==0] = prob_model[Y_previous==0]
        
        return prob

    def save(self, out_path):
        return None

    def load(self, in_path):
        return None

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
