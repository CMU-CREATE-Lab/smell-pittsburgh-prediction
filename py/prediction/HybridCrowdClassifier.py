from util import *
import numpy as np

# A hybrid model that combines crowdsourcing and machine learning classifier
class HybridCrowdClassifier(object):
    def __init__(self,
            base_estimator=None, # if no estimator is specified, only use crowdsourcing
            logger=None):
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        self.base_estimator.fit(X, Y)

    # Be careful that this function only works for binary classification
    def predict(self, X, threshold=0.5):
        prob = self.base_estimator.predict_proba(X)
        res = (prob[:,1] > threshold).astype(int)
        return res

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def save(self, out_path):
        return None

    def load(self, in_path):
        return None

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
