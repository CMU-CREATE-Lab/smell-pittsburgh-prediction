from util import *
import numpy as np

# A hybrid model that combines crowdsourcing and machine learning classifier
class HybridCrowdClassifier(object):
    def __init__(self,
            base_estimator=None, # if no estimator is specified, only use crowdsourcing
            logger=None):
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        return None

    def predict(self, X, threshold=0.6):
        return None

    def predict_proba(self, X):
        return None

    def save(self, out_path):
        return None

    def load(self, in_path):
        return None

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)
