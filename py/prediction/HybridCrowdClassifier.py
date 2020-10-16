import numpy as np


"""
A hybrid model that combines crowdsourcing and machine learning classifier
"""
class HybridCrowdClassifier(object):
    def __init__(self,
            base_estimator=None, # if no estimator is specified, only use crowdsourcing
            crowd_thr=20, # the threshold to determine an event detected by the crowd
            logger=None):
        self.base_estimator = base_estimator
        self.crowd_thr = crowd_thr

    def fit(self, X, Y):
        if self.base_estimator is not None:
            self.base_estimator.fit(X, Y)

    # Three situations for prediction: 0, 1, 2
    # 0 means no event
    # 1 means the event predictedd by the base estimator
    # 2 means the event noticed by the crowd
    def predict(self, X, crowd, X_thr=0.5):
        # Use crowd to predict result
        pred = np.squeeze(crowd>=self.crowd_thr) * 2

        # Use the model to predict result and merge them
        # if pred==0, no event
        # if pred==1, event predicted by the base estimator
        # if pred==2, event detected by the crowd
        # if pred==3, event both predicted by the base estimator and detected by the crowd
        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
            pred_model = (prob_model[:,1] > X_thr).astype(int)
            pred += pred_model

        return pred

    def predict_proba(self, X, crowd):
        # Replace value > crowd_thr with [0.0, 1.0]
        L = len(crowd)
        prob = np.array([[1.0, 0.0]]*L)
        prob[np.squeeze(crowd>=self.crowd_thr)] = [0.0, 1.0]

        if self.base_estimator is not None:
            prob_model = self.base_estimator.predict_proba(X)
            # If no crowd event, then the prediction uses the base estimator
            idx = np.squeeze(crowd<self.crowd_thr)
            prob[idx] = prob_model[idx]

        return prob

    def save(self, out_path):
        return None

    def load(self, in_path):
        return None

    def log(self, msg):
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)
