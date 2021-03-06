import numpy as np


class HybridCrowdClassifier():
    """
    A hybrid model that combines crowdsourcing and machine learning classifier
    """
    
    def __init__(self, base_estimator=None, crowd_thr=20, logger=None):
        """
        Initialize the class

        Input:
            base_estimator: the base scikit-learn model for predicting smell events
                ...if no estimator is specified, only use the crowd-based smell event
            crowd_thr: the threshold to determine an event detected by the crowd
                ...if the sum of total smell ratings for all reports in the previous hour is larger than crowd_thr,
                ...the model will say that there is a crowd-based smell event
        """
        self.base_estimator = base_estimator
        self.crowd_thr = crowd_thr
        self.logger = logger

    def fit(self, X, Y):
        """Train the model"""
        if self.base_estimator is not None:
            self.base_estimator.fit(X, Y)

    def predict(self, X, crowd, X_thr=0.5):
        """
        Predict the result
        
        There are three situations for prediction: 0, 1, 2
            0 means no event
            1 means the event predictedd by the base estimator
            2 means the event noticed by the crowd
        """
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
        """Compute the probability for each class when predicting the result"""
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
        """Override the function for saving the model"""
        return None

    def load(self, in_path):
        """Override the function for loading the model"""
        return None

    def log(self, msg):
        """Log messages"""
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)
