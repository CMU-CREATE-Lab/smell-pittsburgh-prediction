from util import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.utils import shuffle
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from torch.nn.utils import clip_grad_norm

# Deep multi-layer perceptron
class DmlpLearner(object):
    def __init__(self,
            test=None, # the testing set for evaluating performance after each epoch
            batch_size=128, # size for each batch
            num_epochs=40, # number of epochs
            init_lr=0.001, # initial learning rate
            l2_regu_weight_decay=0.001, # loss function regularization
            lr_schedule_step_size=10, # number of epochs for decaying learning rate
            lr_schedule_gamma=0.5, # the decaying factor for learning rate
            use_class_weights=False, # use class weights when computing the loss
            is_regr=False,  # regression or classification
            use_cuda=True, # use GPU or not
            logger=None):

        # Set testing dataset
        self.test = test
        
        # Set hyper-parameters
        self.batch_size = batch_size 
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.l2_regu_weight_decay = l2_regu_weight_decay
        self.lr_schedule_step_size = lr_schedule_step_size
        self.lr_schedule_gamma = lr_schedule_gamma
        self.use_class_weights = use_class_weights
        self.is_regr = is_regr 

        # set the logger
        self.logger = logger

        # use GPU or not
        if use_cuda:
            if torch.cuda.is_available:
                self.log("Cuda available. Use GPU...")
            else:
                use_cuda = False
                self.log("Cuda unavailable. Use CPU...")
        else:
            self.log("Use CPU...")
        self.use_cuda = use_cuda

    # X: input predictors in pandas or numpy format
    # Y: input response in pandas or numpy format
    def fit(self, X, Y):
        start_time = datetime.now()
        
        # Parameters
        input_size = X.shape[1]
        if self.is_regr:
            if len(Y.shape) == 1:
                output_size = 1
            else:
                output_size = Y.shape[-1]
        else:
            output_size = len(np.unique(Y))
       
        # Model
        model = DMLP(input_size, output_size, use_cuda=self.use_cuda)
        self.log(model)
        if self.use_cuda:
            model.cuda()
        self.model = model
        
        # Print hyper parameters
        self.log("==============================================================")
        self.log("Supervised learning with input " + str(X.shape))
        self.log("batch_size = " + str(self.batch_size))
        self.log("num_epochs = " + str(self.num_epochs))
        self.log("init_lr = " + str(self.init_lr))
        self.log("l2_regu_weight_decay = " + str(self.l2_regu_weight_decay))
        self.log("lr_schedule_step_size = " + str(self.lr_schedule_step_size))
        self.log("lr_schedule_gamma = " + str(self.lr_schedule_gamma))
        self.log("use_class_weights = " + str(self.use_class_weights))
        self.log("is_regr = " + str(self.is_regr))
        self.log("--------------------------------------------------------------")
        start_time = datetime.now()
        
        # Loss function
        if self.is_regr:
            criterion = nn.MSELoss()
            #criterion = nn.SmoothL1Loss()
            if self.use_class_weights:
                self.log("Regression will ignore class weights")
        else:
            #criterion = nn.CrossEntropyLoss()
            criterion = nn.MultiMarginLoss()
            # Compute the weight of each class (because the dataset is imbalanced)
            if self.use_class_weights:
                class_weights = float(X.shape[0]) / (output_size * np.bincount(Y.squeeze()))
                class_weights = torch.FloatTensor(class_weights)
                if self.use_cuda: class_weights = class_weights.cuda()
                criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.init_lr, weight_decay=self.l2_regu_weight_decay)

        # Learning rate scheduler
        rule = lambda epoch: self.lr_schedule_gamma ** (epoch // self.lr_schedule_step_size)
        scheduler = LambdaLR(optimizer, lr_lambda=[rule])

        # Save original training data
        self.train = {"X": deepcopy(X), "Y": deepcopy(Y)}

        # Break data into batches
        num_of_left_overs = self.batch_size - (X.shape[0] % self.batch_size)
        X = np.append(X, X[0:num_of_left_overs], 0)
        Y = np.append(Y, Y[0:num_of_left_overs], 0)
        num_of_batches = X.shape[0] // self.batch_size
        X = np.split(X, num_of_batches, 0)
        Y = np.split(Y, num_of_batches, 0)
        
        # Train the Model
        for epoch in range(1, self.num_epochs+1):
            X, Y = shuffle(X, Y) # shuffle batches
            loss_all = [] # for saving the loss in each step
            scheduler.step() # adjust learning rate
            # Loop through all batches
            for x, y in zip(X, Y):
                x = torch.FloatTensor(x)
                if self.is_regr:
                    y = torch.FloatTensor(y)
                else:
                    y = torch.LongTensor(y)
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad() # reset gradient
                outputs = self.model(x) # forward propagation
                loss = criterion(outputs, y) # compute loss
                loss.backward() # backward propagation
                optimizer.step() # optimize
                loss_all.append(loss.data[0]) # save loss for each step
            # Print the result for the entire epoch
            T_tr, P_tr = self.train["Y"], self.predict(self.train["X"])
            m_train = computeMetric(T_tr, P_tr, self.is_regr, flatten=True, simple=True, aggr_axis=True)
            if self.test is not None:
                T_te, P_te = self.test["Y"], self.predict(self.test["X"])
                m_test = computeMetric(T_te, P_te, self.is_regr, flatten=True, simple=True, aggr_axis=True)
            lr_now = optimizer.state_dict()["param_groups"][0]["lr"]
            avg_loss = np.mean(loss_all)
            if self.is_regr:
                if self.test is not None:
                    self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [mse, r2], [%2f, %2f], [%2f, %2f]'
                        %(epoch, self.num_epochs, lr_now, avg_loss, m_train["mse"], m_train["r2"],
                        m_test["mse"], m_test["r2"]))
                else:
                    self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [mse, r2], [%5d, %5d]'
                        %(epoch, self.num_epochs, lr_now, avg_loss, m_train["mse"], m_train["r2"]))
            else:
                cm_names = " ".join(m_train["cm"][0])
                cm_train = " ".join(map(lambda x: '%5d'%(x), m_train["cm"][1]))
                if self.test is not None:
                    cm_test = " ".join(map(lambda x: '%4d'%(x), m_test["cm"][1]))
                    self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [%s], [%s], [%s]'
                        %(epoch, self.num_epochs, lr_now, avg_loss, cm_names, cm_train, cm_test))
                else:
                    self.log('[%2d/%d], LR: %.9f, Loss: %.9f, [%s], [%s]'
                        %(epoch, self.num_epochs, lr_now, avg_loss, cm_names, cm_train))

        self.log("--------------------------------------------------------------")
        self.log("From " + str(datetime.now()) + " to " + str(start_time))
        self.log("--------------------------------------------------------------")
        return self

    def predict(self, X, threshold=0.5):
        if self.is_regr:
            Y_pred = self.forward_from_numpy(X)
            Y_pred = Y_pred.data.numpy().squeeze()
            # smell values should be larger than zero
            Y_pred[Y_pred < 0] = 0
        else:
            outputs = self.predict_proba_torch(X)
            confidence, Y_pred = torch.max(outputs, 1)
            Y_pred = Y_pred.numpy()
            confidence = confidence.numpy()
            Y_pred[confidence < threshold] = 0 # predict zero when the confidence is less than the threshold
        return Y_pred

    def predict_proba_torch(self, X):
        outputs = self.forward_from_numpy(X)
        sm = nn.Softmax()
        outputs = sm(outputs)
        return outputs.cpu().data

    def predict_proba(self, X):
        p = self.predict_proba_torch(X)
        return p.numpy()

    def forward_from_numpy(self, X):
        X = torch.FloatTensor(X)
        if self.use_cuda: X = X.cuda()
        X = Variable(X)
        outputs = self.model(X)
        return outputs.cpu()

    def save(self, out_path):
        torch.save(self.model.state_dict(), out_path)

    def load(self, in_path):
        self.model.load_state_dict(torch.load(in_path))

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)

class DMLP(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(DMLP, self).__init__()
       
        # Fully Connected
        hidden_size = 128
        hidden_size_2 = 64
        self.fc = nn.Sequential(
            nn.SELU(),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size_2, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size_2, output_size, bias=False))

    def forward(self, x):
        f = self.fc(x)
        return f
