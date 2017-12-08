import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.utils import shuffle
from copy import deepcopy
from util import *
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from torch.nn.utils import clip_grad_norm
#from torch.nn.utils import weight_norm

# Recurrent Neural Network
# Use Long Short-Term Memory or Gated Recurrent Unit for time-series prediction
class RnnClassifier(object):
    def __init__(self,
            test=None,
            hidden_size=64,
            num_layers=4,
            batch_size=16,
            num_epochs=100,
            init_lr=0.0005,
            l2_regu_weight_decay=0.0001,
            lr_schedule_step_size=10,
            lr_schedule_gamma=0.5,
            use_class_weights=False,
            clip_grad_norm_type=2,
            reverse_input_seq=False,
            dropout=0,
            max_grad_norm=8,
            bidirectional=False,
            use_all_out=False,
            use_gru=True,
            use_cuda=True,
            logger=None):

        # Set testing dataset
        self.test = test

        # Set hyper-parameters
        self.hidden_size = hidden_size # number of hidden units in each LSTM layer
        self.num_layers = num_layers # number of LSTM layers
        self.batch_size = batch_size # size for each batch for training
        self.num_epochs = num_epochs # number od epochs for training
        self.init_lr = init_lr # initial learning rate
        self.l2_regu_weight_decay = l2_regu_weight_decay # L2 regularization for the loss function
        self.lr_schedule_step_size = lr_schedule_step_size # number of epochs for decaying learning rate
        self.lr_schedule_gamma = lr_schedule_gamma # the decaying factor for learning rate
        self.clip_grad_norm_type = clip_grad_norm_type # the type of norm for clipping gradient
        self.use_class_weights = use_class_weights # use class weights when computing the loss
        self.reverse_input_seq = reverse_input_seq # reverse the input time sequence or not
        self.dropout = dropout # the dropout probability of the LSTM layers
        self.max_grad_norm = max_grad_norm # the maximum norm for clipping the gradient
        self.bidirectional = bidirectional # use bidirectional LSTM or not
        self.use_all_out = use_all_out # connect all hidden units from all sequences to the output or not
        self.use_gru = use_gru # use GRU cell instead of LSTM cell or not
        
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

    # X: input predictors in numpy format
    # Y: input response in numpy format
    def fit(self, X, Y):
        start_time = datetime.now()
        self.log("==============================================================")
        self.log("hidden_size = " + str(self.hidden_size))
        self.log("num_layers = " + str(self.num_layers))
        self.log("batch_size = " + str(self.batch_size))
        self.log("num_epochs = " + str(self.num_epochs))
        self.log("init_lr = " + str(self.init_lr))
        self.log("l2_regu_weight_decay = " + str(self.l2_regu_weight_decay))
        self.log("lr_schedule_step_size = " + str(self.lr_schedule_step_size))
        self.log("lr_schedule_gamma = " + str(self.lr_schedule_gamma))
        self.log("clip_grad_norm_type = " + str(self.clip_grad_norm_type))
        self.log("use_class_weights = " + str(self.use_class_weights))
        self.log("reverse_input_seq = " + str(self.reverse_input_seq))
        self.log("dropout = " + str(self.dropout))
        self.log("max_grad_norm = " + str(self.max_grad_norm))
        self.log("bidirectional = " + str(self.bidirectional))
        self.log("use_all_out = " + str(self.use_all_out))
        self.log("use_gru = " + str(self.use_gru))
        self.log("--------------------------------------------------------------")

        # Parameters
        # X dimension is (batch_size * sequence_length * feature_size)
        sequence_length = X.shape[1]
        feature_size = X.shape[2]
        num_classes = len(np.unique(Y))
       
        # Model
        model = RNN(feature_size, self.hidden_size, self.num_layers, num_classes, sequence_length,
                dropout=self.dropout,
                use_cuda=self.use_cuda,
                bidirectional=self.bidirectional,
                use_all_out=self.use_all_out,
                use_gru=self.use_gru)
        if self.use_cuda: model.cuda()

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Compute the weight of each class (because the dataset is imbalanced)
        if self.use_class_weights:
            class_weights = float(X.shape[0]) / (num_classes * np.bincount(Y))
            class_weights = torch.FloatTensor(class_weights)
            if self.use_cuda: class_weights = class_weights.cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr, weight_decay=self.l2_regu_weight_decay)

        # Learning rate scheduler
        rule = lambda epoch: self.lr_schedule_gamma ** (epoch // self.lr_schedule_step_size)
        scheduler = LambdaLR(optimizer, lr_lambda=[rule])

        # Reverse the input time sequence
        if self.reverse_input_seq: X = np.flip(X, axis=1)

        # Save original training data
        self.train = {"X":deepcopy(X), "Y":deepcopy(Y)}

        # Break data into batches
        
        num_of_left_overs = self.batch_size - (X.shape[0] % self.batch_size)
        X = np.append(X, X[0:num_of_left_overs, :, :], 0)
        Y = np.append(Y, Y[0:num_of_left_overs])
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
                x, y = torch.FloatTensor(x), torch.LongTensor(y)
                if self.use_cuda: x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad() # reset gradient
                outputs = model(x) # forward propagation
                loss = criterion(outputs, y) # compute loss
                loss.backward() # backward propagation
                clip_grad_norm(model.parameters(), self.max_grad_norm, norm_type=self.clip_grad_norm_type) # clip gradient
                optimizer.step() # optimize
                loss_all.append(loss.data[0]) # save loss for each step
            self.model = model # save the model
            # Print the result for the entire epoch
            T_tr, P_tr = self.train["Y"], self.predict(self.train["X"])
            m_train = computeMetric(T_tr, P_tr, False, flatten=True, simple=True, round_to_decimal=2)
            if self.test is not None:
                T_te, P_te = self.test["Y"], self.predict(self.test["X"])
                m_test = computeMetric(T_te, P_te, False, flatten=True, simple=True, round_to_decimal=2)
            lr_now = optimizer.state_dict()["param_groups"][0]["lr"]
            avg_loss = np.mean(loss_all)
            cm_names = " ".join(m_train["cm"][0])
            cm_train = " ".join(map(lambda x: '%5d'%(x), m_train["cm"][1]))
            if self.test is not None:
                cm_test = " ".join(map(lambda x: '%4d'%(x), m_test["cm"][1]))
                self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [%s], [%s], [%s]'
                    %(epoch, self.num_epochs, lr_now, avg_loss, cm_names, cm_train, cm_test))
            else:
                self.log('[%2d/%d], LR: %.9f, Loss: %.9f, [%s], [%s]'
                    %(epoch, self.num_epochs, lr_now, avg_loss, cm_names, cm_train))

        # Log the final result
        self.log("--------------------------------------------------------------")
        m_train = computeMetric(T_tr, P_tr, False)
        for m in m_train:
            self.log("Metric: " + m)
            self.log(m_train[m])
        if self.test is not None:
            self.log("--------------------------------------------------------------")
            m_test = computeMetric(T_te, P_te, False)
            for m in m_test:
                self.log("Metric: " + m)
                self.log(m_test[m])
        self.log("--------------------------------------------------------------")
        self.log("From " + str(datetime.now()) + " to " + str(start_time))
        self.log("==============================================================")
        return self

    def predict(self, X, threshold=0.6):
        if self.reverse_input_seq: X = np.flip(X, axis=1) # Reverse the input time sequence
        outputs = self.forward_from_numpy(X)
        confidence, Y_pred = torch.max(outputs.data, 1)
        Y_pred = Y_pred.numpy()
        confidence = confidence.numpy()
        Y_pred[confidence < threshold] = 0 # predict zero when the confidence is less than the threshold
        return Y_pred

    def predict_proba(self, X):
        return self.forward_from_numpy(X).numpy()

    def forward_from_numpy(self, X):
        X = torch.FloatTensor(X)
        if self.use_cuda: X = X.cuda()
        X = Variable(X)
        outputs = self.model(X)
        sm = nn.Softmax()
        outputs = sm(outputs)
        return outputs.cpu()

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)

# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, num_classes, sequence_length,
            use_cuda=False, dropout=0, bidirectional=False, use_all_out=False, use_gru=False):
        super(RNN, self).__init__()
        
        # Hyper-parameters
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.feature_size = feature_size
        self.use_gru = use_gru
        self.use_all_out = use_all_out
        self.num_directions =  2 if bidirectional else 1
        
        # Layers        
        self.relu = nn.ReLU()
        if use_gru:
            self.gru = nn.GRU(feature_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(feature_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional)
        #for name, param in self.lstm.named_parameters():
        #    if "weight" in name: self.lstm = weight_norm(self.lstm, name)
        self.final_layer_size = hidden_size * self.num_directions
        if use_all_out: self.final_layer_size *= sequence_length
        self.fc = nn.Linear(self.final_layer_size, num_classes)
        #self.fc = weight_norm(self.fc)
        #self.bn = nn.BatchNorm1d(self.final_layer_size)

    def initHidden(self, x):
        d = self.num_layers * self.num_directions
        h0 = torch.zeros(d, x.size(0), self.hidden_size)
        if self.use_cuda: h0 = h0.cuda()
        self.h0 = Variable(h0, requires_grad=False)
        if not self.use_gru:
            c0 = torch.zeros(d, x.size(0), self.hidden_size)
            if self.use_cuda: c0 = c0.cuda()
            self.c0 = Variable(c0, requires_grad=False)

    def forward(self, x):
        # Set initial states
        # Currently we reset the hidden states at each forward step (stateless LSTM) because we shuffle the batches
        # If we want batch n to depend on the state in batch n-1 (stateful LSTM), we cannot shuffle batches
        self.initHidden(x)

        # Forward propagate RNN
        if self.use_gru:
            out, hn = self.gru(x, self.h0)
        else:
            out, (hn, cn) = self.lstm(x, (self.h0, self.c0))
        
        # Decode hidden states
        if self.use_all_out:
            out = out.contiguous().view(-1, self.final_layer_size)
        else:
            out = out[:, -1, :]
        #out = self.bn(out)
        out = self.fc(out)
        return out
