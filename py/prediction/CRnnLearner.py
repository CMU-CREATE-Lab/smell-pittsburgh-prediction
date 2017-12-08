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

# Convolutional Recurrent Neural Network for time-series prediction
# The CNN treats time series data as images
# The channel (e.g. rgb) represents features
# The height of an image represents time
# The width is always one
# A RNN is attached to the CNN
class CRnnLearner(object):
    def __init__(self,
            test=None, # the testing set for evaluating performance after each epoch
            batch_size=128, # size for each batch for training
            num_epochs=60, # number of epochs for training
            init_lr=0.0001, # initial learning rate
            l2_regu_weight_decay=0.0005, # loss function regularization
            lr_schedule_step_size=20, # number of epochs for decaying learning rate
            lr_schedule_gamma=0.5, # the decaying factor for learning rate
            use_class_weights=False, # use class weights when computing the loss
            is_regr=False,  # regression or classification
            use_rnn=True, # add recurrent neural net at the end of CNN or not
            clip_grad=False, # clip gradient or not
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
        self.use_rnn = use_rnn
        self.clip_grad = clip_grad

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

    # X need to have dimension (batch_size * feature_size * sequence_length * 1),
    # which corresponds to CNN input (batch_size * channel_size * height * width)
    # X: input predictors in pandas or numpy format
    # Y: input response in pandas or numpy format
    def fit(self, X, Y):
        start_time = datetime.now()
        self.log("==============================================================")
        self.log("batch_size = " + str(self.batch_size))
        self.log("num_epochs = " + str(self.num_epochs))
        self.log("init_lr = " + str(self.init_lr))
        self.log("l2_regu_weight_decay = " + str(self.l2_regu_weight_decay))
        self.log("lr_schedule_step_size = " + str(self.lr_schedule_step_size))
        self.log("lr_schedule_gamma = " + str(self.lr_schedule_gamma))
        self.log("use_class_weights = " + str(self.use_class_weights))
        self.log("is_regr = " + str(self.is_regr))
        self.log("use_rnn = " + str(self.use_rnn))
        self.log("clip_grad = " + str(self.clip_grad))
        self.log("--------------------------------------------------------------")
        
        # Parameters
        width = X.shape[3]
        height = X.shape[2]
        channel_size = X.shape[1]
        if self.is_regr:
            if len(Y.shape) == 1:
                output_size = 1
            else:
                output_size = Y.shape[-1]
        else:
            output_size = len(np.unique(Y))
       
        # Model
        model = CRNN(channel_size, height, width, output_size, use_rnn=self.use_rnn, use_cuda=self.use_cuda)
        if self.use_cuda:
            model.cuda()

        # Loss function
        if self.is_regr:
            criterion = nn.SmoothL1Loss()
            if self.use_class_weights:
                self.log("Regression will ignore class weights")
        else:
            criterion = nn.CrossEntropyLoss()
            #criterion = nn.MultiMarginLoss(p=1, margin=2)
            # Compute the weight of each class (because the dataset is imbalanced)
            if self.use_class_weights:
                class_weights = float(X.shape[0]) / (output_size * np.bincount(Y.squeeze()))
                class_weights = torch.FloatTensor(class_weights)
                if self.use_cuda: class_weights = class_weights.cuda()
                criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr, weight_decay=self.l2_regu_weight_decay)

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
                outputs = model(x) # forward propagation
                loss = criterion(outputs, y) # compute loss
                loss.backward() # backward propagation
                if self.clip_grad:
                    clip_grad_norm(model.parameters(), 10, norm_type=2)
                optimizer.step() # optimize
                loss_all.append(loss.data[0]) # save loss for each step
            self.model = model # save the model
            # Print the result for the entire epoch
            T_tr, P_tr = self.train["Y"], self.predict(self.train["X"])
            m_train = computeMetric(T_tr, P_tr, self.is_regr, flatten=True, simple=True, round_to_decimal=2)
            if self.test is not None:
                T_te, P_te = self.test["Y"], self.predict(self.test["X"])
                m_test = computeMetric(T_te, P_te, self.is_regr, flatten=True, simple=True, round_to_decimal=2)
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
        self.log("==============================================================")
        return self

    def predict(self, X, threshold=0.6):
        if self.is_regr:
            Y_pred = self.forward_from_numpy(X)
            Y_pred = Y_pred.data.numpy().squeeze()
        else:
            outputs = self.predict_proba(X)
            confidence, Y_pred = torch.max(outputs, 1)
            Y_pred = Y_pred.numpy()
            confidence = confidence.numpy()
            Y_pred[confidence < threshold] = 0 # predict zero when the confidence is less than the threshold
        return Y_pred

    def predict_proba(self, X):
        outputs = self.forward_from_numpy(X)
        sm = nn.Softmax()
        outputs = sm(outputs)
        return outputs.cpu().data

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

# ResNet Block
class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, only_one_branch=False, h_size_b2=16, s_size_b2=1):
        super(ResBlock, self).__init__()

        self.only_one_branch = only_one_branch

        # Branch 1
        if not only_one_branch:
            self.b1 = nn.Sequential(
                nn.SELU(),
                nn.Conv2d(input_size, output_size, kernel_size=1, padding=0, stride=(s_size_b2, 1), bias=False))

        # Branch 
        k_size_b2 = 3
        p_size_b2 = 1
        self.b2 = nn.Sequential(
            nn.SELU(),
            nn.Conv2d(input_size, h_size_b2, kernel_size=1, padding=0, stride=(s_size_b2, 1), bias=False),
            nn.SELU(),
            nn.ReplicationPad2d((0, 0, p_size_b2, p_size_b2)),
            nn.Conv2d(h_size_b2, h_size_b2, kernel_size=(k_size_b2, 1), padding=0, bias=False),
            nn.SELU(),
            nn.Conv2d(h_size_b2, output_size, kernel_size=1, padding=0, bias=False))
        
    def forward(self, x):
        if self.only_one_branch:
            f1 = x
        else:
            f1 = self.b1(x)
        f2 = self.b2(x)
        f = f1 + f2
        return f2

# ResNet Large Block
class ResLargeBlock(nn.Module):
    def __init__(self, input_size, output_size, h_size=16, s_size=1):
        super(ResLargeBlock, self).__init__()

        r1 = ResBlock(input_size, output_size, h_size_b2=h_size, s_size_b2=s_size)
        r2 = ResBlock(output_size, output_size, only_one_branch=True, h_size_b2=h_size, s_size_b2=1)
        r3 = ResBlock(output_size, output_size, only_one_branch=True, h_size_b2=h_size, s_size_b2=1)
        self.r = nn.Sequential(r1, r2, r3)

    def forward(self, x):
        return self.r(x)

# Residual RNN Block
class ResRnnBlock(nn.Module):
    def __init__(self, input_size, sequence_length, two_layer=False, bidirectional=True, use_residual=False):
        super(ResRnnBlock, self).__init__()
        
        self.two_layer = two_layer
        self.use_residual = use_residual

        num_directions = 2 if bidirectional else 1
        h_size = int(input_size/num_directions)

        self.rnn1 = nn.Sequential(
            nn.SELU(),
            nn.GRU(input_size, h_size, 1, batch_first=True, bidirectional=bidirectional, bias=False))
        if two_layer:
            self.rnn2 = nn.Sequential(
                nn.SELU(),
                nn.GRU(h_size*num_directions, h_size, 1, batch_first=True, bidirectional=bidirectional, bias=False))

    def forward(self, x):
        f, _ = self.rnn1(x)
        if self.use_residual:
            f = f.contiguous() + x
        if self.two_layer:
            ff, _ = self.rnn2(f)
            if self.use_residual:
                f = ff + f
            else:
                f = ff
        return f[:,-1,:]

# Fully Connnected Block
class FC(nn.Module):
    def __init__(self, input_size, output_size, bilinear=True):
        super(FC, self).__init__()
        
        self.bilinear = bilinear
        self.selu = nn.SELU()
        if bilinear:
            self.fc1 = nn.Linear(input_size, input_size, bias=True)
            self.fc2 = nn.Bilinear(input_size, input_size, output_size, bias=False)
        else:
            self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        f = self.selu(x)
        if self.bilinear:
            f1 = self.fc1(f)
            f = self.fc2(f, f1)
        else:
            f = self.fc(f)
        return f

class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.SELU(),
            nn.Conv2d(input_size, hidden_size, kernel_size=(3, 1), padding=0, stride=(1, 1), bias=False),
            nn.SELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 1), padding=0, stride=(1, 1), bias=False),
            nn.SELU(),
            nn.Conv2d(hidden_size, output_size, kernel_size=(3, 1), padding=0, stride=(1, 1), bias=False))

    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResNet, self).__init__()

        out_size_c1 = 32
        self.c = nn.Sequential(
            nn.SELU(),
            #nn.ReplicationPad2d((0, 0, 1, 1)),
            nn.Conv2d(input_size, out_size_c1, kernel_size=(3, 1), padding=0, stride=(1, 1), bias=False))
        self.r1 = ResLargeBlock(out_size_c1, output_size, h_size=16, s_size=1)

    def forward(self, x):
        f = self.c(x)
        f = self.r1(f)
        return f

# CNN Model (1D convolution on height of the image) attached with RNN
class CRNN(nn.Module):
    def __init__(self, channel_size, height, width, output_size, use_rnn=False, use_cuda=True):
        super(CRNN, self).__init__()
        
        # Hyper-parameters
        self.width = width
        self.height = height
        self.channel_size = channel_size
        self.use_rnn = use_rnn

        # CNN (Feature Extraction)
        hidden_cnn = 128
        output_cnn = 64
        if use_rnn:
            self.cnn = CNN(channel_size, output_cnn, hidden_cnn)
        else:
            self.cnn = nn.Sequential(
                CNN(channel_size, output_cnn, hidden_cnn),
                nn.MaxPool2d(kernel_size=(2, 1), padding=0, stride=(2, 1)))
        
        # RNN (Time Series)
        if use_rnn:
            rnn_input_size, rnn_input_seq_len = self.rnnLayerInputSize()
            print rnn_input_size, rnn_input_seq_len
            self.rnn = ResRnnBlock(rnn_input_size, rnn_input_seq_len, two_layer=False, use_residual=True)

        # Fully Connected
        fc_input_size = self.fullyConnectedLayerInputSize()
        print fc_input_size
        self.fc = FC(fc_input_size, output_size, bilinear=True)

    def forward(self, x):
        f = self.cnn(x)
        if self.use_rnn:
            f = self.cnnOutputToRnnInput(f)
            f = self.rnn(f)
        f = self.toFcInput(f)
        f = self.fc(f)
        return f

    def cnnOutputToRnnInput(self, x):
        # We use CNN to learn time-series features (convolutional filters)
        # Dimension of input X for CNN is (batch_size, channel_size, height, width)
        # channel_size in CNN is 1
        # height in CNN represents sequence_length
        # width in CNN represents feature_size
        # So the dimension of output X for CNN is (batch_size, feature_size, sequence_length, 1)
        # Dimension of input X for RNN should be (batch_size, sequence_length, feature_size)
        f = x.permute(0, 2, 1, 3)
        f = f.contiguous().view(f.size(0), f.size(1), -1)
        return f

    def toFcInput(self, x):
        f = x.contiguous().view(x.size(0), -1)
        return f

    def rnnLayerInputSize(self):
        x = self.genDummyInput()
        f = self.cnn(x)
        f = self.cnnOutputToRnnInput(f)
        return f.size(2), f.size(1)

    def fullyConnectedLayerInputSize(self):
        x = self.genDummyInput()
        f = self.cnn(x)
        print f.size()
        if self.use_rnn:
            f = self.cnnOutputToRnnInput(f)
            f = self.rnn(f)
        f = self.toFcInput(f)
        return f.size(1)

    def genDummyInput(self):
        return Variable(torch.randn(1, self.channel_size, self.height, self.width))
