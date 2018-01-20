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

# Autoencoder Convolutional Neural Network for time-series prediction
# The CNN treats time series data as images
# The channel (e.g. rgb) represents features
# The height of an image represents time
# The width is always one
class ANCnnLearner(object):
    def __init__(self,
            test=None, # the testing set for evaluating performance after each epoch
            batch_size_pre=128, # size for each batch (pre-train)
            num_epochs_pre=20, # number of epochs (pre-train)
            init_lr_pre=0.001, # initial learning rate (pre-train)
            l2_regu_weight_decay_pre=0.0001, # loss function regularization (pre-train)
            lr_schedule_step_size_pre=5, # number of epochs for decaying learning rate (pre-train)
            lr_schedule_gamma_pre=0.5, # the decaying factor for learning rate (pre-train)
            batch_size=64, # size for each batch
            num_epochs=30, # number of epochs
            init_lr=0.0008, # initial learning rate
            l2_regu_weight_decay=0.0001, # loss function regularization
            lr_schedule_step_size=5, # number of epochs for decaying learning rate
            lr_schedule_gamma=0.5, # the decaying factor for learning rate
            use_class_weights=False, # use class weights when computing the loss
            is_regr=False,  # regression or classification
            use_cuda=True, # use GPU or not
            logger=None):

        # Set testing dataset
        self.test = test
        
        # Set hyper-parameters
        self.batch_size = batch_size 
        self.batch_size_pre = batch_size_pre
        self.num_epochs = num_epochs
        self.num_epochs_pre = num_epochs_pre
        self.init_lr = init_lr
        self.init_lr_pre = init_lr_pre
        self.l2_regu_weight_decay = l2_regu_weight_decay
        self.l2_regu_weight_decay_pre = l2_regu_weight_decay_pre
        self.lr_schedule_step_size = lr_schedule_step_size
        self.lr_schedule_step_size_pre = lr_schedule_step_size_pre
        self.lr_schedule_gamma = lr_schedule_gamma
        self.lr_schedule_gamma_pre = lr_schedule_gamma_pre
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

    # X need to have dimension (batch_size * feature_size * sequence_length * 1),
    # which corresponds to CNN input (batch_size * channel_size * height * width)
    # X: input predictors in pandas or numpy format
    # Y: input response in pandas or numpy format
    def fit(self, X, Y, X_pretrain):
        start_time = datetime.now()
        
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
        model = ANCNN(channel_size, height, width, output_size, use_cuda=self.use_cuda)
        model.apply(self.weights_init)
        self.log(model)
        if self.use_cuda:
            model.cuda()
        self.model = model
        
        #self.pre_train(X_pretrain)
        #model.freezeEncoder()
        self.fine_tune(X, Y)
    
    # Train a denoising autoencoder
    # First corrupt the inputs randomly with missing values (prevent the autoencoer to learn identical mapping)
    # Then train the autoencoder to map corrupted inputs to the original inputs
    def pre_train(self, X):
        self.log("==============================================================")
        self.log("Pretrain a denoising autoencoder with input " + str(X.shape))
        self.log("batch_size_pre = " + str(self.batch_size_pre))
        self.log("num_epochs_pre = " + str(self.num_epochs_pre))
        self.log("init_lr_pre = " + str(self.init_lr_pre))
        self.log("l2_regu_weight_decay_pre = " + str(self.l2_regu_weight_decay_pre))
        self.log("lr_schedule_step_size_pre = " + str(self.lr_schedule_step_size_pre))
        self.log("lr_schedule_gamma_pre = " + str(self.lr_schedule_gamma_pre))
        self.log("--------------------------------------------------------------")
        start_time = datetime.now()
        
        # Randomly corrupt the input with missing values
        C = 0.2 # percentage of missing values
        X_corrupt = deepcopy(X)
        mask = np.random.choice([0, 1], size=X_corrupt.shape, p=[C, 1-C])
        X_corrupt = np.multiply(X_corrupt, mask)

        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr_pre, weight_decay=self.l2_regu_weight_decay_pre)

        # Learning rate scheduler
        rule = lambda epoch: self.lr_schedule_gamma_pre ** (epoch // self.lr_schedule_step_size_pre)
        scheduler = LambdaLR(optimizer, lr_lambda=[rule])

        # Save original training data
        self.train = {"X": deepcopy(X), "X_corrupt": deepcopy(X_corrupt)}

        # Break data into batches
        num_of_left_overs = self.batch_size_pre - (X.shape[0] % self.batch_size_pre)
        X = np.append(X, X[0:num_of_left_overs], 0)
        X_corrupt = np.append(X_corrupt, X_corrupt[0:num_of_left_overs], 0)
        num_of_batches = X.shape[0] // self.batch_size_pre
        X = np.split(X, num_of_batches, 0)
        X_corrupt = np.split(X_corrupt, num_of_batches, 0)

        # Train the Model
        for epoch in range(1, self.num_epochs_pre+1):
            X, X_corrupt = shuffle(X, X_corrupt) # shuffle batches
            loss_all = [] # for saving the loss in each step
            scheduler.step() # adjust learning rate
            # Loop through all batches
            for x, x_corrupt in zip(X, X_corrupt):
                x = torch.FloatTensor(x)
                x_corrupt = torch.FloatTensor(x_corrupt)
                if self.use_cuda:
                    x = x.cuda()
                    x_corrupt = x_corrupt.cuda()
                x = Variable(x)
                x_corrupt = Variable(x_corrupt)
                optimizer.zero_grad() # reset gradient
                outputs = self.model(x_corrupt, pre_train=True) # forward propagation
                loss = criterion(outputs, x) # compute loss
                loss.backward() # backward propagation
                optimizer.step() # optimize
                loss_all.append(loss.data[0]) # save loss for each step
            # Print the result for the entire epoch
            T_tr, P_tr = self.train["X"], self.predict(self.train["X_corrupt"], pre_train=True)
            m_train = computeMetric(T_tr, P_tr, True, flatten=True, simple=True, round_to_decimal=2, no_prf=True)
            if self.test is not None:
                T_te, P_te = self.test["X"], self.predict(self.test["X"], pre_train=True)
                m_test = computeMetric(T_te, P_te, True, flatten=True, simple=True, round_to_decimal=2, no_prf=True)
            lr_now = optimizer.state_dict()["param_groups"][0]["lr"]
            avg_loss = np.mean(loss_all)
            if self.test is not None:
                self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [mse, r2], [%2f, %2f], [%2f, %2f]'
                    %(epoch, self.num_epochs_pre, lr_now, avg_loss, m_train["mse"], m_train["r2"],
                    m_test["mse"], m_test["r2"]))
            else:
                self.log('[%2d/%d], LR: %.8f, Loss: %.8f, [mse, r2], [%5d, %5d]'
                    %(epoch, self.num_epochs_pre, lr_now, avg_loss, m_train["mse"], m_train["r2"]))

        self.log("--------------------------------------------------------------")
        self.log("From " + str(start_time) + " to " + str(datetime.now()))
        self.log("==============================================================")
        return self
    
    # Crop the autoencoder and train a classifier or regressor
    def fine_tune(self, X, Y):
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
            #criterion = nn.SmoothL1Loss()
            criterion = nn.MSELoss()
            if self.use_class_weights:
                self.log("Regression will ignore class weights")
        else:
            criterion = nn.CrossEntropyLoss()
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
            m_train = computeMetric(T_tr, P_tr, self.is_regr, flatten=True, simple=True, aggr_axis=True, no_prf=True)
            if self.test is not None:
                T_te, P_te = self.test["Y"], self.predict(self.test["X"])
                m_test = computeMetric(T_te, P_te, self.is_regr, flatten=True, simple=True, aggr_axis=True, no_prf=True)
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
        self.log("From " + str(start_time) + " to " + str(datetime.now()))
        self.log("--------------------------------------------------------------")
        return self

    def predict(self, X, threshold=0.6, pre_train=False):
        if self.is_regr or pre_train:
            Y_pred = self.forward_from_numpy(X, pre_train=pre_train)
            Y_pred = Y_pred.data.numpy().squeeze()
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

    def forward_from_numpy(self, X, pre_train=False):
        X = torch.FloatTensor(X)
        if self.use_cuda: X = X.cuda()
        X = Variable(X)
        outputs = self.model(X, pre_train=pre_train)
        return outputs.cpu()

    def save(self, out_path):
        torch.save(self.model.state_dict(), out_path)

    def load(self, in_path):
        self.model.load_state_dict(torch.load(in_path))

    def log(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.01)
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.01)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.01)

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
            #nn.ReplicationPad2d((0, 0, p_size_b2, p_size_b2)),
            nn.Conv2d(h_size_b2, h_size_b2, kernel_size=(k_size_b2, 1), padding=(p_size_b2, 0), bias=False),
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

# Residual Neural Network
class ResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResNet, self).__init__()

        out_size_c1 = 32
        self.c = nn.Sequential(
            nn.SELU(),
            #nn.ReplicationPad2d((0, 0, 1, 1)),
            nn.Conv2d(input_size, out_size_c1, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1), bias=False))
        self.r1 = ResLargeBlock(out_size_c1, output_size, h_size=16, s_size=1)

    def forward(self, x):
        f = self.c(x)
        f = self.r1(f)
        return f

# Convolution Neural Network (encoder)
class CnnEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_size_2):
        super(CnnEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.SELU(),
            nn.Conv2d(input_size, hidden_size, kernel_size=(3,1), padding=(1,0), stride=(1,1), bias=False),
            nn.SELU(),
            nn.Conv2d(hidden_size, hidden_size_2, kernel_size=(3,1), padding=(1,0), stride=(1,1), bias=False),
            nn.SELU(),
            nn.Conv2d(hidden_size_2, output_size, kernel_size=(3,1), padding=(1,0), stride=(1,1), bias=False))

    def forward(self, x):
        return self.conv(x)

# Convolution Neural Network (decoder)
class CnnDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_size_2):
        super(CnnDecoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.SELU(),
            nn.ConvTranspose2d(output_size, hidden_size_2, kernel_size=(3, 1), padding=(1,0), stride=(1, 1), bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(hidden_size_2, hidden_size, kernel_size=(3, 1), padding=(1,0), stride=(1, 1), bias=False),
            nn.SELU(),
            nn.ConvTranspose2d(hidden_size, input_size, kernel_size=(3, 1), padding=(1,0), stride=(1, 1), bias=False))

    def forward(self, x):
        return self.conv(x)

# Fully Connnected Block
class FC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_size_2):
        super(FC, self).__init__()
        
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

# Autoencoder CNN Model (1D convolution on height of the image)
class ANCNN(nn.Module):
    def __init__(self, channel_size, height, width, output_size, use_cuda=True):
        super(ANCNN, self).__init__()
        
        # Hyper-parameters
        self.width = width
        self.height = height
        self.channel_size = channel_size

        # CNN Encoder (Feature Extraction)
        hidden_cnn = 64
        hidden2_cnn = 128
        output_cnn = 256
        self.encoder = CnnEncoder(channel_size, output_cnn, hidden_cnn, hidden2_cnn)
       
        # CNN Decoder
        self.decoder = CnnDecoder(channel_size, output_cnn, hidden_cnn, hidden2_cnn)

        # Fully Connected
        hidden_fc = 128
        hidden_fc_2 = 64
        input_fc = self.fullyConnectedLayerInputSize()
        self.fc = FC(input_fc, output_size, hidden_fc, hidden_fc_2)

    def forward(self, x, pre_train=False):
        f = self.encoder(x)
        if pre_train:
            f = self.decoder(f)
        else:
            f = self.toFcInput(f)
            f = self.fc(f)
        return f

    def toFcInput(self, x):
        f = x.contiguous().view(x.size(0), -1)
        return f

    def fullyConnectedLayerInputSize(self):
        x = self.genDummyInput()
        f = self.encoder(x)
        print f.size()
        f = self.toFcInput(f)
        print f.size()
        return f.size(1)

    def genDummyInput(self):
        return Variable(torch.randn(1, self.channel_size, self.height, self.width))

    def freezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
