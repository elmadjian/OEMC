import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CNN_LSTM(nn.Module):

    def __init__(self, input_size, output_size, kernel_size, dropout, 
                 features, lstm_layers, conv_filters=(32, 16, 8)):
        super(CNN_LSTM, self).__init__()
        self.conv_filters = conv_filters
        conv_layers = []
        padding = int(np.floor((kernel_size-1)/2))
        for i, filter in enumerate(self.conv_filters):
            input_conv = input_size if i == 0 else conv_filters[i-1]
            if i > 0:
                conv_layers += [nn.Dropout(dropout)]
            conv_layers += [nn.Conv1d(input_conv, conv_filters[i], kernel_size,
                                      padding=padding, padding_mode='replicate')]
            conv_layers += [nn.BatchNorm1d(conv_filters[i])]
            conv_layers += [nn.ReLU()]        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = TimeDistributed(nn.Flatten())
        self.blstm = nn.LSTM(input_size=features, bidirectional=False,
                             hidden_size=32, num_layers=lstm_layers)
        linear = nn.Linear(32, output_size)
        linear.weight.data.normal_(0, 0.01)
        self.output = TimeDistributed(linear)


    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.flatten(out)
        out,_ = self.blstm(out)
        out = self.output(out[:,-1,:])
        out = F.log_softmax(out, dim=1)
        return out


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0),-1,y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y

        
