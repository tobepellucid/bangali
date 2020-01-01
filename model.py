# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:15:43 2019

@author: 97555
"""
import torch
import torch.nn as nn
import torchsummary
import torchvision
import torch.nn.functional as F
from torch.nn import Sequential
import pretrainedmodels
import torchvision.models as models

class net(nn.Module):
    def __init__(self,n_classes=[168,11,7],use_bn=True): 
        super(net, self).__init__()
        self.conv0 = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
        self.res18 = torchvision.models.resnet34(pretrained=True)
        self.resnet_layer = nn.Sequential(*list(self.res18.children())[:-1])
        activation = F.leaky_relu
        inch = 512
        #hdim = 256
        self.head1 = LinearBlock(inch, n_classes[0], use_bn=use_bn, activation=activation, residual=False)
        self.head2 = LinearBlock(inch, n_classes[1], use_bn=use_bn, activation=activation, residual=False)
        self.head3 = LinearBlock(inch, n_classes[2], use_bn=use_bn, activation=activation, residual=False)
        #self.lin_layers = Sequential(lin1, lin2)
        #self.fc1 = nn.Linear(512,256)
        
        #self.fc2 = nn.Linear(256,n_classes)
    def forward(self,x):
        x_ = self.conv0(x)
        feat = self.resnet_layer(x_)
        h = feat.view(feat.size(0),-1)
        #for layer in self.lin_layers:
        #    h = layer(h)
        #feat1 = self.fc1(feat)
        #feat2 = self.fc2(feat1)
        x1 = self.head1(h)
        x2 = self.head2(h)
        x3 = self.head3(h)
        return x1,x2,x3

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual
    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

#model = net(n_classes=168)
#torchsummary(model,(1,128,128))
#x = torch.Tensor(1,3,128,128)
#y = model(x)
#print(y.shape)