#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 2021

@author: byeongjokim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .r2plus1d import r2plus1d_34_32_ig65m, r2plus1d_34_32_kinetics, flow_r2plus1d_34_32_ig65m
from .BERT.bert import  BERT5

__all__ = [ 'rgb_Depth_r2plus1d_64f_34_bert10', 'rgb_Depth_concat_r2plus1d_64f_34_bert10']

def make_features(modelPath):
    features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=modelPath, progress=True).children())[:-2])

    return features

class rgb_Depth_r2plus1d_64f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_Depth_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.rgb_avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.depth_avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        
        self.rgb_features = make_features(modelPath)
        self.depth_features = make_features(modelPath)

        self.fc_input = nn.Linear(2 * self.hidden_size, self.hidden_size)
                
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
      
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        for param in self.depth_features.parameters():
            param.requires_grad = True
  
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x1, x2):
        #x1 -> rgb
        #x2 -> depth

        x1 = self.rgb_features(x1)
        x1 = self.rgb_avgpool(x1)
        x1 = x1.view(x1.size(0), self.hidden_size, 8)
        x1 = x1.transpose(1,2)

        x2 = self.depth_features(x2)
        x2 = self.depth_avgpool(x2)
        x2 = x2.view(x2.size(0), self.hidden_size, 8)
        x2 = x2.transpose(1,2)

        x = torch.cat([x1, x2], -1)
        
        x = self.fc_input(x)
        
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), self.hidden_size, 4)
        # x = x.transpose(1,2)

        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample

class rgb_Depth_concat_r2plus1d_64f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_Depth_concat_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.rgb_avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.depth_avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        
        self.rgb_features = make_features(modelPath)
        self.depth_features = make_features(modelPath)

        self.fc_rgb = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.fc_d = nn.Linear(self.hidden_size, int(self.hidden_size/2))
                
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
      
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        for param in self.depth_features.parameters():
            param.requires_grad = True
  
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x1, x2):
        #x1 -> rgb
        #x2 -> depth

        x1 = self.rgb_features(x1)
        x1 = self.rgb_avgpool(x1)
        x1 = x1.view(x1.size(0), self.hidden_size, 8)
        x1 = x1.transpose(1,2)
        x1 = self.fc_rgb(x1)

        x2 = self.depth_features(x2)
        x2 = self.depth_avgpool(x2)
        x2 = x2.view(x2.size(0), self.hidden_size, 8)
        x2 = x2.transpose(1,2)
        x2 = self.fc_d(x2)

        x = torch.cat([x1, x2], -1)

        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample

