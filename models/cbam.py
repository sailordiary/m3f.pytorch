# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        
        return x


class ChannelGate(nn.Module):
    '''
    Given a feature map of shape (B, C, H, W):
    - Compute MaxPool and AvgPool of shape (B, C, 1)
    - MLP (B, C/ratio, 1) - (B, C, 1)
    - Sum activations (B, C, 1)
    - Broadcast to original size
    '''
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    
    def forward(self, x):
        avg_pool = x.mean(-1).mean(-1)
        max_pool = x.max(-1)[0].max(-1)[0]
        
        channel_att_sum = self.mlp(avg_pool) + self.mlp(max_pool)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale

    
class ChannelPool(nn.Module):
    '''
    Given a channel refined feature:
    - Compute MaxPool and AvgPool feature maps across channels
    - Stack into a 2-channel feature map
    '''
    def forward(self, x):
        channel_max = torch.max(x, 1)[0].unsqueeze(1) # (B, 1, T, H, W)
        channel_mean = torch.mean(x, 1).unsqueeze(1)
        
        return torch.cat((channel_max, channel_mean), dim=1) # (B, 2, T, H, W)


class SpatialGate(nn.Module):
    '''
    Given a channel refined feature:
    - Compute channel-pooled feature map
    - Apply (in=2, out=1, kernel=5) convolution
    - Apply sigmoid to obtain spatial attention
    '''
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        
        return x * scale # point-wise product


class CBAM(nn.Module):
    '''
    Given an input feature map of shape (B, C, H, W):
    - Apply channel attention 
    - Apply spatial attention 
    '''
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    
    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        
        return x

