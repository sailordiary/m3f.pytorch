# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet, ResNetV2, BasicBlock, BasicBlockV2
from .densenet import DenseNet52_3D
from .vggface import VGGFace
from .rnn import GRU


class VA_VGGFace(nn.Module):
    def __init__(self, inputDim=4096, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru'):
        super(VA_VGGFace, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        
        self.vgg = VGGFace()

        # backend
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        b = x.size(0)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.vgg(x)
        x = x.view(b, -1, x.size(1))
        if self.backend == 'gru':
            x = self.gru(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class VA_3DVGGM(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', norm_layer='bn'):
        super(VA_3DVGGM, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        
        self.v2p = nn.Sequential(
            # conv1 + pool1
            nn.Conv3d(3, 64, 3, stride=(1,2,2), padding=(1,0,0)),
            nn.BatchNorm3d(64) if norm_layer == 'bn' else nn.GroupNorm(32, 64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            # conv2 + pool2
            nn.Conv3d(64, 128, 3, 1, padding=(1,0,0)),
            nn.BatchNorm3d(128) if norm_layer == 'bn' else nn.GroupNorm(32, 128),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            # conv3 + pool3
            nn.Conv3d(128, 256, 3, 1, (1,0,0)),
            nn.BatchNorm3d(256) if norm_layer == 'bn' else nn.GroupNorm(32, 256),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            # conv4
            nn.Conv3d(256, 512, 3, 1, (1,0,0)),
            nn.BatchNorm3d(512) if norm_layer == 'bn' else nn.GroupNorm(32, 512),
            nn.ReLU(True),

            # conv5 + pool5
            nn.Conv3d(512, 512, 3, 1, (1,0,0)),
            nn.BatchNorm3d(512) if norm_layer == 'bn' else nn.GroupNorm(32, 512),
            nn.ReLU(True),
            # NOTE: no pool5 for 112*112 input
            # nn.MaxPool3d(kernel_size=(1,2,2), stride=1)
        )
        # backend
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.v2p(x)
        x = x.squeeze().transpose(1, 2)
        if self.backend == 'gru':
            x = self.gru(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VA_3DResNet(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', use_cbam=False, resnet_ver='v2', resnet_depth=18, frontend_agg_mode='ap'):
        super(VA_3DResNet, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        
        final_fmap_size = 3
        
        self.c3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        assert resnet_depth in [18, 34] and resnet_ver in ['v1', 'v2'], 'unsupported ResNet configuration: {}, {}'.format(resnet_depth, resnet_ver)
        block_config = [2, 2, 2, 2] if resnet_depth == 18 else [3, 4, 6, 3]
        if resnet_ver == 'v2':
            self.resnet = ResNetV2(BasicBlockV2, block_config, self.inputDim, zero_init_residual=False, agg_mode=frontend_agg_mode, fmap_out_size=final_fmap_size, use_cbam=use_cbam)
        else:
            self.resnet = ResNet(BasicBlock, block_config, self.inputDim, zero_init_residual=True, agg_mode=frontend_agg_mode, fmap_out_size=final_fmap_size, use_cbam=use_cbam)
        # backend
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.c3d(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(-1, self.frameLen, self.inputDim)
        if self.backend == 'gru':
            x = self.gru(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VA_3DDenseNet(nn.Module):
    def __init__(self, inputDim=392, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', frontend_agg_mode='ap'):
        super(VA_3DDenseNet, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        
        final_fmap_size = 3
        
        self.c3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.densenet = DenseNet52_3D(self.inputDim, agg_mode=frontend_agg_mode, fmap_out_size=final_fmap_size)
        # backend
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.c3d(x)
        x = self.densenet(x)
        if self.backend == 'gru':
            x = self.gru(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
