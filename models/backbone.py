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
from .tcn import TemporalConvNet


class VA_VGGFace(nn.Module):
    def __init__(self, inputDim=4096, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', nFCs=1):
        super(VA_VGGFace, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        self.nFCs = nFCs
        
        self.vgg = VGGFace()

        # backend
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.nFCs)

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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VA_3DVGGM(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', norm_layer='bn', nFCs=1):
        super(VA_3DVGGM, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        self.nFCs = nFCs
        
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
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.nFCs)
        elif self.backend == 'tcn':
            self.tcn = nn.ModuleList([
                TemporalConvNet(self.inputDim, [self.hiddenDim] * self.nLayers, 3),
                nn.Linear(self.hiddenDim, 2)
            ])
        elif self.backend == 'tcn_simple':
            self.tcn = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(self.inputDim, self.hiddenDim, 3, 1, 1),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Conv1d(self.hiddenDim, self.hiddenDim, 3, 1, 1),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                ),
                nn.Linear(self.hiddenDim, 2)
            ])
        elif self.backend == 'fc':
            self.fc = nn.Sequential(
                nn.Linear(512, self.hiddenDim),
                nn.ReLU(True),
                nn.Linear(self.hiddenDim, self.nClasses)
            )

        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.v2p(x).squeeze()
        if self.backend == 'gru':
            x = x.transpose(1, 2)
            x = self.gru(x)
        elif self.backend.startswith('tcn'):
            x = self.tcn[0](x).transpose(1, 2).contiguous()
            x = self.tcn[1](x)
        elif self.backend == 'fc':
            x = torch.mean(x, dim=-1)
            x = self.fc(x)
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


class VA_3DVGGM_Split(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=512, nLayers=2, frameLen=16, nClasses=2, backend='gru', norm_layer='bn', split_layer=5, nFCs=1, use_mtl=False):
        super(VA_3DVGGM_Split, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.nClasses = nClasses
        self.backend = backend
        self.split_layer = split_layer
        self.norm_layer = norm_layer
        self.nFCs = nFCs
        self.use_mtl = use_mtl

        assert split_layer >= 2, 'degenerate multi-tower structure'
        shared = [
            nn.Conv3d(3, 64, 3, stride=(1,2,2), padding=(1,0,0)),
            nn.BatchNorm3d(64) if norm_layer == 'bn' else nn.GroupNorm(32, 64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        ]
        if self.split_layer != 5: v_private, a_private = [], []
    
        for i in range(2, 6):
            if split_layer >= i: getattr(self, 'add_conv{}'.format(i))(shared)
            else:
                getattr(self, 'add_conv{}'.format(i))(v_private)
                getattr(self, 'add_conv{}'.format(i))(a_private)
        self.shared = nn.Sequential(*shared)
        if self.split_layer != 5:
            self.v_private = nn.Sequential(*v_private)
            self.a_private = nn.Sequential(*a_private)
       
        # backend
        if self.backend == 'gru':
            if split_layer == 5:
                self.gru = GRU(self.inputDim + 512 + 256, self.hiddenDim, self.nLayers, self.nClasses, self.nFCs)
            else:
                if self.use_mtl:
                    self.gru_v = GRU(self.inputDim + 512, self.hiddenDim, self.nLayers, 7 + 1, self.nFCs)
                    self.gru_a = GRU(self.inputDim + 256, self.hiddenDim, self.nLayers, 8 + 1, self.nFCs)
                else:
                    self.gru_v = GRU(self.inputDim + 512, self.hiddenDim, self.nLayers, 1, self.nFCs)
                    self.gru_a = GRU(self.inputDim + 256, self.hiddenDim, self.nLayers, 1, self.nFCs)

        # initialize
        self._initialize_weights()
    
    def add_conv2(self, x):
        x.extend([
            nn.Conv3d(64, 128, 3, 1, padding=(1,0,0)),
            nn.BatchNorm3d(128) if self.norm_layer == 'bn' else nn.GroupNorm(32, 128),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        ])
    
    def add_conv3(self, x):
        x.extend([
            nn.Conv3d(128, 256, 3, 1, (1,0,0)),
            nn.BatchNorm3d(256) if self.norm_layer == 'bn' else nn.GroupNorm(32, 256),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        ])
    
    def add_conv4(self, x):
        x.extend([
            nn.Conv3d(256, 512, 3, 1, (1,0,0)),
            nn.BatchNorm3d(512) if self.norm_layer == 'bn' else nn.GroupNorm(32, 512),
            nn.ReLU(True)
        ])
    
    def add_conv5(self, x):
        x.extend([
            nn.Conv3d(512, 512, 3, 1, (1,0,0)),
            nn.BatchNorm3d(512) if self.norm_layer == 'bn' else nn.GroupNorm(32, 512),
            nn.ReLU(True)
        ])

    def forward(self, x, se, au):
        x = self.shared(x)
        if self.split_layer != 5:
            x_v = self.v_private(x).squeeze()
            x_a = self.a_private(x).squeeze()
            x_v = torch.cat((x_v, se), dim=1) # valence / SENet
            x_a = torch.cat((x_a, au), dim=1) # arousal / TCAE-AU
            if self.backend == 'gru':
                x_v = self.gru_v(x_v.transpose(1, 2))
                x_a = self.gru_a(x_a.transpose(1, 2))
                return torch.cat((x_v, x_a), dim=-1)
        else:
            x = x.squeeze()
            x = torch.cat((x, se, au), dim=1)
            if self.backend == 'gru':
                x = self.gru(x.transpose(1, 2))
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
    def __init__(self, inputDim=512, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', use_cbam=False, resnet_ver='v2', resnet_depth=18, frontend_agg_mode='ap', nFCs=1):
        super(VA_3DResNet, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        self.nFCs = nFCs
        
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
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.nFCs)

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
    def __init__(self, inputDim=392, hiddenDim=512, nLayers=2, nClasses=2, frameLen=16, backend='gru', frontend_agg_mode='ap', nFCs=1):
        super(VA_3DDenseNet, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        self.nFCs = nFCs
        
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
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.nFCs)

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
