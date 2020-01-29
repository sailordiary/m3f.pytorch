import torch
import torch.nn as nn


class _DenseLayer_3D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer_3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('dp', nn.Dropout3d(p=drop_rate))

    def forward(self, x):
        new_features = super(_DenseLayer_3D, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock_3D(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock_3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_3D(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition_3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pooling=True):
        super(_Transition_3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if pooling:
            self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))


class DenseNet52_3D(nn.Module):
    def __init__(self, num_classes=256, num_init_features=64, bn_size=4, block_config=(4, 6, 8, 6), growth_rate=32, dp=0., agg_mode='ap', fmap_out_size=3):
        super(DenseNet52_3D, self).__init__()
        num_features = num_init_features
        self.agg_mode = agg_mode
        
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock_3D(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=dp)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                if len(block_config) > 4 and i == 1:
                    trans = _Transition_3D(num_input_features=num_features, num_output_features=num_features // 2, pooling=False)
                else:
                    trans = _Transition_3D(num_input_features=num_features, num_output_features=num_features // 2, pooling=True)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU())
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features * fmap_out_size * fmap_out_size, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)

        if self.agg_mode == 'ap':
            b, c, t = x.size(0), x.size(1), x.size(2)
            x = x.view(-1, x.size(3), x.size(4))
            x = self.avgpool(x)
            x = x.view(b, c, t)
            x = x.transpose(1, 2).contiguous()
        elif self.agg_mode == 'fc':
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.size(0), x.size(1), -1)
            x = self.fc(x)
        
        return x
