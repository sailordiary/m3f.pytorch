import torch
import torch.nn as nn

from .cbam import CBAM


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=256, zero_init_residual=True, agg_mode='ap', fmap_out_size=3, use_cbam=False):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.agg_mode = agg_mode
        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=use_cbam)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * fmap_out_size * fmap_out_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # XXX: CBAM initialization
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if 'bn' in key and 'SpatialGate' in key:
                    self.state_dict()[key][...] = 0

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.agg_mode == 'ap':
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        if self.agg_mode == 'fc':
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 is_first_block_of_first_layer=False, use_cbam=False):
        super(BasicBlockV2, self).__init__()
        self.is_first_block_of_first_layer = is_first_block_of_first_layer
        if not is_first_block_of_first_layer:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride
        
        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        out = x
        if not self.is_first_block_of_first_layer:
            out = self.bn1(x)
            out = self.relu(out)
        identity = self.downsample(out) if self.downsample is not None else x
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.cbam is not None:
            out = self.cbam(out)

        return out + identity


class ResNetV2(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    Args:
        block (Module) : class for the residual block. Options are BasicBlockV1, BottleneckV1.
        layers (list of int) : numbers of layers in each block
        num_classes (int) :, default 1000, number of classification classes.
    """
    def __init__(self, block, layers, num_classes=256, zero_init_residual=False, agg_mode='ap', fmap_out_size=3, use_cbam=False):
        super(ResNetV2, self).__init__()
        self.inplanes = 64
        self.agg_mode = agg_mode

        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=use_cbam)

        self.bn5 = nn.BatchNorm2d(self.inplanes)
        self.relu5 = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * fmap_out_size * fmap_out_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # XXX: CBAM initialization
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if 'bn' in key and 'SpatialGate' in key:
                    self.state_dict()[key][...] = 0

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockV2):
                    nn.init.constant_(m.bn1.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)

        is_first_block_of_first_layer = stride == 1
        layers = [block(self.inplanes, planes, stride,
                        downsample, is_first_block_of_first_layer, use_cbam=use_cbam)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn5(x)
        x = self.relu5(x)

        if self.agg_mode == 'ap':
            x = self.avgpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.agg_mode == 'fc':
            x = self.fc(x)
        
        return x

