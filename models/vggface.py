import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFace(nn.Module):
    def __init__(self):
        """VGGFace model, assuming 112x112 input.
        """
        super().__init__()
        self.conv1 = _ConvBlock(3, 64, 64)
        self.conv2 = _ConvBlock(64, 128, 128)
        self.conv3 = _ConvBlock(128, 256, 256, 256)
        self.conv4 = _ConvBlock(256, 512, 512, 512)
        self.conv5 = _ConvBlock(512, 512, 512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * 512, 4096)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return x


class _ConvBlock(nn.Module):
    """A Convolutional block."""

    def __init__(self, *units):
        """Create a block with len(units) - 1 convolutions.
        convolution number i transforms the number of channels from 
        units[i - 1] to units[i] channels.
        """
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_, out, 3, 1, 1)
            for in_, out in zip(units[:-1], units[1:])
        ])
        
    def forward(self, x):
        # Each convolution is followed by a ReLU, then the block is
        # concluded by a max pooling.
        for c in self.convs:
            x = F.relu(c(x))
        return F.max_pool2d(x, 2, 2, 0, ceil_mode=True)
