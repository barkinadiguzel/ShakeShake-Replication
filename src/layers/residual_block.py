import torch
import torch.nn as nn
from .conv_block import ConvBlock
from .shake_layer import shake_forward
from .downsample_block import DownsampleBlock

class ShakeResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=None):
        super().__init__()
        self.branch1 = ConvBlock(in_ch, out_ch)
        self.branch2 = ConvBlock(in_ch, out_ch)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = shake_forward(out1, out2, self.training)
        return shortcut + out
