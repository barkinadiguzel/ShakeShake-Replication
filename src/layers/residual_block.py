import torch.nn as nn
from .conv_block import ConvBNReLU
from .shake_layer import shake_shake


class ShakeBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch1 = ConvBNReLU(in_c, in_c)
        self.branch2 = ConvBNReLU(in_c, in_c)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = shake_shake(x1, x2, self.training)
        return x + out
