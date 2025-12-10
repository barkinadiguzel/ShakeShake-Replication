# src/modules/backbone_resnet.py

import torch.nn as nn
from layers.residual_block import ShakeResidualBlock
from layers.downsample_block import DownsampleBlock
from layers.conv_block import ConvBlock

class ShakeResNetBackbone(nn.Module):
    def __init__(self, in_ch=3, base_ch=16, num_blocks=[4,4,4]):
        super().__init__()
        self.stem = ConvBlock(in_ch, base_ch)
        self.stage_channels = [base_ch, base_ch*2, base_ch*4]

        self.stages = nn.ModuleList()
        in_channels = base_ch
        for i, blocks in enumerate(num_blocks):
            stage = []
            out_channels = self.stage_channels[i]
            for j in range(blocks):
                if j == 0 and in_channels != out_channels:
                    down = DownsampleBlock(in_channels, out_channels)
                else:
                    down = None
                stage.append(ShakeResidualBlock(in_channels, out_channels, downsample=down))
                in_channels = out_channels
            self.stages.append(nn.Sequential(*stage))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        return x
