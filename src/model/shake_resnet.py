import torch
import torch.nn as nn
from modules.backbone_resnet import ShakeResNetBackbone
from layers.pool_fc_block import PoolFCBlock  

class ShakeResNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.backbone = ShakeResNetBackbone(in_ch=in_ch)
        self.pool_fc = PoolFCBlock(self.backbone.stage_channels[-1], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool_fc(x)
        return x
