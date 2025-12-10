import torch
import torch.nn as nn

class PoolFCBlock(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
