import torch.nn as nn

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        return self.conv(x)
