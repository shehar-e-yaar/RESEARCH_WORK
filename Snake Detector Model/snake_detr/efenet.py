import torch
import torch.nn as nn
from .gelan import CAAGELAN

class ADown(nn.Module):
    """
    ADown: Asymmetric Downsampling Block
    As depicted in Figure 9 (Page 9) of the Snake-DETR paper.
    Splits input into two halves:
      - Half 1: Average pooling -> Conv
      - Half 2: Max pooling -> Conv
    Concatenates the results to retain rich spatial and semantic downsampled features.
    """
    def __init__(self, c1, c2):
        super(ADown, self).__init__()
        c_half = c2 // 2
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv_avg = nn.Sequential(
            nn.Conv2d(c1 // 2, c_half, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )
        self.conv_max = nn.Sequential(
            nn.Conv2d(c1 // 2, c_half, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

    def forward(self, x):
        # Chunk channels into two halves
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Branch 1: AvgPool then Conv (stride 2)
        out1 = self.conv_avg(self.avg_pool(x1))
        
        # Branch 2: MaxPool (stride 2) then Conv
        out2 = self.conv_max(self.max_pool(x2))
        
        # Concatenate both pooled halves
        return torch.cat([out1, out2], dim=1)


class EFENet(nn.Module):
    """
    EFENet: Enhanced Feature Extraction Network Based on Context Anchor Attention
    The lightweight custom backbone network from Section EFENet and Figure 8 (Page 9).
    Outputs the three hierarchical feature stages {S3, S4, S5} for the RT-DETR hybrid encoder.
    """
    def __init__(self, in_channels=3):
        super(EFENet, self).__init__()
        
        # Stage 0: Initial Stem Convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        # Stage 1 (S3): 128 channels
        self.down1 = ADown(64, 128)
        self.stage1 = CAAGELAN(128, 128)
        
        # Stage 2 (S4): 256 channels
        self.down2 = ADown(128, 256)
        self.stage2 = CAAGELAN(256, 256)
        
        # Stage 3 (S5): 512 channels
        self.down3 = ADown(256, 512)
        self.stage3 = CAAGELAN(512, 512)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stage S3
        x = self.down1(x)
        s3 = self.stage1(x)
        
        # Stage S4
        x = self.down2(s3)
        s4 = self.stage2(x)
        
        # Stage S5
        x = self.down3(s4)
        s5 = self.stage3(x)
        
        # Return multi-scale feature tuple {S3, S4, S5} for hybrid encoder
        return s3, s4, s5
