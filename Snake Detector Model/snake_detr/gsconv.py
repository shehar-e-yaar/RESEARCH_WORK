import torch
import torch.nn as nn

class GSConv(nn.Module):
    """
    Group-Shuffle Convolution (GSConv)
    As described in Section 'GSConv' and Figure 10 (Page 10) of the paper.
    Combines the parameter-efficiency of Depthwise Separable Convolutions (DSC)
    with the feature-representation strength of standard convolutions through channel shuffle.
    """
    def __init__(self, c1, c2, k=1, s=1):
        super(GSConv, self).__init__()
        c_mid = c2 // 2
        
        # Branch 1: Standard 1x1 Convolution to reduce channels to C2/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU()
        )
        
        # Branch 2: 5x5 Depthwise Convolution
        self.dw_conv = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=5, stride=1, padding=2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU()
        )

    def forward(self, x):
        # 1. Standard conv
        x1 = self.conv1(x)
        # 2. Depthwise conv on branch 1
        x2 = self.dw_conv(x1)
        
        # 3. Concatenate along channel dimension
        out = torch.cat([x1, x2], dim=1)
        
        # 4. Channel Shuffle across groups
        # Reshape to (B, 2, C2/2, H, W) -> Transpose to (B, C2/2, 2, H, W) -> Flatten to (B, C2, H, W)
        b, c, h, w = out.shape
        out = out.view(b, 2, c // 2, h, w)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, c, h, w)
        
        return out
