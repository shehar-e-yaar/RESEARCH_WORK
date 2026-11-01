import torch
import torch.nn as nn
from .caa import ContextAnchorAttention

class CAAGELAN(nn.Module):
    """
    CAA-GELAN Module: Enhanced Generalized Efficient Layer Aggregation Network 
    Based on Context Anchor Attention.
    As shown in Figure 5 (Page 7) of the Snake-DETR paper.
    
    Structure:
      - Initial 1x1 Conv -> Split/Chunk into 2 branches
      - Multiple gradient propagation stages through RepNCSP / Convs
      - Concatenation of intermediate layers
      - Feature enhancement using the Context Anchor Attention (CAA) module
      - Final 1x1 Conv
    """
    def __init__(self, c1, c2, n=1):
        super(CAAGELAN, self).__init__()
        c_half = c2 // 2
        
        # Initial projection
        self.conv_in = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # Stage 1 block
        self.block1 = nn.Sequential(
            nn.Conv2d(c_half, c_half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )
        
        # Stage 2 block
        self.block2 = nn.Sequential(
            nn.Conv2d(c_half, c_half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )
        
        # Context Anchor Attention (CAA) enhancement
        total_channels = c_half * 3  # concatenation of identity branch + block1 + block2
        self.caa = ContextAnchorAttention(total_channels)
        
        # Final projection to C2
        self.conv_out = nn.Sequential(
            nn.Conv2d(total_channels, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        feat = self.conv_in(x)
        x_branch1, x_branch2 = torch.chunk(feat, 2, dim=1)
        
        y1 = self.block1(x_branch2)
        y2 = self.block2(y1)
        
        # Concatenate branches (GELAN gradient path aggregation)
        merged = torch.cat([x_branch1, y1, y2], dim=1)
        
        # Context Anchor Attention enhancement
        enhanced = self.caa(merged)
        
        # Final output
        return self.conv_out(enhanced)
