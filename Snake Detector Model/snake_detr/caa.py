import torch
import torch.nn as nn

class ContextAnchorAttention(nn.Module):
    """
    Context Anchor Attention (CAA) Module
    Exact implementation of Equations (1) through (5) from Page 7 of Wang et al. (2025):
    
      Eq. (1): F_pool = Conv_1x1( P_avg( X ) )
      Eq. (2): F_w    = DWConv_{1 x kb}( F_pool )
      Eq. (3): F_h    = DWConv_{kb x 1}( F_w )
      Eq. (4): A      = Sigmoid( Conv_1x1( F_h ) )
      Eq. (5): F_attn = (A ⊙ P) ⊕ P
      
    Where:
      - P_avg: Average pooling operation condensing local spatial regions.
      - DWConv_{1 x kb} & DWConv_{kb x 1}: Depthwise-separable convolutions capturing
        long-range horizontal and vertical contextual interdependencies.
      - A: Attention weight map in range (0, 1).
      - ⊙: Element-wise multiplication.
      - ⊕: Element-wise addition (residual context anchor).
    """
    def __init__(self, channels, kb=11, pool_kernel=3):
        super(ContextAnchorAttention, self).__init__()
        self.channels = channels
        self.kb = kb
        
        # Eq. (1): Local average pooling P_avg + Conv_1x1
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU()
        )

        # Eq. (2): DWConv_{1 x kb} (Width-wise strip convolution)
        self.dw_conv_w = nn.Sequential(
            nn.Conv2d(
                channels // 2, channels // 2,
                kernel_size=(1, kb),
                padding=(0, kb // 2),
                groups=channels // 2,
                bias=False
            ),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU()
        )

        # Eq. (3): DWConv_{kb x 1} (Height-wise strip convolution)
        self.dw_conv_h = nn.Sequential(
            nn.Conv2d(
                channels // 2, channels // 2,
                kernel_size=(kb, 1),
                padding=(kb // 2, 0),
                groups=channels // 2,
                bias=False
            ),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU()
        )

        # Eq. (4): Conv_1x1 + Sigmoid
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input tensor P
        identity = x
        
        # Eq. (1): Condense local features with pooling then 1x1 conv
        f_pool = self.conv1(self.avg_pool(x))
        
        # Eq. (2): Horizontal contextual interaction
        f_w = self.dw_conv_w(f_pool)
        
        # Eq. (3): Vertical contextual interaction
        f_h = self.dw_conv_h(f_w)
        
        # Eq. (4): Attention weight matrix A ∈ (0, 1)
        A = self.sigmoid(self.conv2(f_h))
        
        # Eq. (5): F_attn = (A ⊙ P) ⊕ P
        f_attn = (A * identity) + identity
        
        return f_attn
