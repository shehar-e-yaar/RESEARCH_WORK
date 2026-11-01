import torch
import torch.nn as nn
from .efenet import EFENet
from .gsconv import GSConv
from .gelan import CAAGELAN
from .piou_loss import PowerfulIoULoss

class SnakeDETR(nn.Module):
    """
    Snake-DETR: Complete End-to-End Object Detection Architecture
    As formulated in the paper (Figure 13, Page 12):
    
    1. Backbone: EFENet
       Extracts multi-scale features {S3, S4, S5} from input image.
    2. Lightweight Hybrid Neck:
       Fuses multi-scale features using Group-Shuffle Convolutions (GSConv)
       and Context Anchor GELAN (CAA-GELAN).
    3. Detection Heads:
       Produces bounding box regressions and class predictions (num_classes).
    4. Loss Function:
       Integrated Powerful-IoU (PIoU) loss for accurate box regression.
    """
    def __init__(self, num_classes=2):
        super(SnakeDETR, self).__init__()
        self.num_classes = num_classes
        
        # 1. Custom Backbone (EFENet)
        self.backbone = EFENet(in_channels=3)
        
        # 2. Hybrid Encoder Neck with GSConv & CAA-GELAN
        # Top-down feature pathway (S5 -> S4 -> S3)
        self.reduce_s5 = GSConv(512, 256, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.fuse_s4 = CAAGELAN(256 + 256, 256)
        self.reduce_s4 = GSConv(256, 128, k=1)
        
        self.fuse_s3 = CAAGELAN(128 + 128, 128)
        
        # 3. Multi-Scale Detection Prediction Heads
        # Head for small objects (S3 scale: 128 channels)
        self.head_s3 = nn.Conv2d(128, (num_classes + 4), kernel_size=1)
        # Head for medium objects (S4 scale: 256 channels)
        self.head_s4 = nn.Conv2d(256, (num_classes + 4), kernel_size=1)
        # Head for large objects (S5 scale: 256 channels)
        self.head_s5 = nn.Conv2d(256, (num_classes + 4), kernel_size=1)
        
        # 4. Box Regression Loss Function
        self.piou_loss = PowerfulIoULoss()

    def forward(self, x):
        """
        Forward pass:
        Input: (B, 3, H, W)
        Returns: Multi-scale prediction heads tuple (out_s3, out_s4, out_s5)
        """
        # Step 1: Extract multi-scale features from EFENet backbone
        s3, s4, s5 = self.backbone(x)
        
        # Step 2: Cross-scale fusion using GSConv and CAA-GELAN
        # Process S5
        p5 = self.reduce_s5(s5)
        
        # Upsample P5 and fuse with S4
        p5_up = self.upsample(p5)
        p4_cat = torch.cat([p5_up, s4], dim=1)
        p4 = self.fuse_s4(p4_cat)
        
        # Upsample P4 and fuse with S3
        p4_up = self.upsample(self.reduce_s4(p4))
        p3_cat = torch.cat([p4_up, s3], dim=1)
        p3 = self.fuse_s3(p3_cat)
        
        # Step 3: Compute predictions across the 3 scales
        out_s3 = self.head_s3(p3) # (B, num_classes + 4, H/8, W/8)
        out_s4 = self.head_s4(p4) # (B, num_classes + 4, H/16, W/16)
        out_s5 = self.head_s5(p5) # (B, num_classes + 4, H/32, W/32)
        
        return out_s3, out_s4, out_s5

    def compute_loss(self, pred_boxes, target_boxes, pred_cls, target_cls):
        """
        Computes combined training loss: PIoU Box Loss + Cross Entropy Class Loss
        """
        # Box regression loss via Powerful-IoU
        box_loss = self.piou_loss(pred_boxes, target_boxes)
        
        # Classification loss
        cls_loss = nn.functional.cross_entropy(pred_cls, target_cls)
        
        total_loss = box_loss + (0.5 * cls_loss)
        return total_loss, box_loss, cls_loss
