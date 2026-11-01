import torch
import torch.nn as nn

class PowerfulIoULoss(nn.Module):
    """
    Powerful-IoU (PIoU) Loss Function
    As formulated in Equations 8-10 (Page 11) of the Snake-DETR paper.
    
    Equations:
      P = ((dw1 / W_gt) + (dw2 / W_gt) + (dh1 / h_gt) + (dh2 / h_gt)) / 4   (Eq. 8)
      PIoU = IoU - (1 - exp(-P^2))                                            (Eq. 9)
      L_PIoU = 1 - PIoU = L_IoU + 1 - exp(-P^2)                                (Eq. 10)
      
    Where:
      - W_gt, h_gt: Width and Height of the ground-truth box.
      - dw1, dw2: Distances between predicted and ground-truth horizontal edges.
      - dh1, dh2: Distances between predicted and ground-truth vertical edges.
      - P: Size-adaptive penalty factor.
    """
    def __init__(self, eps=1e-7):
        super(PowerfulIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
          pred_boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
          target_boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        Returns:
          loss: Scalar PIoU loss value
        """
        # Predicted box coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(-1)
        # Target box coordinates
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.unbind(-1)

        # 1. Standard IoU calculation
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area_pred = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        area_target = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union_area = area_pred + area_target - inter_area + self.eps

        iou = inter_area / union_area

        # 2. Dimensions of ground-truth box
        w_gt = (b2_x2 - b2_x1).clamp(min=self.eps)
        h_gt = (b2_y2 - b2_y1).clamp(min=self.eps)

        # 3. Edge distance penalties (dw1, dw2, dh1, dh2)
        dw1 = torch.abs(b1_x1 - b2_x1)
        dw2 = torch.abs(b1_x2 - b2_x2)
        dh1 = torch.abs(b1_y1 - b2_y1)
        dh2 = torch.abs(b1_y2 - b2_y2)

        # Eq. 8: Penalty factor P adapted to object size
        P = ((dw1 / w_gt) + (dw2 / w_gt) + (dh1 / h_gt) + (dh2 / h_gt)) / 4.0

        # Eq. 9: PIoU formulation
        piou = iou - (1.0 - torch.exp(-(P ** 2)))

        # Eq. 10: Loss L_PIoU = 1 - PIoU
        loss = 1.0 - piou
        return loss.mean()
