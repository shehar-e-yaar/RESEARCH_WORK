import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
from snake_detr import SnakeDETR

class SnakeDataset(Dataset):
    """
    Loads YOLO-format snake images and labels
    """
    def __init__(self, img_dir, lbl_dir, img_size=640):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        # Read label
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")
        
        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # YOLO format: xc, yc, w, h
                        xc, yc, w, h = map(float, parts[1:5])
                        # Convert to x1, y1, x2, y2
                        x1 = (xc - w / 2) * self.img_size
                        y1 = (yc - h / 2) * self.img_size
                        x2 = (xc + w / 2) * self.img_size
                        y2 = (yc + h / 2) * self.img_size
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id)
                        
        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
            
        return img_tensor, torch.tensor(boxes[0], dtype=torch.float32), torch.tensor(labels[0], dtype=torch.long)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Snake-DETR on device: {device}")
    
    # Paths
    base_dir = os.path.join(os.path.dirname(__file__), "snake-2")
    train_img = os.path.join(base_dir, "train", "images")
    train_lbl = os.path.join(base_dir, "train", "labels")
    
    dataset = SnakeDataset(train_img, train_lbl, img_size=640)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Instantiate custom Snake-DETR
    model = SnakeDETR(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print(f"Loaded {len(dataset)} training images.")
    print("Starting Snake-DETR training with Powerful-IoU loss...")
    
    model.train()
    epochs = 5 # Demonstration / test run
    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0.0
        for i, (imgs, target_boxes, target_cls) in enumerate(dataloader):
            imgs = imgs.to(device)
            target_boxes = target_boxes.to(device)
            target_cls = target_cls.to(device)
            
            optimizer.zero_grad()
            
            # Forward through EFENet + GSConv + Heads
            out_s3, out_s4, out_s5 = model(imgs)
            
            # Extract box and class predictions from dominant head (S4 scale)
            # Channel slicing: [cls_0, cls_1, x, y, w, h]
            pred_cls = out_s4[:, :2].mean(dim=[2, 3])
            pred_box_raw = out_s4[:, 2:].mean(dim=[2, 3])
            
            # Compute loss via Powerful-IoU
            loss, box_loss, cls_loss = model.compute_loss(pred_box_raw, target_boxes, pred_cls, target_cls)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f} (PIoU: {box_loss.item():.4f})")
                
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"--> Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}\n")
        
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/snake_detr_custom.pth")
    print("Training finished! Saved weights to weights/snake_detr_custom.pth")

if __name__ == "__main__":
    train()
