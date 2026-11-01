from ultralytics import YOLO
import os

def main():
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Path to dataset config
    data_path = os.path.join(os.path.dirname(__file__), "snake-2", "data.yaml")
    
    print("Starting training on Snake dataset...")
    # Train the model
    # Note: On CPU, 20-30 epochs is recommended for a quick test. On GPU, 50-100 epochs is standard.
    results = model.train(
        data=data_path,
        epochs=30,
        imgsz=640,
        batch=16,
        name="snake_detector_yolov8n"
    )
    
    print("Training complete! Best weights saved to runs/detect/snake_detector_yolov8n/weights/best.pt")

if __name__ == "__main__":
    main()
