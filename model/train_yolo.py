from ultralytics import YOLO
import torch
print(">>> Script started")

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Load a pretrained YOLO model
    model = YOLO("yolov8m.pt")  # BEST choice based on EDA

    print(">>> Starting YOLO train now...")
    # Train for 1 epoch
    model.train(
        data="EDA/config/dataset.yaml",
        epochs=1,
        imgsz=1280,       # higher resolution → better for traffic light, sign
        batch=4,          # RTX 3050 sweet spot
        workers=2,
        device=0,
        lr0=0.001,
        mosaic=1.0,       # keep mosaic ON for diversity
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )
    print(">>> Training function executed")

if __name__ == "__main__":
    main()
