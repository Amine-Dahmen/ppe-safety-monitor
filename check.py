import torch
import cv2
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)

model = YOLO("yolov8n.pt")
print("YOLOv8 ready!")