# YOLO Project - How to Run

## Environment Setup

This project uses YOLOv8 with PyTorch for object detection on construction site safety and PPE datasets.

### Verify Your Environment

Run `python check.py` to verify your environment has all required dependencies:

```bash
python check.py
```

This will check:
- CUDA availability and GPU name
- PyTorch version
- OpenCV version
- YOLOv8 model loading

### Merged Dataset

I've merged two datasets into a single unified dataset:

- **Construction-Site-Safety-23** (2,521 images: 2,350 train + 100 valid + 71 test)
- **PPE-vest-hardhat-1** (2,368 train images)

**Merged dataset location:** `./merged_dataset/`

The merged dataset structure:
```
merged_dataset/
├── train/      (4,718 images)
│   ├── images/
│   └── labels/
├── valid/      (100 images)
│   ├── images/
│   └── labels/
├── test/       (71 images)
│   ├── images/
│   └── labels/
└── data.yaml
```

### Running Inference

To run object detection on images/video:

```bash
# Using the merged dataset
yolo predict model=yolov8n.pt source=./merged_dataset/train/images/000001.jpg

# Or with the check script
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model.predict(source='path/to/image.jpg')"
```

### Training

To train a YOLO model on the merged dataset:

```bash
yolo train data=merged_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Or use the check script's model:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Loads pre-trained model
# Then train:
# model.train(data='merged_dataset/data.yaml', epochs=50)
```

### Detection on Webcam

```bash
python webcam_test.py  # Already in your project
```

### Project Structure

```
yolo-project/
├── merged_dataset/      <- Merged datasets (created by merge_datasets.py)
├── Construction-Site-Safety-23/  <- Original dataset 1
├── PPE-vest-hardhat-1/   <- Original dataset 2
├── yolov8n.pt           <- Pre-trained YOLOv8n model
├── check.py             <- Environment verification script
├── merge_datasets.py    <- Script to merge datasets
├── webcam_test.py       <- Webcam test script
├── download_data.py     <- Data download script
└── explore_data.py      <- Data exploration script
```

### Quick Start

1. Verify environment: `python check.py`
2. Test with webcam: `python webcam_test.py`
3. Run inference on merged dataset: `yolo predict model=yolov8n.pt source=./merged_dataset/train/images/000001.jpg`
4. Train custom model: `yolo train data=merged_dataset/data.yaml model=yolov8n.pt epochs=50`

### Notes

- The merged dataset combines both original datasets with unique filename numbering
- Class names are preserved from the original datasets (9 classes total)
- Make sure you have sufficient disk space and RAM for training
- For best results, consider training with at least 50 epochs