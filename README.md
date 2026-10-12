# 🦺 Industrial Safety Monitor

A real-time computer vision system for **PPE (Personal Protective Equipment) compliance monitoring** on construction sites, built with YOLOv8 and deployed via Streamlit.

---

## 📋 Overview

Detects construction site safety violations in real-time from images or webcam feed:

| **Safe (Green)** | **Violations (Red)** |
| ---------------- | -------------------- |
| Hardhat ✅       | NO-Hardhat 🚨        |
| Mask ✅          | NO-Mask 🚨           |
| Safety-Vest ✅   | NO-Safety-Vest 🚨    |
| Safety-Cone ✅   | —                    |
| Vehicle ✅       | —                    |

---
## 🌐 Live Demo

👉 [Try it here](https://ppe-safety-monitor-fsc3xntt4dpmd3824th2fu.streamlit.app)

Upload any construction site image and get instant PPE compliance analysis.

## 🎯 Features

- **Real-time webcam monitoring** with violation logging
- **Image upload interface** via Streamlit web app
- **CSV violation logging** with timestamps and confidence scores
- **REST API** via FastAPI for system integration
- **Fine-tuned YOLOv8s** model achieving **85.1% mAP50**

---

## 🛠️ Tech Stack

- **YOLOv8** (Ultralytics) — Object detection
- **PyTorch** — Deep learning framework
- **Streamlit** — Web app interface
- **FastAPI** — REST API backend
- **OpenCV** — Image processing
- **Azure** — Designed for cloud deployment

---

## 📁 Project Structure

```
yolo-project/
├── app.py                   # Streamlit web interface
├── safety_monitor.py        # Real-time webcam monitoring
├── train.py                 # Model training script
├── merge_datasets.py        # Dataset merging utility
├── explore_data.py          # Dataset exploration
├── check.py                 # Environment verification
├── requirements.txt         # Python dependencies
└── runs/
    └── detect/ppe_runs/ppe_v1/
        └── weights/
            └── best.pt      # Trained model (85.1% mAP50)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended)

### 1. Clone the repo

```bash
git clone https://github.com/Amine-Dahmen/ppe-safety-monitor.git
cd ppe-safety-monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify environment

```bash
python check.py
```

### 4. Choose your interface

#### Option A — Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Visit: `http://localhost:8501`

#### Option B — Real-time Webcam Monitor

```bash
python safety_monitor.py
```

#### Option C — FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit: `http://localhost:8000`

---

## 📊 Model Performance

**Architecture**: YOLOv8s fine-tuned on merged PPE dataset  
**Overall mAP50**: **0.851**

| Class          | mAP50 |
| -------------- | ----- |
| Hardhat        | 0.900 |
| NO-Hardhat     | 0.731 |
| Safety-Vest    | 0.949 |
| NO-Safety-Vest | 0.794 |
| Mask           | 0.948 |
| NO-Mask        | 0.700 |
| Person         | 0.862 |
| Safety-Cone    | 0.925 |
| Vehicle        | 0.853 |

---

## 🗄️ Training Dataset

Merged from two Roboflow datasets:

| Dataset                     | Images                          |
| --------------------------- | ------------------------------- |
| Construction-Site-Safety-23 | 4,718 train / 100 val / 71 test |
| PPE-vest-hardhat-1          | 171 additional train images     |
| **Total**                   | **4,889 images**                |

**Class distribution in training set:**

| Class          | Instances |
| -------------- | --------- |
| Person         | 8,752     |
| Hardhat        | 7,189     |
| NO-Safety-Vest | 7,171     |
| Vehicle        | 5,466     |
| Safety-Vest    | 3,895     |
| NO-Hardhat     | 3,041     |
| Safety-Cone    | 2,858     |
| NO-Mask        | 2,869     |
| Mask           | 1,628     |

---

## 🔬 API Reference

### POST `/detect`

Upload an image for PPE compliance checking.

**Request:**

```
Content-Type: multipart/form-data
file: [image file]
```

**Response Headers:**

| Header         | Description                        |
| -------------- | ---------------------------------- |
| `X-Status`     | `COMPLIANT` or `VIOLATION`         |
| `X-Violations` | Comma-separated list of violations |
| `X-Safe`       | Comma-separated safe detections    |

**Response Body:** Annotated image with bounding boxes drawn.

---

## 📝 Violation Log

Violations are automatically logged to `violations_log.csv`:

```
timestamp,violation,confidence
2026-10-12 14:30:25,NO-Hardhat,0.87
2026-10-12 14:30:26,NO-Safety-Vest,0.92
```

---

## 🐛 Troubleshooting

| Issue                  | Solution                                                  |
| ---------------------- | --------------------------------------------------------- |
| `CUDA not available`   | Install CUDA-enabled PyTorch via conda                    |
| `conda not recognized` | Use Anaconda Prompt or run `conda init powershell`        |
| `Webcam not opening`   | Use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` on Windows       |
| `No module found`      | Run `conda activate yolo` before any script               |
| `Model not found`      | Check path: `runs/detect/ppe_runs/ppe_v1/weights/best.pt` |

---

## 📄 License

MIT License
