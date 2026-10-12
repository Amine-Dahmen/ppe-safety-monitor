import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Industrial Safety Monitor", page_icon="🦺", layout="centered")

st.title("🦺 Industrial Safety Monitor")
st.markdown("Upload a construction site image to check PPE compliance.")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/ppe_runs/ppe_v1/weights/best.pt")

model = load_model()

VIOLATION_CLASSES = {'NO-Hardhat', 'NO-Safety-Vest', 'NO-Mask'}
CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety-Vest', 'Person', 'Safety-Cone', 'Safety-Vest', 'vehicle'
]
COLORS = {
    'Hardhat':         (0, 255, 0),
    'Mask':            (0, 255, 0),
    'Safety-Vest':     (0, 255, 0),
    'Safety-Cone':     (0, 255, 0),
    'Person':          (255, 165, 0),
    'vehicle':         (255, 165, 0),
    'NO-Hardhat':      (0, 0, 255),
    'NO-Mask':         (0, 0, 255),
    'NO-Safety-Vest':  (0, 0, 255),
}

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing image..."):
        results = model(frame, conf=0.5, verbose=False)
        detections = results[0].boxes

    violations = []
    for box in detections:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if class_name in VIOLATION_CLASSES:
            violations.append(class_name)

    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(output, caption="Detection Result", use_container_width=True)

    if violations:
        st.error(f"🚨 VIOLATION DETECTED: {', '.join(set(violations))}")
    else:
        st.success("✅ ALL CLEAR — PPE COMPLIANT")