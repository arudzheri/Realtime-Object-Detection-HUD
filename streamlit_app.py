import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIRS = [BASE_DIR / "assets", BASE_DIR]


def resolve_asset(filename: str) -> str:
    for asset_dir in ASSET_DIRS:
        candidate = asset_dir / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Asset '{filename}' was not found in {ASSET_DIRS}")


st.title("🛡️ Realtime Object Detection HUD")

run = st.checkbox("Activate Webcam", value=False)
frame_window = st.empty()

model = YOLO('yolov8n.pt')
cap = None

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please make sure a camera is connected and available.")
        st.stop()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera read failed. Please check your webcam connection.")
            break

        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, '🛡️ SYSTEM ONLINE', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame, channels="RGB")
        time.sleep(0.03)

    if cap is not None:
        cap.release()

else:
    st.info("Toggle the webcam switch to start the object detection HUD.")
