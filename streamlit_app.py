import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

st.title("🛡️ Realtime Object Detection HUD")

model = YOLO('yolov8n.pt')
run = st.checkbox('Activate Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Simulate HUD effect
    cv2.putText(frame, '🛡️ SYSTEM ONLINE', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
