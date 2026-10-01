import time
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIRS = [BASE_DIR / "assets", BASE_DIR]


def resolve_asset(filename: str) -> str:
    for asset_dir in ASSET_DIRS:
        candidate = asset_dir / filename
        if candidate.exists():
            return str(candidate)
    return ""


def validate_runtime_environment():
    required_assets = ["hud_overlay.png", "target_icon.png", "Orbitron-Regular.ttf"]
    missing_assets = [name for name in required_assets if not resolve_asset(name)]
    if missing_assets:
        raise FileNotFoundError(
            "Missing required HUD assets: " + ", ".join(missing_assets)
            + ". Put them in the project root or an assets/ folder."
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Please connect a camera and ensure your OS allows access to it."
        )
    camera.release()

    try:
        YOLO("yolov8n.pt")
    except Exception as exc:
        raise RuntimeError(
            "YOLO model could not be initialized. Ensure ultralytics can download yolov8n.pt and you have internet access."
        ) from exc


@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")


def draw_hud(frame, model):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)

    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(frame, "🛡️ SYSTEM ONLINE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (255, 0, 255), 1)
    cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (255, 0, 255), 1)
    cv2.putText(frame, f"Targets: {len(results.boxes)}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame


def main():
    st.title("🛡️ Realtime Object Detection HUD")

    if "running" not in st.session_state:
        st.session_state.running = False

    col1, col2 = st.columns(2)
    if col1.button("Start webcam"):
        try:
            validate_runtime_environment()
            st.session_state.running = True
        except Exception as exc:
            st.error(f"Runtime check failed: {exc}")
            st.session_state.running = False
    if col2.button("Stop webcam"):
        st.session_state.running = False

    placeholder = st.empty()
    model = get_model()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please make sure a camera is connected and available.")
            st.session_state.running = False
            st.stop()

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera read failed. Please check your webcam connection.")
                    break

                frame = draw_hud(frame, model)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame, channels="RGB", use_container_width=True)
                time.sleep(0.03)
        finally:
            cap.release()
            st.session_state.running = False
    else:
        st.info("Press Start webcam to begin the object detection HUD.")


if __name__ == "__main__":
    main()
