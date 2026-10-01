# 🎯 Realtime Object Detection HUD

A real-time object detection system with a sci-fi inspired Heads-Up Display (HUD), similar to Iron Man's helmet view. Using YOLO, OpenCV, and Python, it detects objects from a live camera feed and overlays HUD information in real time.

---

## 🚀 Features

- Real-time detection with YOLO models
- HUD-style overlay: crosshair, bounding boxes, labels, FPS counter
- Python + OpenCV implementation
- Optional Streamlit UI for a cleaner interactive interface
- Customizable overlays for different use-cases

---

## 🛠️ Installation

```bash
git clone https://github.com/arudzheri/Realtime-Object-Detection-HUD.git
cd Realtime-Object-Detection-HUD
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The project automatically looks for required assets in either the project root or an `assets/` directory, so it works without manual relocation.

---

## ▶️ Usage

### Desktop OpenCV HUD

```bash
python hud_app.py
```

Press `q` to quit the window.

### Streamlit version

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Tech stack

- Python
- OpenCV
- YOLO (v3 / v5 / v8)
- Streamlit (optional)
- NumPy, Pillow, pyttsx3

---

## Notes

- A webcam is required for live detection.
- If your system cannot access the camera, make sure the device is connected and available to Python.
- The app expects the YOLO weights file `yolov8n.pt` to be present in the project folder.
