# Realtime Object Detection HUD

A real-time object detection system with a sci-fi inspired Heads-Up Display (HUD), similar to Iron Man's helmet view. It uses YOLO, OpenCV, and Python to detect objects from a live camera feed and overlay HUD-style information.

---

## Features

- Real-time detection with YOLO models
- HUD-style overlay: crosshair, bounding boxes, labels, FPS counter
- Python + OpenCV implementation
- Optional Streamlit UI for an interactive browser-based display
- Customizable overlays for different use-cases

---

## Requirements

Before installing, make sure you have:

- Python 3.9+ recommended
- A working webcam connected to your machine
- A desktop display or GUI environment available for the OpenCV HUD
- Internet access for the first YOLO model download (`yolov8n.pt`)

---

## Local setup

### 1) Clone the repository

```bash
git clone https://github.com/arudzheri/Realtime-Object-Detection-HUD.git
cd Realtime-Object-Detection-HUD
```

### 2) Create and activate a virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Windows (Command Prompt):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

Optional developer tools:

```bash
pip install -r requirements-dev.txt
```

### 4) Confirm required assets are present

The project expects these files to exist in the project root or in an `assets/` directory:

- `hud_overlay.png`
- `target_icon.png`
- `Orbitron-Regular.ttf`

If any are missing, the app will stop early with a clear error message explaining what is missing.

---

## Run the project

### OpenCV desktop HUD

```bash
python hud_app.py
```

Press `q` to quit.

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## Startup checks and troubleshooting

The app now validates the environment before launching the camera feed. Common startup issues and fixes:

- Webcam not detected:
  - confirm the device is connected
  - check OS camera permissions
  - try a different camera index if needed (for example `cv2.VideoCapture(1)` in code)

- Missing assets:
  - make sure `hud_overlay.png`, `target_icon.png`, and `Orbitron-Regular.ttf` are in the project root or `assets/`

- YOLO model download fails:
  - check internet access
  - ensure `ultralytics` can download `yolov8n.pt`
  - try reinstalling dependencies with `pip install -r requirements.txt`

- Audio engine issue:
  - the sound feedback is optional
  - if `pyttsx3` fails to initialize, the app continues without voice output

---

## Tech stack

- Python
- OpenCV
- YOLO (v3 / v5 / v8)
- Streamlit (optional)
- NumPy, Pillow, pyttsx3

---

## Notes

- A webcam is required for live detection.
- The app expects the YOLO weights file `yolov8n.pt` to be available.
- The desktop app requires a GUI-enabled machine with a display.
