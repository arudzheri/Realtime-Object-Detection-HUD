# 🎯 Realtime Object Detection HUD
A real-time object detection system with a Heads-Up Display (HUD) inspired by sci-fi interfaces like Iron Man’s helmet view. This Python application uses YOLO and OpenCV to detect objects via webcam and overlays a futuristic HUD for enhanced visualization.

🚀 Features
Real-time webcam-based object detection

Integration with YOLO (You Only Look Once) for accurate, fast detection

Futuristic HUD UI with bounding boxes and labels

Customizable display overlays (crosshair, frame rate, detection stats)

Optional: Streamlit/Flask interface for deployment

📸 Demo

🛠️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/arudzheri/Realtime-Object-Detection-HUD.git
cd Realtime-Object-Detection-HUD
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt

Download YOLO weights

Download pretrained YOLOv3 weights from YOLO official site or use OpenCV's built-in YOLO models.

Place them in the yolo/ directory.

Run the app

bash
Copy
Edit
python detect.py
(Or launch streamlit_app.py / app.py if using a Streamlit/Flask UI)

🧠 Technologies Used
Python

OpenCV

YOLO (v3 or v5)

Streamlit or Flask (optional GUI)

NumPy, time, argparse, etc.
