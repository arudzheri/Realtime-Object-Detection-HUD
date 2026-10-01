import cv2
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from ultralytics import YOLO
import time
import pyttsx3

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIRS = [BASE_DIR / "assets", BASE_DIR]


def resolve_asset(filename: str) -> str:
    for asset_dir in ASSET_DIRS:
        candidate = asset_dir / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Asset '{filename}' was not found in {ASSET_DIRS}")


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]

    # Slice the overlay into the background
    slice_img = img[y:y + h, x:x + w]
    slice_img[:] = (slice_img * (1 - alpha_mask) + img_overlay[:, :, :3] * alpha_mask).astype(np.uint8)


# 🔊 AI Voice Feedback (safe if audio engine isn't available)
try:
    engine = pyttsx3.init()
    engine.say("Targeting system online. Scanning initiated.")
    engine.runAndWait()
except Exception:
    engine = None

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Please make sure a camera is connected and available.")

# Load HUD overlay once
hud_overlay_path = resolve_asset("hud_overlay.png")
hud_overlay = cv2.imread(hud_overlay_path, cv2.IMREAD_UNCHANGED)
if hud_overlay is None:
    raise FileNotFoundError(f"Could not load HUD overlay from {hud_overlay_path}")

# Load PNG icon with alpha channel
icon_path = resolve_asset("target_icon.png")
icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
if icon is None:
    raise FileNotFoundError(f"Could not load target icon from {icon_path}")
icon = cv2.resize(icon, (50, 50))

# Extract alpha mask and RGB channels
alpha_icon = icon[:, :, 3] / 255.0
icon_rgb = icon[:, :, :3]

# Get dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Fallback dimensions if camera API returns 0
if width <= 0 or height <= 0:
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

center = (width // 2, height // 2)

# FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize + overlay HUD image
    resized_overlay = cv2.resize(hud_overlay, (frame.shape[1], frame.shape[0]))
    alpha_overlay = resized_overlay[:, :, 3] / 255.0
    overlay_rgb = resized_overlay[:, :, :3]
    overlay_image_alpha(frame, overlay_rgb, (0, 0), alpha_overlay)

    # 🔥 Glowing box in the bottom-right corner
    glow_overlay = frame.copy()
    cv2.rectangle(glow_overlay, (frame.shape[1] - 150, frame.shape[0] - 80),
                (frame.shape[1] - 10, frame.shape[0] - 10), (0, 255, 255), -1)
    frame = cv2.addWeighted(glow_overlay, 0.4, frame, 0.6, 0)

    # 🧑‍🎨 Convert frame to PIL for drawing text
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)

    # Load futuristic font (make sure the file exists in 'assets/' or project root)
    font_path = resolve_asset("Orbitron-Regular.ttf")
    font = ImageFont.truetype(font_path, 24)

    # Draw futuristic HUD text
    draw.text((50, 50), "TARGET LOCKED", font=font, fill=(0, 255, 0))

    # Convert back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    # Run detection
    results = model(frame)[0]

    # Draw YOLO detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = f"{model.names[cls]} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Add the semi-transparent overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # HUD Overlay: Crosshair
    cv2.line(frame, (center[0] - 20, center[1]),
            (center[0] + 20, center[1]), (255, 0, 255), 1)
    cv2.line(frame, (center[0], center[1] - 20),
            (center[0], center[1] + 20), (255, 0, 255), 1)

    # Pulsing energy circle
    radius = int(30 + 10 * np.sin(time.time() * 3))
    cv2.circle(frame, center, radius, (0, 255, 255), 2)

    # Radar arcs
    for r in range(80, 150, 20):
        cv2.ellipse(frame, center, (r, r), 0, 0, 90, (0, 128, 255), 1)

    # Scanning line animation
    line_offset = int((time.time() * 80) % height)
    cv2.line(frame, (0, line_offset), (width, line_offset), (0, 255, 255), 1)

    # System status
    cv2.putText(frame, '🛡️ SYSTEM STATUS: ONLINE', (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Object count
    cv2.putText(frame, f'Targets: {len(results.boxes)}', (10, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Simulate Iron Man HUD style (overlay)
    cv2.circle(frame, (50, 50), 20, (0, 255, 255), 2)
    cv2.putText(frame, 'TARGET SYSTEM ONLINE', (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

    overlay_image_alpha(frame, icon_rgb, (10, 10), alpha_icon)
    cv2.imshow('Iron Man HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
