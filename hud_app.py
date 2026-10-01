import cv2
import time
from pathlib import Path

import numpy as np
import pyttsx3
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

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
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return

    alpha_mask = np.asarray(alpha_mask, dtype=np.float32)
    if alpha_mask.shape != (h, w):
        alpha_mask = cv2.resize(alpha_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    slice_img = img[y:y + h, x:x + w]
    blend = slice_img.astype(np.float32) * (1.0 - alpha_mask[:, :, None]) + img_overlay[:, :, :3].astype(np.float32) * alpha_mask[:, :, None]
    slice_img[:] = blend.astype(np.uint8)


def load_png_asset(path: str):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image asset from {path}")
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def main():
    engine = None
    try:
        engine = pyttsx3.init()
        engine.say("Targeting system online. Scanning initiated.")
        engine.runAndWait()
    except Exception:
        engine = None

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Please make sure a camera is connected and available.")

    hud_overlay_path = resolve_asset("hud_overlay.png")
    hud_overlay = cv2.imread(hud_overlay_path, cv2.IMREAD_UNCHANGED)
    if hud_overlay is None:
        raise FileNotFoundError(f"Could not load HUD overlay from {hud_overlay_path}")
    if hud_overlay.shape[2] == 3:
        hud_overlay = cv2.cvtColor(hud_overlay, cv2.COLOR_BGR2BGRA)

    icon_path = resolve_asset("target_icon.png")
    icon = load_png_asset(icon_path)
    icon = cv2.resize(icon, (50, 50))
    alpha_icon = icon[:, :, 3] / 255.0 if icon.shape[2] > 3 else np.ones((icon.shape[0], icon.shape[1]), dtype=np.float32)
    icon_rgb = icon[:, :, :3]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center = (width // 2, height // 2)
    prev_time = 0.0
    font_path = resolve_asset("Orbitron-Regular.ttf")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if hud_overlay.shape[2] == 4:
                resized_overlay = cv2.resize(hud_overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                alpha_overlay = resized_overlay[:, :, 3] / 255.0
                overlay_rgb = resized_overlay[:, :, :3]
                overlay_image_alpha(frame, overlay_rgb, (0, 0), alpha_overlay)

            glow_overlay = frame.copy()
            cv2.rectangle(
                glow_overlay,
                (frame.shape[1] - 150, frame.shape[0] - 80),
                (frame.shape[1] - 10, frame.shape[0] - 10),
                (0, 255, 255),
                -1,
            )
            frame = cv2.addWeighted(glow_overlay, 0.4, frame, 0.6, 0)

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)
            try:
                font = ImageFont.truetype(font_path, 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((50, 50), "TARGET LOCKED", font=font, fill=(0, 255, 0))
            frame = cv2.cvtColor(np.asarray(pil_frame), cv2.COLOR_RGB2BGR)

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names.get(cls, str(cls))} {conf:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (255, 0, 255), 1)
            cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (255, 0, 255), 1)

            radius = int(30 + 10 * np.sin(time.time() * 3))
            cv2.circle(frame, center, radius, (0, 255, 255), 2)

            for r in range(80, 150, 20):
                cv2.ellipse(frame, center, (r, r), 0, 0, 90, (0, 128, 255), 1)

            line_offset = int((time.time() * 80) % height)
            cv2.line(frame, (0, line_offset), (width, line_offset), (0, 255, 255), 1)

            cv2.putText(frame, "🛡️ SYSTEM STATUS: ONLINE", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"Targets: {len(results.boxes)}", (10, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.circle(frame, (50, 50), 20, (0, 255, 255), 2)
            cv2.putText(frame, "TARGET SYSTEM ONLINE", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

            overlay_image_alpha(frame, icon_rgb, (10, 10), alpha_icon)
            cv2.imshow("Iron Man HUD", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
