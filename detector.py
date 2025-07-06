import os
import time
import csv
import glob
import cv2
import torch
import requests
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ✅ Settings
MODEL_PATH = "monkeydetector.pt"
INPUT_FOLDER = "mp4"
OUTPUT_FOLDER = "output"
CROP_DIR = "detections/crops"
LOG_PATH = "logs/detections.csv"
SAVE_OUTPUT_VIDEO = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Setup
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ✅ Init log if it doesn't exist
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame_id", "timestamp", "class", "confidence", "x1", "y1", "x2", "y2"])

# ✅ Helpers
def log_detection(video_name, frame_id, timestamp, class_name, conf, bbox):
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([video_name, frame_id, timestamp, class_name, round(conf, 3), *map(int, bbox)])

def save_crop(frame, bbox, video_name, frame_id, conf):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    crop_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    filename = f"{video_name}_frame{frame_id}_{int(conf*100)}.jpg"
    crop_img.save(os.path.join(CROP_DIR, filename))

def send_discord_alert(video, frame_id, conf):
    if not WEBHOOK_URL:
        return
    msg = {
        "content": f"🚨 Monkey detected in **{video}**, frame `{frame_id}`, confidence `{conf:.2f}`"
    }
    try:
        requests.post(WEBHOOK_URL, json=msg)
    except Exception as e:
        print(f"❌ Discord alert failed: {e}")

try:
    import winsound
    def play_alert():
        winsound.Beep(1000, 300)
except:
    def play_alert():
        print("🔊 Monkey detected (sound disabled on this OS)")

# ✅ Load model and move to GPU if available
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ✅ Get videos
video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))
if not video_files:
    print("❌ No videos found in 'mp4/' folder.")
    exit()

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    print(f"\n▶️ Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {video_name}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30  # fallback if corrupted or undefined

    if SAVE_OUTPUT_VIDEO:
        out_path = os.path.join(OUTPUT_FOLDER, f"{video_name}_out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_id = 0
    monkey_count = 0
    monkey_alert_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame_time = frame_id / fps
        timestamp = time.strftime('%H:%M:%S', time.gmtime(frame_time))

        # ✅ GPU-powered detection
        results = model.predict(frame, imgsz=640, conf=0.5, device=DEVICE, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            for box in detections:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name.lower() == "monkey":
                    monkey_count += 1

                    if not monkey_alert_sent:
                        print(f"🚨 Monkey in {video_name} — frame {frame_id} @ {timestamp} (conf {conf:.2f})")
                        play_alert()
                        send_discord_alert(video_name, frame_id, conf)
                        monkey_alert_sent = True

                    # ✅ Always save and log
                    save_crop(frame, [x1, y1, x2, y2], video_name, frame_id, conf)
                    log_detection(video_name, frame_id, timestamp, class_name, conf, [x1, y1, x2, y2])

        if SAVE_OUTPUT_VIDEO:
            out.write(results[0].plot())

        print(f"⏳ {video_name} - Frame {frame_id}", end="\r")

    cap.release()
    if SAVE_OUTPUT_VIDEO:
        out.release()

    print(f"\n✅ Done: {video_name} — {monkey_count} monkey detections")
    print(f"📊 Summary for {video_name}")
    print(f"   • Frames processed: {frame_id}")
    print(f"   • Monkeys detected: {monkey_count}")
    print(f"   • First alert sent: {'✅' if monkey_alert_sent else '❌'}")

print("\n✅ All videos processed.")
