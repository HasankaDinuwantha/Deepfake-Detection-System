import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"                 # path to trained model
VIDEO_PATH = "SORA 2.mp4"   # path to input video
CONF_THRES = 0.4

# =========================
# CHECKS
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# =========================
# LOAD MODEL
# =========================
print("🔍 Loading watermark detection model...")
model = YOLO(MODEL_PATH)

# =========================
# RUN DETECTION
# =========================
print("🚀 Processing video...")
results = model.predict(
    source=VIDEO_PATH,
    conf=CONF_THRES,
    save=True
)

# =========================
# FIND OUTPUT VIDEO
# =========================
runs_dir = "runs/detect"
predict_folders = [
    os.path.join(runs_dir, d)
    for d in os.listdir(runs_dir)
    if d.startswith("predict")
]

latest_folder = max(predict_folders, key=os.path.getmtime)

output_video = None
for f in os.listdir(latest_folder):
    if f.endswith((".mp4", ".avi", ".mov")):
        output_video = os.path.join(latest_folder, f)
        break

if output_video:
    print(f"✅ Done!")
    print(f"📁 Output saved to: {output_video}")
else:
    print("❌ Could not find output video.")
