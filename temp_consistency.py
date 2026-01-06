import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- CONFIGURATION ---
# Path to your trained model (make sure this file is in the same folder)
MODEL_PATH = 'Deepfake_Detector_Final.h5' 

# Path to the video you want to test
VIDEO_PATH = 'my_test_video.mp4'  # <--- REPLACE THIS with your video filename
# ---------------------

# 1. LOAD YOUR TRAINED MODEL
if os.path.exists(MODEL_PATH):
    print(f"⏳ Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded! It's ready to see.")
else:
    print(f"❌ Error: Model file '{MODEL_PATH}' not found in the current directory.")
    exit()

# 2. HELPER: PREPROCESS VIDEO
def prepare_video(video_path):
    IMG_SIZE = 224
    SEQ_LENGTH = 20
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video.")
        return None

    # Use Haar Cascade (Lightweight Face Detector)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Skip frames to cover the whole video
    skip_rate = max(1, total_frames // SEQ_LENGTH)
    
    count = 0
    frames_extracted = 0
    
    print("🔍 Processing video frames...")
    
    while frames_extracted < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret: break
        
        if count % skip_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = frame[y:y+h, x:x+w]
            else:
                # Fallback: Center crop
                h, w, _ = frame.shape
                c_h, c_w = h//2, w//2
                face = frame[max(0,c_h-100):c_h+100, max(0,c_w-100):c_w+100]

            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                frames.append(face)
                frames_extracted += 1
            except Exception as e:
                pass
        count += 1
    
    cap.release()
    
    # If video was too short, pad it
    if len(frames) == 0: return None
    data = np.array(frames) / 255.0
    if len(data) < SEQ_LENGTH:
        padding = np.zeros((SEQ_LENGTH - len(data), IMG_SIZE, IMG_SIZE, 3))
        data = np.concatenate((data, padding), axis=0)
        
    return np.expand_dims(data, axis=0)

# 3. PREDICT
print(f"\n🎥 Testing video: {VIDEO_PATH}")
video_data = prepare_video(VIDEO_PATH)

if video_data is not None:
    prediction = model.predict(video_data)[0][0]
    score = prediction * 100
    
    print(f"--------------------------------------------------")
    print(f"🤖 DEEPFAKE SCORE: {score:.2f}%")
    print(f"--------------------------------------------------")
    
    if score > 50:
        print("🚨 RESULT: FAKE")
    else:
        print("✅ RESULT: REAL")
else:
    print("⚠️ Could not process video data.")