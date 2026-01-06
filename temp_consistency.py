import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# 1. SETTINGS & SUPPRESSION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide all but error logs
MODEL_PATH = 'Deepfake_Detector_Final.h5' 
VIDEO_PATH = 'newvid03.mp4'  # <-- CHANGE THIS to your filename
SEQ_LENGTH = 20
IMG_SIZE = 224

# 2. DEFINE THE ARCHITECTURE (Identical to your Colab Training)
def build_model_architecture():
    # Load InceptionV3 without weights (we will load your trained ones)
    cnn_base = applications.InceptionV3(weights=None, include_top=False, pooling='avg')
    
    inputs = layers.Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    
    # TimeDistributed allows the CNN to see each of the 20 frames
    x = layers.TimeDistributed(cnn_base)(inputs)
    
    # LSTM processes the temporal relationship (movement) between frames
    x = layers.LSTM(64, return_sequences=False, dropout=0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 3. SMART LOADING LOGIC
print("⏳ Attempting to load your model...")

model = None

# Attempt 1: Standard Direct Load
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Method 1 Success: Full model loaded directly.")
except Exception:
    # Attempt 2: Architecture Rebuild + Weights Load
    try:
        model = build_model_architecture()
        model.load_weights(MODEL_PATH)
        print("✅ Method 2 Success: Weights loaded into rebuilt architecture.")
    except Exception as e:
        # Attempt 3: By-Layer Loading (The "Nuclear" Option)
        try:
            model = build_model_architecture()
            model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
            print("⚠️ Method 3 Warning: Loaded weights by matching layer names (partial success).")
        except Exception as e_final:
            print(f"❌ Critical Error: Could not load the model.\nDetails: {e_final}")
            exit()

# 4. PREPROCESSING FUNCTION
def prepare_video(path):
    if not os.path.exists(path):
        print(f"❌ Error: Video file '{path}' not found.")
        return None

    cap = cv2.VideoCapture(path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQ_LENGTH:
        print(f"❌ Error: Video only has {total_frames} frames. Need at least {SEQ_LENGTH}.")
        cap.release()
        return None

    # Calculate exactly which frames to grab to cover the whole video
    skip_rate = max(1, total_frames // SEQ_LENGTH)
    
    print(f"🔍 Analyzing {total_frames} frames...")
    
    curr_frame = 0
    while len(frames) < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret: break
        
        if curr_frame % skip_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Take largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = frame[y:y+h, x:x+w]
            else:
                # Fallback to center crop
                h_f, w_f, _ = frame.shape
                face = frame[h_f//4:3*h_f//4, w_f//4:3*w_f//4]

            # Resize and normalize
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            frames.append(face / 255.0)
            
        curr_frame += 1
    
    cap.release()
    
    # Return with batch dimension: (1, 20, 224, 224, 3)
    return np.expand_dims(np.array(frames), axis=0)

# 5. RUN INFERENCE
video_data = prepare_video(VIDEO_PATH)

if video_data is not None:
    print("🧠 Predicting...")
    prediction = model.predict(video_data, verbose=0)[0][0]
    score = prediction * 100
    
    print("\n" + "="*40)
    print(f"🎥 VIDEO: {VIDEO_PATH}")
    print(f"🤖 DEEPFAKE SCORE: {score:.2f}%")
    if score > 50:
        print("🚨 RESULT: FAKE DETECTED")
    else:
        print("✅ RESULT: LIKELY REAL")
    print("="*40 + "\n")