import torch
import torch.nn as nn
import librosa
import numpy as np
import os

# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================
class AudioDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. LOAD MODEL
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "audio_deepfake_detector.pth"

def load_model():
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = AudioDetector().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Error: '{MODEL_PATH}' not found.")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        exit()

# ==============================================================================
# 3. PREPROCESS
# ==============================================================================
def preprocess_tensor(y, sr):
    target_len = 16000 * 4
    if len(y) < target_len: y = np.pad(y, (0, target_len - len(y)))
    else: y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm_mel = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(norm_mel).float().unsqueeze(0).unsqueeze(0)

def run_inference(tensor, model):
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        return torch.softmax(logits, dim=1)[0][1].item()

# ==============================================================================
# 4. SILENCE-AWARE PREDICTION
# ==============================================================================
def predict_smart(audio_path, model):
    SR = 16000
    WINDOW = 4.0
    STRIDE = 2.0
    TARGET_LEN = int(SR * WINDOW)
    
    print(f"🔍 Scanning {audio_path}...")
    try:
        # 1. Load Audio
        y_raw, sr = librosa.load(audio_path, sr=SR)
        
        # 2. REMOVE SILENCE (Critical Fix for sample3)
        # trim returns (audio, index), we just want audio [0]
        y_trimmed, _ = librosa.effects.trim(y_raw, top_db=20) 
        
        if len(y_trimmed) < 1000: # If file was empty or purely silent
            print("   ⚠️ File is mostly silence. Using raw audio.")
            y = y_raw
        else:
            y = y_trimmed
            
        total_len = len(y)
        
        # If short, predict once
        if total_len < TARGET_LEN:
            score = run_inference(preprocess_tensor(y, sr), model)
            return "FAKE (AI)" if score > 0.5 else "REAL (Human)", score * 100

        # 3. Scan Chunks
        probs = []
        start = 0
        while start + TARGET_LEN <= total_len:
            chunk = y[start : start + TARGET_LEN]
            
            # Extra Energy Check: Skip chunks that are still too quiet
            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.005: # Only process if there is sound
                probs.append(run_inference(preprocess_tensor(chunk, sr), model))
            
            start += int(SR * STRIDE)
            
        if not probs: return "ERROR", 0.0
        
        # --- CALIBRATED LOGIC (V3) ---
        avg_score = sum(probs) / len(probs)
        max_score = max(probs)
        
        # Raise "Suspicious" bar to 0.90 to ignore human noise (Fix for sample6/sample2)
        suspicious_chunks = sum(1 for p in probs if p > 0.90) 
        
        print(f"   > Stats: Avg={avg_score:.2f}, Max={max_score:.2f}, High Confidence Chunks={suspicious_chunks}/{len(probs)}")

        # Rule 1: The "Smoking Gun" (Fix for sample3)
        # If we found a chunk with 95% confidence, it's AI, even if average is low.
        if max_score > 0.95:
             return "FAKE (AI)", max_score * 100

        # Rule 2: High Average (Standard Deepfake)
        if avg_score > 0.55:
            return "FAKE (AI)", avg_score * 100
            
        # Rule 3: Frequent Suspicious Activity
        if suspicious_chunks >= 2 and (suspicious_chunks / len(probs)) > 0.3:
             return "FAKE (AI)", max_score * 100

        # Otherwise Real
        return "REAL (Human)", (1.0 - avg_score) * 100

    except Exception as e:
        print(f"❌ Error: {e}")
        return "ERROR", 0.0

# ==============================================================================
# 5. MAIN
# ==============================================================================
if __name__ == "__main__":
    model = load_model()
    while True:
        print("\n" + "="*40)
        filename = input("🎵 Enter filename (q to quit): ").strip('"')
        if filename.lower() == 'q': break
        if not os.path.exists(filename):
            print("⚠️ File not found."); continue
            
        label, conf = predict_smart(filename, model)
        print("-" * 30)
        print(f"📁 File: {filename}")
        print(f"🤖 Prediction: {label}")
        print(f"📊 Confidence: {conf:.2f}%")
        print("="*40)