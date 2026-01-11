import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import tempfile
import torch
import torch.nn as nn
import librosa

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="FaceForge",
    page_icon="🛡️",
    layout="centered"
)

# ==============================
# GLOBAL SESSION STATE INIT (FIX)
# ==============================
if "app_version" not in st.session_state:
    st.session_state["app_version"] = 0

# ==============================
# HEADER (LOGO + TITLE SAME LINE)
# ==============================
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    st.image("logo.png", width=160)

with col_title:
    st.markdown(
        """
        <h1 style="margin-bottom:0;">FaceForge</h1>
        <p style="margin-top:0; color:gray;">
            AI-powered Media Forensics Platform
        </p>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ==============================
# GLOBAL SIDEBAR (ALWAYS VISIBLE)
# ==============================
with st.sidebar:
    st.title("🛡️ FaceForge")
    st.caption("Media Forensics Control Panel")
    st.divider()

    if st.button("🧹 Reset Analysis"):
        st.session_state["app_version"] += 1
        for key in list(st.session_state.keys()):
            if key != "app_version":
                del st.session_state[key]
        st.rerun()

# ==============================
# TABS
# ==============================
tabs = st.tabs([
    "🧠 ViT-AD",
    "💧 Watermark Detection",
    "⏱️ Temporal Consistency",
    "🔊 Audio Metadata Import"
])

# =========================================================
# TAB 1 — ViT-AD (Placeholder)
# =========================================================
with tabs[0]:
    st.info("🚧 ViT-based Anomaly Detection module coming soon.")

# =========================================================
# TAB 2 — Watermark Detection (Placeholder)
# =========================================================
with tabs[1]:
    st.info("🚧 Watermark detection module coming soon.")

# =========================================================
# TAB 3 — Temporal Consistency (Placeholder)
# =========================================================
with tabs[2]:
    st.info("🚧 Temporal consistency analysis module coming soon.")

# =========================================================
# TAB 4 — AUDIO METADATA IMPORT (FULL ML LOGIC)
# =========================================================
with tabs[3]:
    st.subheader("🔊 Audio Metadata & Deepfake Detection")
    st.caption("Detects AI-generated or manipulated speech using spectral analysis")

    # ------------------------------
    # MODEL ARCHITECTURE
    # ------------------------------
    class AudioDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.net(x)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AUDIO_MODEL_PATH = "audio_deepfake_detector.pth"

    # ------------------------------
    # LOAD MODEL
    # ------------------------------
    @st.cache_resource
    def load_audio_model():
        if not os.path.exists(AUDIO_MODEL_PATH):
            st.error(f"❌ Model not found: {AUDIO_MODEL_PATH}")
            st.stop()

        model = AudioDetector().to(DEVICE)
        state_dict = torch.load(AUDIO_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    audio_model = load_audio_model()

    # ------------------------------
    # PREPROCESSING
    # ------------------------------
    def preprocess_tensor(y, sr):
        target_len = 16000 * 4
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=80, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        norm_mel = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return torch.tensor(norm_mel).float().unsqueeze(0).unsqueeze(0)

    def run_inference(tensor):
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            logits = audio_model(tensor)
            return torch.softmax(logits, dim=1)[0][1].item()

    # ------------------------------
    # SMART AUDIO PREDICTION
    # ------------------------------
    def predict_smart(audio_path):
        SR = 16000
        WINDOW = 4.0
        STRIDE = 2.0
        TARGET_LEN = int(SR * WINDOW)

        y_raw, sr = librosa.load(audio_path, sr=SR)
        y_trimmed, _ = librosa.effects.trim(y_raw, top_db=20)
        y = y_trimmed if len(y_trimmed) > 1000 else y_raw

        total_len = len(y)

        if total_len < TARGET_LEN:
            score = run_inference(preprocess_tensor(y, sr))
            return ("FAKE (AI)", score * 100) if score > 0.5 else ("REAL (Human)", (1-score) * 100)

        probs = []
        start = 0
        while start + TARGET_LEN <= total_len:
            chunk = y[start:start + TARGET_LEN]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.005:
                probs.append(run_inference(preprocess_tensor(chunk, sr)))
            start += int(SR * STRIDE)

        if not probs:
            return "ERROR", 0.0

        avg_score = np.mean(probs)
        max_score = max(probs)
        suspicious = sum(1 for p in probs if p > 0.90)

        if max_score > 0.95:
            return "FAKE (AI)", max_score * 100
        if avg_score > 0.55:
            return "FAKE (AI)", avg_score * 100
        if suspicious >= 2 and suspicious / len(probs) > 0.3:
            return "FAKE (AI)", max_score * 100

        return "REAL (Human)", (1 - avg_score) * 100

    # ------------------------------
    # FILE UPLOAD
    # ------------------------------
    upload_key = f"audio_{st.session_state['app_version']}"
    uploaded_audio = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac"],
        key=upload_key
    )

    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name

        st.audio(uploaded_audio)

        if st.button("🚀 Analyze Audio", use_container_width=True):
            with st.spinner("Analyzing audio authenticity..."):
                label, confidence = predict_smart(audio_path)

                st.divider()
                if "FAKE" in label:
                    st.error(f"🚨 RESULT: {label} ({confidence:.2f}%)")
                elif "REAL" in label:
                    st.success(f"✅ RESULT: {label} ({confidence:.2f}%)")
                else:
                    st.warning("⚠️ Analysis failed")

        if os.path.exists(audio_path):
            os.remove(audio_path)
