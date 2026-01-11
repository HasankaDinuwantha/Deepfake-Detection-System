import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from PIL import Image
import torchvision.transforms as transforms

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="FaceForge",
    page_icon="🛡️",
    layout="centered"
)

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
# TABS
# ==============================
tabs = st.tabs([
    "🧠 ViT-AD",
    "💧 Watermark Detection",
    "⏱️ Temporal Consistency",
    "🔊 Audio Metadata Import"
])

# =========================================================
# TAB 1 — ViT-AD (TORCHSCRIPT LOGIC)
# =========================================================
with tabs[0]:
    st.subheader("🧠 ViT-AD Analysis")
    st.caption("Frame-level face analysis using Vision Transformer")

    IMG_SIZE = 224
    MODEL_PATH = "vit_deepfake_scripted.pt"

    # -----------------------------
    # SESSION STATE
    # -----------------------------
    if "vit_app_version" not in st.session_state:
        st.session_state["vit_app_version"] = 0

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    @st.cache_resource
    def load_vit_model():
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model

    model = load_vit_model()

    # -----------------------------
    # TRANSFORMS
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # FACE DETECTOR
    # -----------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # -----------------------------
    # PREDICTION FUNCTIONS
    # -----------------------------
    def predict_pil(img: Image.Image):
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        label = "FAKE" if pred.item() == 1 else "REAL"
        return label, conf.item() * 100

    def predict_video(video_path, frame_interval=15):
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prog = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_interval != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.2, 5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces[:1]:
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                results.append(predict_pil(face_pil))

            prog.progress(min(frame_idx / max(total_frames, 1), 1.0))

        cap.release()
        return results

    def summarize_results(results):
        if len(results) == 0:
            return "NO FACE DETECTED", 0.0

        fake = [c for l, c in results if l == "FAKE"]
        real = [c for l, c in results if l == "REAL"]

        if len(fake) > len(real):
            return "FAKE", float(np.mean(fake))
        else:
            return "REAL", float(np.mean(real))

    # -----------------------------
    # SIDEBAR RESET
    # -----------------------------
    with st.sidebar:
        st.title("🛡️ FaceForge")
        st.caption("Media Forensics Control Panel")
        st.divider()
        
        if st.button("🧹 Reset Analysis", key="vit_reset"):
            st.session_state["vit_app_version"] += 1
            for k in list(st.session_state.keys()):
                if k != "vit_app_version":
                    del st.session_state[k]
            st.rerun()

    # -----------------------------
    # FILE UPLOAD
    # -----------------------------
    upload_key = f"vit_video_{st.session_state['vit_app_version']}"
    uploaded_file = st.file_uploader(
        "Upload a video for analysis",
        type=["mp4", "mov", "avi"],
        key=upload_key
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(uploaded_file)

        if st.button("🚀 Analyze Video", use_container_width=True):
            with st.spinner("Running ViT-AD analysis..."):
                results = predict_video(video_path)
                label, confidence = summarize_results(results)

                st.divider()
                if label == "FAKE":
                    st.error(f"🚨 RESULT: FAKE ({confidence:.2f}%)")
                elif label == "REAL":
                    st.success(f"✅ RESULT: REAL ({confidence:.2f}%)")
                else:
                    st.warning("⚠️ No face detected")

                st.caption(f"Frames analyzed: {len(results)}")

        if os.path.exists(video_path):
            os.remove(video_path)

# =========================================================
# TAB 2 — Watermark Detection
# =========================================================
with tabs[1]:
    st.info("🚧 Watermark detection module coming soon.")

# =========================================================
# TAB 3 — Temporal Consistency (Placeholder)
# =========================================================
with tabs[2]:
    st.info("🚧 Temporal consistency analysis module coming soon.")

# =========================================================
# TAB 4 — Audio Metadata Import
# =========================================================
with tabs[3]:
    st.info("🚧 Audio metadata & codec forensics coming soon.")
