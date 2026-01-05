import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

IMG_SIZE = 224

# -----------------------------
# Load TorchScript model
# -----------------------------
model = torch.jit.load("vit_deepfake_scripted.pt", map_location="cpu")
model.eval()

# -----------------------------
# Transforms (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# Face detector (OpenCV Haar)
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Single image prediction
# -----------------------------
def predict_pil(img: Image.Image):
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    label = "FAKE" if pred.item() == 1 else "REAL"
    return label, conf.item() * 100


# -----------------------------
# Video prediction
# -----------------------------
def predict_video(
    video_path,
    frame_interval=15,   # analyze every Nth frame
    max_faces_per_frame=1
):
    cap = cv2.VideoCapture(video_path)

    results = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces[:max_faces_per_frame]:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            label, conf = predict_pil(face_pil)
            results.append((label, conf))

    cap.release()
    return results


# -----------------------------
# Aggregate video results
# -----------------------------
def summarize_results(results):
    if len(results) == 0:
        return "NO FACE DETECTED", 0.0

    fake_scores = [conf for lbl, conf in results if lbl == "FAKE"]
    real_scores = [conf for lbl, conf in results if lbl == "REAL"]

    if len(fake_scores) > len(real_scores):
        return "FAKE", np.mean(fake_scores)
    else:
        return "REAL", np.mean(real_scores)


# -----------------------------
# Usage
# -----------------------------
if __name__ == "__main__":
    video_path = "newvid02.mp4"

    results = predict_video(video_path)
    label, confidence = summarize_results(results)

    print(f"Video Prediction: {label} ({confidence:.2f}%)")
    print(f"Frames analyzed: {len(results)}")
