import torch
from PIL import Image
import torchvision.transforms as transforms

IMG_SIZE = 224

# Load TorchScript model
model = torch.jit.load("vit_deepfake_scripted.pt", map_location="cpu")
model.eval()

# Same normalization used during training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1,3,224,224)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_class = probs.max(dim=1)

    label = "FAKE" if pred_class.item() == 1 else "REAL"
    return label, conf.item()*100

# Usage
if __name__ == "__main__":
    img_path = "test8.jpeg"   # change this
    label, confidence = predict(img_path)
    print(f"Prediction: {label} ({confidence:.2f}%)")
